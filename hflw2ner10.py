'''
    基于W2NER的命名实体识别统一框架
    可以处理FLAT、NESTED、DISCONTINUED三种

    Qlora -4bit + 半精度float16

'''

from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BitsAndBytesConfig
from transformers import get_scheduler, TrainerCallback, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torch.nn import Linear, Dropout
from torch.optim import AdamW
from bitsandbytes.optim import AdamW8bit
from tqdm import tqdm
import torch, os
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig, prepare_model_for_kbit_training
import warnings
from dataclasses import dataclass
import time
from time import strftime, gmtime
import logging
import argparse
from utils import format_time,create_logger
import random

warnings.filterwarnings("ignore")

checkpoint = "bert-base-chinese"
device = 'cuda'

mycheckpoint = "models/hflw2ner10"
if not os.path.exists(mycheckpoint):
    os.makedirs(mycheckpoint)

tokenizer = BertTokenizer.from_pretrained(checkpoint)
label2id = {
    'NONE': 0,
    'NNW': 1,
    '治疗': 2,
    '身体部位': 3,
    '症状和体征': 4,
    '检查和检验': 5,
    '疾病和诊断': 6
}
id2label = {v: k for k, v in label2id.items()}

txt_max_len = 200

batch_size = 8
accum_steps = 2

num_epochs = 20
lr = 5e-3


class MyDataset(Dataset):
    '''
    从txt文件读取文本数据
    '''

    def __init__(self, originaldir):
        self.filelist = []

        for dirname in os.listdir(originaldir):
            first_path = os.path.join(originaldir, dirname)

            for filename in os.listdir(first_path):
                if "txtoriginal" in filename:
                    file_path = os.path.join(first_path, filename)
                    with open(file_path, "r", encoding="utf-8") as fr:
                        text_title = fr.readline().rstrip()
                        lt = len(text_title)
                        if lt > 0:
                            self.filelist.append(file_path)

        random.shuffle(self.filelist)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        data = self.filelist[idx]

        return data


@dataclass
class MyDataCollator:
    tokenizer: BertTokenizer

    def __call__(self, filenames):
        '''
        整理batch数据并编码
        :param filenames: 输入batch文件名
        :return: 返回编码数据
        '''

        datasen = []

        datalab = []
        for file in filenames:
            with open(file, "r", encoding="utf-8") as fr:
                sent = fr.readline().rstrip()[:txt_max_len]
                datasen.append(sent)

                labelfile = file.replace("txtoriginal.", "")

                lab = dict()
                with open(labelfile, "r", encoding="utf-8") as fr:
                    result = fr.readlines()
                    if len(result) > 0:
                        for re in result:
                            _, startid, endid, label = re.rstrip().split("\t")
                            if int(startid) >= txt_max_len:
                                break
                            else:
                                lab[startid + ";" + str(min(int(endid), txt_max_len - 1))] = label
                datalab.append(lab)

        datasen = [[j if j in tokenizer.vocab.keys() else "[UNK]" for j in ds] for ds in datasen]
        ls = [len(i) for i in datasen]
        batch_size = len(ls)

        sdata = tokenizer.batch_encode_plus(datasen,
                                            padding=True,
                                            return_tensors="pt",
                                            return_attention_mask=True,
                                            return_token_type_ids=False,
                                            is_split_into_words=True,
                                            truncation=True,
                                            max_length=txt_max_len + 2)

        lmax = np.max(ls)
        ldata = torch.zeros(batch_size, lmax, lmax, dtype=torch.int)

        for i, lb in enumerate(datalab):
            if len(lb) == 0:
                continue

            for k, v in lb.items():
                startid, endid = k.split(";")
                ldata[i, int(endid), int(startid)] = label2id[v]

                for id in range(int(startid), int(endid)):
                    ldata[i, id, id + 1] = label2id["NNW"]

        return {
            "input_ids": sdata["input_ids"],
            "attention_mask": sdata["attention_mask"],
            "seqlen": torch.tensor(ls),
            "labels": ldata,
        }


def sequence_mask(lengths, max_len=None):
    lengths_shape = lengths.shape
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)


def tri2(x, y, N, num_lables):
    a = sequence_mask(torch.arange(1, N + 1), max_len=N).to(device)
    aa = torch.tile(torch.unsqueeze(a, dim=2), [1, 1, num_lables])

    bb = aa * y[0]

    aaa = torch.tile(torch.unsqueeze(torch.logical_not(a), dim=2), [1, 1, num_lables])

    bbb = aaa * y[1]

    xx = x + bb + bbb

    return xx


def focal_loss(y_true, y_pred, gamma=2.0):
    """
    Focal Loss 针对样本不均衡
    :param y_true: 样本标签 B*N*N
    :param y_pred: 预测值（softmax） B*N*N*n_class
    :return: focal loss
    """

    batch_size, seq_len, _, n_class = y_pred.shape

    softmax = y_pred.reshape([-1])

    labels = y_true.reshape([-1])
    labels = torch.arange(0, batch_size * seq_len * seq_len).to(device) * n_class + labels

    prob = torch.gather(softmax, 0, labels)

    weight = torch.pow(1. - prob, gamma)

    loss = -torch.multiply(weight, torch.log(prob))

    loss = torch.reshape(loss, [batch_size, seq_len, seq_len])

    return loss


class MYW2NER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = Dropout(self.config.hidden_dropout_prob)
        self.fc = Linear(self.config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask, seqlen, labels=None):
        out = self.bert(input_ids, attention_mask)

        out = self.dropout(out.last_hidden_state)

        out = out[:, 1:-1]

        N = torch.max(seqlen)

        x1 = torch.tile(torch.unsqueeze(out, dim=2), [1, 1, N, 1])
        x2 = x1.permute(0, 2, 1, 3)

        xx = x1 * x2

        logits = self.fc(xx)

        t1 = torch.tensor([[0.0, 1.0], [0.0, 0.0]]).to(device)
        t2 = torch.concat((torch.zeros(1, self.config.num_labels - 2), torch.ones(1, self.config.num_labels - 2)),
                          dim=0).to(device)
        t = torch.concat((t1, t2), dim=1)
        y = (1. - 2. ** (31.)) * t

        logits = tri2(logits, y, N, self.config.num_labels)

        predict = logits.argmax(dim=-1)

        val = sequence_mask(seqlen, max_len=N)

        # B*N*N
        val1 = torch.tile(torch.unsqueeze(val, dim=1), [1, N, 1])
        val2 = torch.tile(torch.unsqueeze(val, dim=2), [1, 1, N])

        # B*N*N
        val = torch.logical_and(val1, val2)

        predict *= val

        if labels is not None:
            softmax = logits.softmax(dim=-1)
            loss = focal_loss(labels, softmax)
            loss *= val
            seqlen2sum = torch.sum(torch.square(seqlen))

            loss = torch.sum(loss) / seqlen2sum

            # 为实体，预测也为该实体
            tp = torch.sum(torch.logical_and(torch.gt(labels, 0), torch.eq(predict, labels)))

            # 为实体，预测错误
            fn = torch.sum(torch.logical_and(torch.gt(labels, 0), torch.logical_not(torch.eq(predict, labels))))

            # 非实体，预测为实体
            fp = torch.sum(torch.logical_and(torch.eq(labels, 0), torch.gt(predict, 0)))

            return {"loss": loss, "predict": predict, "tp": tp, "fn": fn, "fp": fp}
        else:
            return {"predict": predict}

class MyTrainCallback(TrainerCallback):
    def __init__(self, mylogger):
        self.mylogger = mylogger

    def on_epoch_end(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        model.eval()
        tp = 0
        fn = 0
        fp = 0

        for data in eval_dataloader:
            with torch.no_grad():
                res = model(data["input_ids"].to(device),
                            data["attention_mask"].to(device),
                            data["seqlen"].to(device),
                            data["labels"].to(device),
                            )

            tp += res["tp"].item()
            fn += res["fn"].item()
            fp += res["fp"].item()

        precision = tp / (tp + fp + 1.0e-6)
        recall = tp / (tp + fn + 1.0e-6)
        f1 = 2 * precision * recall / (precision + recall + 1.0e-6)

        self.mylogger.info({
            "epoch": int(state.epoch),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        model.train()

    def on_train_end(self, args, state, control, **kwargs):
        history = state.log_history

        historyinfo = "\n"
        for loghis in history[:-1]:
            historyinfo += str({
                "epoch": int(loghis["epoch"]),
                "loss": loghis["loss"],
                "learning_rate": loghis["learning_rate"],
            }) + "\n"

        self.mylogger.info(historyinfo)

        loghis = history[-1]
        self.mylogger.info("耗时：" + format_time(loghis['train_runtime']))


def train():
    logger = create_logger(name="train_log",
                           filename=mycheckpoint + "/hflw2ner.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Train Logging......")
    logger.info(
        "基于BERT实现医疗NER任务，超参设置 --mode train --num_epochs %d --lr %e --batch_size %d --accum_steps %d --model_path %s --pretrained_checkpoint %s" % (
            num_epochs, lr, batch_size, accum_steps, mycheckpoint, checkpoint))

    logger.info("开始创建分词器...")
    tokenizer = BertTokenizer.from_pretrained(checkpoint)

    logger.info("开始读取数据...")
    dataset = MyDataset("D:/pythonwork/W2NER/data/OriginalFiles/data_origin")

    # dataset_train, dataset_val = random_split(dataset, [0.9, 0.1])
    dataset_train = dataset[:1024]
    dataset_val = dataset[1024:1024 + 128]

    data_collator = MyDataCollator(tokenizer)

    logger.info("开始创建模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = MYW2NER.from_pretrained(checkpoint,
                                    # quantization_config=bnb_config,
                                    id2label=id2label,
                                    label2id=label2id,
                                    # torch_dtype=torch.float16,
                                    )

    # model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(r=10)
    logger.info(lora_config)
    model = get_peft_model(model, lora_config)
    logger.info(lora_config)

    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    model.to(device)

    for k, v in model.named_parameters():
        if "fc" in k:
            v.requires_grad = True

    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    training_args = TrainingArguments(
        output_dir=mycheckpoint,
        overwrite_output_dir=True,
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        num_train_epochs=num_epochs,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        dataloader_drop_last=False,
        learning_rate=lr,
        weight_decay=1e-2,
        adam_epsilon=1e-6,
        max_grad_norm=1.0,
        save_strategy="no",
        # optim="paged_adamw_8bit",
        fp16=True,
    )

    mytraincallback = MyTrainCallback(logger)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        data_collator=data_collator,
        callbacks=[mytraincallback],
    )

    trainer.callback_handler.eval_dataloader = DataLoader(dataset_val,
                                                          batch_size,
                                                          shuffle=False,
                                                          collate_fn=data_collator)
    logger.info("开始训练...")
    trainer.train()

    logger.info("保存模型")
    trainer.model.save_pretrained(mycheckpoint)


def inference(sentence):
    tokenizer = BertTokenizer.from_pretrained(mycheckpoint)

    model = torch.load(mycheckpoint + "/pytorch_model.bin")

    sen = [word if word in tokenizer.vocab.keys() else '[UNK]' for word in sentence]
    lsen = len(sen)

    data = tokenizer.batch_encode_plus([sen],
                                       is_split_into_words=True,
                                       return_token_type_ids=False,
                                       return_tensors="pt")
    ls = torch.tensor([lsen])

    data.to(device)
    ls = ls.to(device)

    model.to(device)

    model.eval()

    predict = model(data, ls)[0]

    print(sentence)

    # for i in range(lsen):
    #     for j in range(i):
    #         if predict[i, j] > 0:
    #             print(''.join([sen[k] for k in range(j, i + 1)]), id2label[predict[i, j].item()])

    labels = ["O"] * lsen

    for i in range(lsen):
        for j in range(i):
            if predict[i, j] > 0:
                res = id2label[predict[i, j].item()]
                labels[j:i + 1] = [res + "-B"] + [res + "-I"] * (i - j)

    for i in range(lsen):
        print(sentence[i], '\t', labels[i])
    print('\n------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, required=True)

    pargs = parser.parse_args()

    if pargs.mode == "train":
        train()
    elif pargs.mode == "infer":
        inference()

    # "患者精神状况好，无发热，诉右髋部疼痛，饮食差，二便正常。"
    # "腹叩移动性浊音阴性，肠鸣音正常，未闻及高调肠鸣音及气过水声。"
    # "我腹部有点疼痛。"
    # "韩凤科 男 74岁 汉族 已婚 现住双塔山棋盘地村 主因发作性头痛头晕伴左侧肢体无力1天于2016-1-26  11：06入院。"
