'''
    基于W2NER的命名实体识别统一框架
    可以处理FLAT、NESTED、DISCONTINUED三种

    梯度累加
    梯度按序列长度比率算术平均

    lora + 半精度float16

    自定义for循环训练
    NEFTTUNE写在for循环里（效果差）

    注意：自定义模型使用lora时使用torch.save保存模型权重model.state_dict()，不要使用model.save_pretrained，
    因为此时是peft保存模型的lora化部分，新添加的参数无法保存。
'''

from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BitsAndBytesConfig
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torch.nn import Linear, Dropout
from torch.optim import AdamW
from bitsandbytes.optim import AdamW8bit, PagedAdamW8bit
from tqdm import tqdm
import torch, os
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig, prepare_model_for_kbit_training
from utils import format_time, create_logger, NEFTune
from dataclasses import dataclass
import argparse
import random
from time import time
import warnings

warnings.filterwarnings("ignore")

checkpoint = "bert-base-chinese"
device = 'cuda'

mycheckpoint = "models/hflw2ner82"
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

batch_size = 16
accum_steps = 4

num_epochs = 20
lr = 5e-3

noise_alpha = 5.0


class MyDataset(Dataset):
    '''
    从txt文件读取文本数据
    '''

    def __init__(self, filelisttxt):
        self.filelist = []

        with open(filelisttxt, "r", encoding="utf-8") as fr:
            for line in fr:
                self.filelist.append(line.strip())

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

    def forward(self, input_ids, attention_mask, seqlen, input_embeds=None, labels=None):
        out = self.bert(input_ids, attention_mask, inputs_embeds=input_embeds)

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

            loss = torch.sum(loss)

            # 为实体，预测也为该实体
            tp = torch.sum(torch.logical_and(torch.gt(labels, 0), torch.eq(predict, labels)))

            # 为实体，预测错误
            fn = torch.sum(torch.logical_and(torch.gt(labels, 0), torch.logical_not(torch.eq(predict, labels))))

            # 非实体，预测为实体
            fp = torch.sum(torch.logical_and(torch.eq(labels, 0), torch.gt(predict, 0)))

            return {"loss": loss, "predict": predict, "tp": tp, "fn": fn, "fp": fp}
        else:
            return {"predict": predict}


def train():
    logger = create_logger(name="train_log",
                           filename=mycheckpoint + "/hflw2ner.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Train Logging......")
    logger.info(
        "基于Lora微调BERT实现医疗NER任务，超参设置 --mode train --num_epochs %d --lr %e --batch_size %d --accum_steps %d --model_path %s --pretrained_checkpoint %s" % (
            num_epochs, lr, batch_size, accum_steps, mycheckpoint, checkpoint))

    logger.info("开始创建分词器...")
    tokenizer = BertTokenizer.from_pretrained(checkpoint)

    logger.info("开始读取数据...")
    dataset_train = MyDataset("data/train_filelist.txt")
    dataset_val = MyDataset("data/val_filelist.txt")

    data_collator = MyDataCollator(tokenizer)

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size * accum_steps,
                                  shuffle=True,
                                  collate_fn=data_collator,
                                  drop_last=True)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=data_collator)

    logger.info("开始创建模型...")
    model = MYW2NER.from_pretrained(checkpoint,
                                    id2label=id2label,
                                    label2id=label2id,
                                    torch_dtype=torch.bfloat16,
                                    )
    model.gradient_checkpointing_enable()

    logger.info("设置lora参数...")
    lora_config = LoraConfig(r=10)
    logger.info(lora_config)

    logger.info("模型转化为peft model...")
    model = get_peft_model(model, lora_config)
    logger.info(lora_config)

    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    model.to(device)

    paras_bert = []
    paras_last = []

    for k, v in dict(model.named_parameters()).items():
        if "bert" in k:
            paras_bert += [{'params': [v]}]
        else:
            paras_last += [{'params': [v]}]

    for k, v in model.named_parameters():
        if "fc" in k:
            v.requires_grad = True

    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    oneepoch_stepping_steps = len(dataloader_train)
    num_stepping_steps = num_epochs * oneepoch_stepping_steps
    oneepoch_training_step = oneepoch_stepping_steps * accum_steps

    optimizerbert = AdamW(
        paras_bert,
        lr=lr,
        eps=1.0e-6,
    )
    lr_schedulerbert = get_scheduler(
        name="linear",
        optimizer=optimizerbert,
        num_training_steps=num_stepping_steps,
        num_warmup_steps=num_stepping_steps // 10,
    )

    optimizerlast = AdamW(
        paras_last,
        lr=10. * lr,
        eps=1.0e-6,
    )
    lr_schedulerlast = get_scheduler(
        name="linear",
        optimizer=optimizerlast,
        num_training_steps=num_stepping_steps,
        num_warmup_steps=num_stepping_steps // 10,
    )

    btime = time()
    for epoch in range(num_epochs):
        total_loss = 0.
        model.train()

        for batch, data in enumerate(dataloader_train):
            optimizerbert.zero_grad()
            optimizerlast.zero_grad()
            model.zero_grad()

            seqlen2sum = torch.sum(torch.square(data["seqlen"]), dtype=torch.float)

            for k in range(accum_steps):
                miniseqlen = data["seqlen"][k * batch_size:(k + 1) * batch_size].to(device)
                minimaxls = miniseqlen.max()

                miniinput_ids = data["input_ids"][k * batch_size:(k + 1) * batch_size, :minimaxls + 2].to(device)
                miniattention_mask = data["attention_mask"][k * batch_size:(k + 1) * batch_size, :minimaxls + 2].to(
                    device)

                minilabels = data["labels"][k * batch_size:(k + 1) * batch_size, :minimaxls, :minimaxls].to(device)

                # NEFTUNE: 对嵌入加入均匀分布噪声
                miniembeds = model.bert.embeddings.word_embeddings.forward(miniinput_ids)
                dims = torch.tensor(miniembeds.size(1) * miniembeds.size(2))
                mag_norm = noise_alpha / torch.sqrt(dims)
                miniembeds = miniembeds + torch.zeros_like(miniembeds).uniform_(-mag_norm, mag_norm)

                res = model(None,
                            miniattention_mask,
                            miniseqlen,
                            miniembeds,
                            minilabels,
                            )

                miniseqlen2sum = torch.sum(torch.square(miniseqlen), dtype=torch.float)

                losst = res["loss"] / miniseqlen2sum
                total_loss += losst.item()

                loss = res["loss"] / seqlen2sum

                loss.backward()

                print("\repoch: %d  %d|%d  loss: %f " % (
                    epoch + 1, batch * accum_steps + k + 1, oneepoch_training_step, losst.item()), end="")

            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2)

            optimizerbert.step()
            optimizerlast.step()

            lr_schedulerbert.step()
            lr_schedulerlast.step()

            optimizerbert.zero_grad()
            optimizerlast.zero_grad()
            model.zero_grad()

        print()
        avg_train_loss = total_loss / oneepoch_training_step

        model.eval()

        tp = 0
        fn = 0
        fp = 0

        for data in dataloader_val:
            with torch.no_grad():
                res = model(data["input_ids"].to(device),
                            data["attention_mask"].to(device),
                            data["seqlen"].to(device),
                            labels=data["labels"].to(device),
                            )

            tp += res["tp"].item()
            fn += res["fn"].item()
            fp += res["fp"].item()

        precision = tp / (tp + fp + 1.0e-6)
        recall = tp / (tp + fn + 1.0e-6)
        f1 = 2 * precision * recall / (precision + recall + 1.0e-6)

        logger.info({"epoch": epoch + 1,
                     "loss": avg_train_loss,
                     "precision": precision,
                     "recall": recall,
                     "f1": f1,
                     }
                    )

        torch.save(model.state_dict(), mycheckpoint + "/pytorch_model.bin")

    etime = time()

    logger.info("耗时：" + format_time(etime - btime))


def inference():
    logger = create_logger(name="infer_log",
                           filename=mycheckpoint + "/hflw2ner.log")
    logger.info(
        "------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Infer Logging......")

    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    model = MYW2NER.from_pretrained(checkpoint,
                                    id2label=id2label,
                                    label2id=label2id,
                                    )

    lora_config = LoraConfig(r=10)

    model = get_peft_model(model, lora_config)

    params = torch.load(mycheckpoint + "/pytorch_model.bin")

    model.load_state_dict(params)

    model = model.bfloat16()

    model.to(device)

    model.eval()

    intent = True

    while intent:
        sentence = input("文本：")

        if sentence == "":
            intent = False
            continue

        sen = [word if word in tokenizer.vocab.keys() else '[UNK]' for word in sentence]
        lsen = len(sen)

        data = tokenizer.batch_encode_plus([sen],
                                           is_split_into_words=True,
                                           return_attention_mask=True,
                                           return_token_type_ids=False,
                                           return_tensors="pt")
        ls = torch.tensor([lsen])

        data.to(device)
        ls = ls.to(device)

        predict = model(data["input_ids"], data["attention_mask"], ls)["predict"][0]

        labels = ["O"] * lsen

        for i in range(lsen):
            for j in range(i):
                if predict[i, j] > 0:
                    res = id2label[predict[i, j].item()]
                    labels[j:i + 1] = [res + "-B"] + [res + "-I"] * (i - j)

        res = '\n\t\t' + sentence + "\n\n"
        for i in range(lsen):
            res += '\t\t' + sentence[i] + '\t' + labels[i] + "\n"

        logger.info(res)


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
