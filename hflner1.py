'''
    医疗领域命名实体识别
'''

from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, Dropout, CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
import torch, os
import numpy as np

checkpoint = "bert-base-chinese"
device = 'cuda'

mycheckpoint = "hflner1"
if not os.path.exists(mycheckpoint):
    os.makedirs(mycheckpoint)

tokenizer = BertTokenizer.from_pretrained(checkpoint)
label2id = {
    'O': 0,
    'TREATMENT-I': 1,
    'TREATMENT-B': 2,
    'BODY-B': 3,
    'BODY-I': 4,
    'SIGNS-I': 5,
    'SIGNS-B': 6,
    'CHECK-B': 7,
    'CHECK-I': 8,
    'DISEASE-I': 9,
    'DISEASE-B': 10
}
id2label = {v: k for k, v in label2id.items()}

class MyDataset(Dataset):
    '''
    从txt文件读取文本数据
    '''

    def __init__(self, originalnerfile):
        self.nerdata = []

        with open(originalnerfile, "r", encoding="utf-8") as fr:
            words = []
            labels = []

            for line in tqdm(fr):
                line = line.rstrip().split("\t")

                char = line[0]
                label = line[1]

                if char not in tokenizer.vocab.keys():
                    words.append('[UNK]')
                else:
                    words.append(char)

                labels.append(label)

                if char == "。":
                    self.nerdata.append((words, labels, len(words)))
                    words = []
                    labels = []

    def __len__(self):
        return len(self.nerdata)

    def __getitem__(self, idx):
        data = self.nerdata[idx]

        return data


def collate_fn(data):
    '''
    整理batch数据并编码
    :param data: 输入batch文本数据
    :return: 返回编码数据
    '''
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    ls = [i[2] for i in data]

    sdata = tokenizer.batch_encode_plus(sents,
                                        padding=True,
                                        return_tensors="pt",
                                        is_split_into_words=True)
    lmax = np.max(ls)

    labels2id = []

    for i in range(len(ls)):
        pad = (lmax - ls[i]) * [0]
        labels2id.append(torch.tensor([label2id[lb] for lb in labels[i]] + pad))

    ldata = torch.stack(labels2id).to(device)

    return sdata.to(device), ldata


class MYNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = Dropout(self.config.hidden_dropout_prob)
        self.fc = Linear(self.config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, inputs):
        out = self.bert(**inputs)

        out = self.dropout(out.last_hidden_state)

        out = self.fc(out)

        return out


def loss_compute(loss, input_ids, attention_mask):
    '''
    考虑无效token如pad、cls、sep，校正损失
    :param loss: 原损失
    :param input_ids: 输入tokens的id
    :param attention_mask: 消除无效token影响的屏蔽矩阵
    :return: 经过校正的损失
    '''

    new_attention_mask = attention_mask * torch.logical_not(torch.eq(input_ids, 102))
    new_attention_mask = new_attention_mask[:, 1:-1]
    loss = torch.sum(loss * new_attention_mask) / torch.sum(new_attention_mask)

    return loss


def TP_compute(predict, label, input_ids, attention_mask):
    '''
    计算TP、FN、FP
    :param predict: 预测结果
    :param label: 真实标注
    :param input_ids: 输入tokens的id
    :param attention_mask: 消除无效token影响的屏蔽矩阵
    :return: TP、FN、FP
    '''

    new_attention_mask = attention_mask * torch.logical_not(torch.eq(input_ids, 102))

    new_attention_mask = new_attention_mask[:, 1:-1]

    # 为实体，预测也为该实体
    tp = torch.sum(torch.logical_and(torch.gt(label, 0), torch.eq(predict, label)))

    # 为实体，预测错误
    fn = torch.sum(torch.logical_and(torch.gt(label, 0), torch.logical_not(torch.eq(predict, label))))

    # 非实体，预测为实体
    fp = torch.sum(torch.logical_and(torch.eq(label, 0), torch.gt(predict * new_attention_mask, 0)))

    return tp, fn, fp


def train():
    dataset = MyDataset("D:/pythonwork/NERs/medical/data/OriginalFiles/train.txt")
    dataset_train, dataset_val = random_split(dataset, [0.9, 0.1])

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=8,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=8,
                                shuffle=False,
                                collate_fn=collate_fn)

    model = MYNER.from_pretrained(checkpoint,
                                  id2label=id2label,
                                  label2id=label2id)

    model.save_pretrained(mycheckpoint)
    tokenizer.save_pretrained(mycheckpoint)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1.0e-6)
    num_epochs = 5
    num_training_steps = num_epochs * len(dataloader_train)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=0,
    )
    criterion = CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):
        total_loss = 0.
        model.train()

        for sen, label in tqdm(dataloader_train):
            model.zero_grad()

            outputs = model(sen)

            input = outputs[:, 1:-1]
            input = input.transpose(1, 2)

            loss = criterion(input, label)

            loss = loss_compute(loss, sen["input_ids"], sen["attention_mask"])

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()

        avg_train_loss = total_loss / len(dataloader_train)

        model.eval()

        tp = 0
        fn = 0
        fp = 0

        for sen, label in tqdm(dataloader_val):
            with torch.no_grad():
                outputs = model(sen)

            out = outputs.argmax(dim=2)

            predict = out[:, 1:-1]

            tpb, fnb, fpb = TP_compute(predict, label, sen["input_ids"], sen["attention_mask"])

            tp += tpb.item()
            fn += fnb.item()
            fp += fpb.item()

        precision = tp / (tp + fp + 1.0e-6)
        recall = tp / (tp + fn + 1.0e-6)
        f1 = 2 * precision * recall / (precision + recall)

        print("\nepoch: ", epoch + 1,
              " loss: ", avg_train_loss,
              " eval_precision: ", precision,
              " eval_recall: ", recall,
              " eval_f1: ", f1,
              )

        model.save_pretrained(mycheckpoint)


def inference(sentence):
    tokenizer = BertTokenizer.from_pretrained(mycheckpoint)

    sen = [word if word in tokenizer.vocab.keys() else '[UNK]' for word in sentence]

    data = tokenizer.batch_encode_plus([sen], is_split_into_words=True, return_tensors="pt")

    model = MYNER.from_pretrained(mycheckpoint)

    data.to(device)
    model.to(device)

    model.eval()

    out = model(data)
    predict = out.argmax(dim=2)[0][1:-1]

    print('--------------------------------------------------------')
    for i, w in enumerate(sentence):
        print(w, "   ", model.config.id2label[predict[i].item()])


if __name__ == "__main__":
    # train()

    inference("患者精神状况好，无发热，诉右髋部疼痛，饮食差，二便正常。")
    inference("一般情况可，心肺未见异常。腹平坦，未见胃肠型及蠕动波。上腹部压痛阳性，无反跳痛及肌紧张。肝脾肋下未触及，全腹未触及 异常包块。腹叩移动性浊音阴性，肠鸣音正常，未闻及高调肠鸣音及气过水声。")
    inference("我肚子有点疼痛。")
