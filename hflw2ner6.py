'''
    基于W2NER的命名实体识别统一框架
    可以处理FLAT、NESTED、DISCONTINUED三种

    梯度累加
    梯度按累积次数平均
'''

from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torch.nn import Linear, Dropout
from torch.optim import AdamW
from tqdm import tqdm
import torch, os
import numpy as np

checkpoint = "bert-base-chinese"
device = 'cuda'

mycheckpoint = "hflw2ner6"
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

txt_max_len = 400

batch_size = 2
accum_step = 2

num_epochs = 10
lr = 5e-5


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

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        data = self.filelist[idx]

        return data


def collate_fn(filenames):
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

    return sdata.to(device), torch.tensor(ls, dtype=torch.int).to(device), ldata.to(device)


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

    def forward(self, bertinputs, seqlen, span=None):
        out = self.bert(**bertinputs)

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

        if span is not None:
            softmax = logits.softmax(dim=-1)
            loss = focal_loss(span, softmax)
            loss *= val
            seqlen2sum = torch.sum(torch.square(seqlen))
            loss = torch.sum(loss) / seqlen2sum

            # 为实体，预测也为该实体
            tp = torch.sum(torch.logical_and(torch.gt(span, 0), torch.eq(predict, span)))

            # 为实体，预测错误
            fn = torch.sum(torch.logical_and(torch.gt(span, 0), torch.logical_not(torch.eq(predict, span))))

            # 非实体，预测为实体
            fp = torch.sum(torch.logical_and(torch.eq(span, 0), torch.gt(predict, 0)))

            return predict, loss, tp, fn, fp
        else:
            return predict


def train():
    dataset = MyDataset("D:/pythonwork/W2NER/data/OriginalFiles/data_origin")

    dataset_train, dataset_val = random_split(dataset, [0.9, 0.1])

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=collate_fn)

    model = MYW2NER.from_pretrained(checkpoint,
                                    id2label=id2label,
                                    label2id=label2id)
    model.config.__dict__["val_f1"] = []

    model.to(device)
    model.save_pretrained(mycheckpoint)
    tokenizer.save_pretrained(mycheckpoint)

    paras_bert = []
    paras_last = []

    for k, v in dict(model.named_parameters()).items():
        if k.startswith("bert"):
            paras_bert += [{'params': [v]}]
        else:
            paras_last += [{'params': [v]}]

    oneepoch_train_steps = len(dataloader_train)
    oneepoch_stepping_steps = len(dataloader_train) // accum_step
    num_stepping_steps = num_epochs * oneepoch_stepping_steps

    optimizerbert = AdamW(paras_bert, lr=lr, eps=1.0e-6)
    lr_schedulerbert = get_scheduler(
        name="linear",
        optimizer=optimizerbert,
        num_training_steps=num_stepping_steps,
        num_warmup_steps=oneepoch_stepping_steps,
    )

    optimizerlast = AdamW(paras_last, lr=100. * lr, eps=1.0e-6)
    lr_schedulerlast = get_scheduler(
        name="linear",
        optimizer=optimizerlast,
        num_training_steps=num_stepping_steps,
        num_warmup_steps=oneepoch_stepping_steps,
    )

    for epoch in range(num_epochs):
        total_loss = 0.
        model.train()

        optimizerbert.zero_grad()
        optimizerlast.zero_grad()
        model.zero_grad()

        for batch, (sen, ls, span) in enumerate(dataloader_train):
            _, loss, _, _, _ = model(sen, ls, span)

            total_loss += loss.item()

            loss = loss / accum_step
            loss.backward()

            if (batch + 1) % accum_step == 0:
                clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2)

                optimizerbert.step()
                optimizerlast.step()

                lr_schedulerbert.step()
                lr_schedulerlast.step()

                optimizerbert.zero_grad()
                optimizerlast.zero_grad()
                model.zero_grad()

            print("\repoch: %d  %d|%d  loss: %f " % (epoch + 1, batch + 1, oneepoch_train_steps, loss.item()), end="")

        print()

        avg_train_loss = total_loss / oneepoch_train_steps

        model.eval()

        tp = 0
        fn = 0
        fp = 0

        for sen, ls, span in tqdm(dataloader_val):
            with torch.no_grad():
                _, _, tpb, fnb, fpb = model(sen, ls, span)

            tp += tpb.item()
            fn += fnb.item()
            fp += fpb.item()

        precision = tp / (tp + fp + 1.0e-6)
        recall = tp / (tp + fn + 1.0e-6)
        f1 = 2 * precision * recall / (precision + recall + 1.0e-6)

        print("\nepoch: ", epoch + 1,
              " loss: ", avg_train_loss,
              " eval_precision: ", precision,
              " eval_recall: ", recall,
              " eval_f1: ", f1,
              )

        model.config.__dict__["val_f1"].append(f1)

        model.save_pretrained(mycheckpoint)


def inference(sentence):
    tokenizer = BertTokenizer.from_pretrained(mycheckpoint)

    sen = [word if word in tokenizer.vocab.keys() else '[UNK]' for word in sentence]
    lsen = len(sen)

    data = tokenizer.batch_encode_plus([sen],
                                       is_split_into_words=True,
                                       return_token_type_ids=False,
                                       return_tensors="pt")
    ls = torch.tensor([lsen])

    data.to(device)
    ls = ls.to(device)

    model = MYW2NER.from_pretrained(mycheckpoint)
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
    train()
    # inference("患者精神状况好，无发热，诉右髋部疼痛，饮食差，二便正常。")
    # inference("腹叩移动性浊音阴性，肠鸣音正常，未闻及高调肠鸣音及气过水声。")
    # inference("我腹部有点疼痛。")
    # inference("韩凤科 男 74岁 汉族 已婚 现住双塔山棋盘地村 主因发作性头痛头晕伴左侧肢体无力1天于2016-1-26  11：06入院。")
