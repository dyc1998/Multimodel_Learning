import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # 导入所需要的库
import sys


def F1_score(pre, recall):
    return 2 * pre * recall / (pre + recall)


def train_one_epoch_A(model, device, data_loader, loss_f, optimizer, epoch):
    model.train()
    scalar = GradScaler()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    recall_pos = torch.zeros(1).to(device)  # recall所需要的分子
    recall_num = torch.zeros(1).to(device)  # 计算recall所需要的分母
    pre_num=torch.zeros(1).to(device)  # precision所需要的分母
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, colour='YELLOW', ncols=120)
    for step, data in enumerate(data_loader):
        wav, label = data
        wav = wav.to(device)
        label = label.to(device)
        sample_num += label.shape[0]
        recall_num += torch.sum(label == 1)
        with autocast():
            pred = model(wav)[0]  # 因为代码里有这个feature
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, label).sum()
            recall_pos += torch.sum((pred_classes == 1) & (label == 1))
            pre_num+=torch.sum((pred_classes==1))
            loss = loss_f(pred, label)
            loss = (loss-0.2).abs()+0.1
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        accu_loss += loss.detach()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        optimizer.zero_grad()
        l=accu_loss.item() / (step + 1)
        acc=accu_num.item() / sample_num
        if(pre_num.item()==0):
            continue
        precision=recall_pos.item()/(pre_num.item())
        recall=recall_pos.item() / recall_num.item()
        f1=F1_score(precision,recall)
        data_loader.desc = "[A_train_epoch {}]  loss:{:.3f}, acc:{:.3f},pre:{:.3f}, recall:{:.3f},  F1-score:{:.3f}".format(epoch,l,acc,precision,recall,f1)
    return l, acc, precision,recall, f1


@torch.no_grad()
def eval_one_epoch_A(model, data_loader, device, loss_f, epoch):
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    recall_pos = torch.zeros(1).to(device)  # recall所需要的分子
    recall_num = torch.zeros(1).to(device)  # 计算recall所需要的分母
    pre_num = torch.zeros(1).to(device)  # precision所需要的分母
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, ncols=120)
    for step, data in enumerate(data_loader):
        wav, label = data
        wav = wav.to(device)
        label = label.to(device)
        sample_num += label.shape[0]
        recall_num += torch.sum(label == 1)
        pred = model(wav)[0]
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label).sum()
        recall_pos += torch.sum((label == 1) & (pred_classes == 1))
        pre_num+=torch.sum(pred_classes==1)
        loss = loss_f(pred, label)
        accu_loss += loss.item()

        l = accu_loss.item() / (step + 1)
        acc = accu_num.item() / sample_num
        precision = recall_pos.item() / pre_num.item()
        recall = recall_pos.item() / recall_num.item()
        f1 = F1_score(precision, recall)
        data_loader.desc = "[A_eval_epoch {}]  loss:{:.3f}, acc:{:.3f},pre:{:.3f}, recall:{:.3f},  F1-score:{:.3f}".format(epoch, l, acc, precision, recall, f1)
    return l, acc, precision, recall, f1