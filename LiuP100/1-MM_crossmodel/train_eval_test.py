import torch
import torchmetrics
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # 导入所需要的库
import sys


def F1_score(pre, recall):
    return 2 * pre * recall / (pre + recall)


def train_one_epoch(model, device, data_loader, loss_f, optimizer, epoch):
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
        wav, imgs, label = data
        imgs = imgs.to(device)
        wav = wav.to(device)
        label = label.to(device)
        sample_num += label.shape[0]
        recall_num += torch.sum(label == 1)

        visual_inputs = imgs.permute(0, 2, 1, 3, 4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0] * visual_inputs.shape[1], visual_inputs.shape[2],
                                              visual_inputs.shape[3], visual_inputs.shape[4])
        with autocast():
            pred = model(wav, visual_inputs)
            pred_classes = torch.max(pred, dim=1)[1]


            accu_num += torch.eq(pred_classes, label).sum()
            recall_pos += torch.sum((pred_classes == 1) & (label == 1))
            pre_num+=torch.sum((pred_classes==1))

            loss = loss_f(pred, label)
            loss = (loss-0.2).abs()+0.2
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
        if (pre_num.item() == 0):
            continue
        precision=recall_pos.item()/pre_num.item()
        recall=recall_pos.item() / recall_num.item()
        if (pre_num.item() == 0):
            continue
        f1=F1_score(precision,recall)
        data_loader.desc = "[train_epoch {}]  loss:{:.3f}, acc:{:.3f},pre:{:.3f}, recall:{:.3f},  F1-score:{:.3f}".format(epoch,l,acc,precision,recall,f1)
    return l, acc, precision,recall, f1


@torch.no_grad()
def eval_one_epoch(model, data_loader, device, loss_f, epoch):
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    recall_pos = torch.zeros(1).to(device)  # recall所需要的分子
    recall_num = torch.zeros(1).to(device)  # 计算recall所需要的分母
    pre_num = torch.zeros(1).to(device)  # precision所需要的分母
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, ncols=120)
    for step, data in enumerate(data_loader):
        wav, imgs, label = data
        imgs = imgs.to(device)
        wav = wav.to(device)
        label = label.to(device)
        sample_num += label.shape[0]
        recall_num += torch.sum(label == 1)
        visual_inputs = imgs.permute(0, 2, 1, 3, 4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0] * visual_inputs.shape[1], visual_inputs.shape[2],
                                              visual_inputs.shape[3], visual_inputs.shape[4])
        pred = model(wav, visual_inputs)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label).sum()
        recall_pos += torch.sum((label == 1) & (pred_classes == 1))
        pre_num+=torch.sum(pred_classes==1)
        if pre_num.item()==0:
            continue
        loss = loss_f(pred, label)
        accu_loss += loss.item()

        l = accu_loss.item() / (step + 1)
        acc = accu_num.item() / sample_num
        precision = recall_pos.item() / pre_num.item()
        recall = recall_pos.item() / recall_num.item()
        f1 = F1_score(precision, recall)
        data_loader.desc = "[test_epoch {}]  loss:{:.3f}, acc:{:.3f},pre:{:.3f}, recall:{:.3f},  F1-score:{:.3f}".format(epoch, l, acc, precision, recall, f1)
    return l, acc, precision, recall, f1


def train_one_epoch_AV(device, data_loader, loss_f, optimizer_a, optimizer_v, epoch, model_audio, model_video,
                        model_multi=None):
    model_video.train()
    model_audio.train()
    scalar = GradScaler()
    train_loss = torch.zeros(3).unsqueeze(1).to(device)
    accu_num = torch.zeros(3).unsqueeze(1).to(device)  # [audio,video,multi]
    recall_pos = torch.zeros(3).unsqueeze(1).to(device)  # recall所需要的分子

    optimizer_a.zero_grad()
    optimizer_v.zero_grad()
    sample_num = 0
    recall_num = 0  # 标签为1的个数
    data_loader = tqdm(data_loader, file=sys.stdout, colour='YELLOW', ncols=150)
    for step, data in enumerate(data_loader):
        wav, imgs, label = data
        imgs = imgs.to(device)
        wav = wav.to(device)
        label = label.to(device)

        sample_num += label.shape[0]
        recall_num += torch.sum(label == 1).item()

        visual_inputs = imgs.permute(0, 2, 1, 3, 4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0] * visual_inputs.shape[1], visual_inputs.shape[2],
                                              visual_inputs.shape[3], visual_inputs.shape[4])
        with autocast():
            pred_a = model_audio(wav)
            pred_v = model_video(visual_inputs)
            # pred_m = model_multi(wav, visual_inputs)
            pred_classes_a = torch.max(pred_a, dim=1)[1]
            pred_classes_v = torch.max(pred_v, dim=1)[1]
            # pred_classes_m = torch.max(pred_m, dim=1)[1]

            accu_num[0] += torch.eq(pred_classes_a, label).sum()
            accu_num[1] += torch.eq(pred_classes_v, label).sum()
            # accu_num[2] += torch.eq(pred_classes_m, label).sum()
            recall_pos[0] += torch.sum((pred_classes_a == 1) & (label == 1))
            recall_pos[1] += torch.sum((pred_classes_v == 1) & (label == 1))
            # recall_pos[2] += torch.sum((pred_classes_m == 1) & (label == 1))
            loss_a = loss_f(pred_a, label)
            loss_v = loss_f(pred_v, label)
            # loss_m = loss_f(pred_m, label)
        if not torch.isfinite(loss_a):
            print('WARNING: non-finite loss, ending training ', loss_a)
            sys.exit(1)
        train_loss[0] += loss_a.detach()
        train_loss[1] += loss_v.detach()
        # train_loss[2] += loss_m.detach()
        scalar.scale(loss_a).backward()
        scalar.scale(loss_v).backward()
        scalar.step(optimizer_a)
        scalar.step(optimizer_v)
        scalar.update()
        optimizer_a.zero_grad()
        optimizer_v.zero_grad()

        data_loader.desc = "[train_epoch {}] , A_acc:{:.3f}, A_recall:{:.3f},  A_F1-score:{:.3f};" \
                           "A_acc:{:.3f}, A_recall:{:.3f},  A_F1-score:{:.3f}".format(epoch,
                                                                                      accu_num[0].item() / sample_num,
                                                                                      recall_pos[0].item() / recall_num,
                                                                                      F1_score(
                                                                                          accu_num[
                                                                                              0].item() / sample_num,
                                                                                          recall_pos[
                                                                                              0].item() / recall_num),
                                                                                      accu_num[1].item() / sample_num,
                                                                                      recall_pos[1].item() / recall_num,
                                                                                      F1_score(
                                                                                          accu_num[
                                                                                              1].item() / sample_num,
                                                                                          recall_pos[
                                                                                              1].item() / recall_num)
                                                                                      )
        # print(f'train_loss:{loss_train.val},train_acc:{acc_train.val}')
    return accu_num[0].item() / sample_num, recall_pos[0].item() / recall_num, F1_score(accu_num[0].item() / sample_num,
                                                                                        recall_pos[
                                                                                            0].item() / recall_num), \
           accu_num[1].item() / sample_num, recall_pos[1].item() / recall_num, F1_score(accu_num[1].item() / sample_num,
                                                                                        recall_pos[
                                                                                            1].item() / recall_num)


@torch.no_grad()
def eval_one_epoch_AV(device, data_loader, loss_f, epoch, model_audio, model_video, model_multi=None):
    model_video.eval()
    model_audio.eval()
    train_loss = torch.zeros(3).unsqueeze(1).to(device)
    accu_num = torch.zeros(3).unsqueeze(1).to(device)  # [audio,video,multi]
    recall_pos = torch.zeros(3).unsqueeze(1).to(device)  # recall所需要的分子

    sample_num = 0
    recall_num = 0  # 标签为1的个数
    data_loader = tqdm(data_loader, file=sys.stdout, ncols=150)
    for step, data in enumerate(data_loader):
        wav, imgs, label = data
        imgs = imgs.to(device)
        wav = wav.to(device)
        label = label.to(device)

        sample_num += label.shape[0]
        recall_num += torch.sum(label == 1).item()

        visual_inputs = imgs.permute(0, 2, 1, 3, 4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0] * visual_inputs.shape[1], visual_inputs.shape[2],
                                              visual_inputs.shape[3], visual_inputs.shape[4])

        pred_a = model_audio(wav)
        pred_v = model_video(visual_inputs)
        # pred_m = model_multi(wav, visual_inputs)
        pred_classes_a = torch.max(pred_a, dim=1)[1]
        pred_classes_v = torch.max(pred_v, dim=1)[1]
        # pred_classes_m = torch.max(pred_m, dim=1)[1]

        accu_num[0] += torch.eq(pred_classes_a, label).sum()
        accu_num[1] += torch.eq(pred_classes_v, label).sum()
        # accu_num[2] += torch.eq(pred_classes_m, label).sum()
        recall_pos[0] += torch.sum((pred_classes_a == 1) & (label == 1))
        recall_pos[1] += torch.sum((pred_classes_v == 1) & (label == 1))
        # recall_pos[2] += torch.sum((pred_classes_m == 1) & (label == 1))
        loss_a = loss_f(pred_a, label)
        loss_v = loss_f(pred_v, label)
        # loss_m = loss_f(pred_m, label)

        train_loss[0] += loss_a.detach()
        train_loss[1] += loss_v.detach()
        # train_loss[2] += loss_m.detach()

        data_loader.desc = "[eval_epoch {}] , A_acc:{:.3f}, A_recall:{:.3f},  A_F1-score:{:.3f};" \
                           "V_acc:{:.3f}, V_recall:{:.3f},  V_F1-score:{:.3f}".format(epoch,
                                                                                      accu_num[0].item() / sample_num,
                                                                                      recall_pos[0].item() / recall_num,
                                                                                      F1_score(
                                                                                          accu_num[
                                                                                              0].item() / sample_num,
                                                                                          recall_pos[
                                                                                              0].item() / recall_num),
                                                                                      accu_num[1].item() / sample_num,
                                                                                      recall_pos[1].item() / recall_num,
                                                                                      F1_score(
                                                                                          accu_num[
                                                                                              1].item() / sample_num,
                                                                                          recall_pos[
                                                                                              1].item() / recall_num)
                                                                                      )
        # print(f'train_loss:{loss_train.val},train_acc:{acc_train.val}')
    return accu_num[0].item() / sample_num, recall_pos[0].item() / recall_num, F1_score(accu_num[0].item() / sample_num,
                                                                                        recall_pos[
                                                                                            0].item() / recall_num), \
           accu_num[1].item() / sample_num, recall_pos[1].item() / recall_num, F1_score(accu_num[1].item() / sample_num,
                                                                                        recall_pos[
                                                                                            1].item() / recall_num)


def train_one_epoch_MAV(device, data_loader, loss_f, optimizer_a, optimizer_v, optimizer_m, epoch, model_audio,model_video,model_multi=None):
    model_multi.train()
    model_video.train()
    model_audio.train()
    scalar = GradScaler()
    train_loss = torch.zeros(4).unsqueeze(1).to(device)
    accu_num = torch.zeros(3).unsqueeze(1).to(device)  # [audio,video,multi]
    recall_pos = torch.zeros(3).unsqueeze(1).to(device)  # recall所需要的分子

    optimizer_a.zero_grad()
    optimizer_v.zero_grad()
    optimizer_m.zero_grad()

    sample_num = 0
    recall_num = 0  # 标签为1的个数
    data_loader = tqdm(data_loader, file=sys.stdout, colour='YELLOW', ncols=180)
    for step, data in enumerate(data_loader):
        wav, imgs, label = data
        imgs = imgs.to(device)
        wav = wav.to(device)
        label = label.to(device)

        sample_num += label.shape[0]
        recall_num += torch.sum(label == 1).item()

        visual_inputs = imgs.permute(0, 2, 1, 3, 4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0] * visual_inputs.shape[1], visual_inputs.shape[2],
                                              visual_inputs.shape[3], visual_inputs.shape[4])
        with autocast():
            pred_a, audio_feature = model_audio(wav)
            pred_v, video_feature = model_video(visual_inputs)
            pred_m = model_multi(audio_feature, video_feature)
            pred_classes_a = torch.max(pred_a, dim=1)[1]
            pred_classes_v = torch.max(pred_v, dim=1)[1]
            pred_classes_m = torch.max(pred_m, dim=1)[1]

            accu_num[0] += torch.eq(pred_classes_a, label).sum()
            accu_num[1] += torch.eq(pred_classes_v, label).sum()
            accu_num[2] += torch.eq(pred_classes_m, label).sum()
            recall_pos[0] += torch.sum((pred_classes_a == 1) & (label == 1))
            recall_pos[1] += torch.sum((pred_classes_v == 1) & (label == 1))
            recall_pos[2] += torch.sum((pred_classes_m == 1) & (label == 1))
            loss_a = loss_f(pred_a, label)
            loss_v = loss_f(pred_v, label)
            loss_m = loss_f(pred_m, label)
            loss_al = 0.5 * loss_m + 0.3 * loss_v + 0.2 * loss_a
        if not torch.isfinite(loss_al):
            print('WARNING: non-finite loss, ending training ', loss_a)
            sys.exit(1)
        train_loss[0] += loss_a.detach()
        train_loss[1] += loss_v.detach()
        train_loss[2] += loss_m.detach()
        train_loss[3] += loss_al.detach()
        # scalar.scale(loss_a).backward()
        # scalar.scale(loss_v).backward()
        scalar.scale(loss_al).backward()

        scalar.step(optimizer_a)
        scalar.step(optimizer_v)
        scalar.step(optimizer_m)
        scalar.update()
        optimizer_a.zero_grad()
        optimizer_v.zero_grad()
        optimizer_m.zero_grad()

        A_acc = accu_num[0].item() / sample_num
        A_recall = recall_pos[0].item() / recall_num
        A_f1 = F1_score(accu_num[0].item() / sample_num,recall_pos[0].item() / recall_num)
        V_acc = accu_num[1].item() / sample_num
        V_recall = recall_pos[1].item() / recall_num
        V_f1 = F1_score(accu_num[1].item() / sample_num, recall_pos[1].item() / recall_num)
        M_acc = accu_num[2].item() / sample_num
        M_recall = recall_pos[2].item() / recall_num
        M_f1 = F1_score(accu_num[2].item() / sample_num, recall_pos[2].item() / recall_num)
        loss_al=train_loss[3].item() / (step + 1)

        data_loader.desc = "[train{}],Loss:{:.3f}" \
                           "A_acc:{:.3f},A_rec:{:.3f},A_F1:{:.3f};" \
                           "V_acc:{:.3f},V_rec:{:.3f},V_F1:{:.3f};" \
                           "M_acc:{:.3f},M_rec:{:.3f},M_F1:{:.3f}".format(epoch,loss_al,A_acc,A_recall,A_f1,V_acc,V_recall,V_f1,M_acc,M_recall,M_f1)

    return loss_al,A_acc,A_recall,A_f1,V_acc,V_recall,V_f1,M_acc,M_recall,M_f1


@torch.no_grad()
def eval_one_epoch_MAV(device, data_loader, loss_f, epoch, model_audio, model_video, model_multi=None):
    model_video.eval()
    model_audio.eval()
    model_multi.eval()
    train_loss = torch.zeros(4).unsqueeze(1).to(device)
    accu_num = torch.zeros(3).unsqueeze(1).to(device)  # [audio,video,multi]
    recall_pos = torch.zeros(3).unsqueeze(1).to(device)  # recall所需要的分子

    sample_num = 0
    recall_num = 0  # 标签为1的个数
    data_loader = tqdm(data_loader, file=sys.stdout, ncols=180)
    for step, data in enumerate(data_loader):
        wav, imgs, label = data
        imgs = imgs.to(device)
        wav = wav.to(device)
        label = label.to(device)

        sample_num += label.shape[0]
        recall_num += torch.sum(label == 1).item()

        visual_inputs = imgs.permute(0, 2, 1, 3, 4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0] * visual_inputs.shape[1], visual_inputs.shape[2],
                                              visual_inputs.shape[3], visual_inputs.shape[4])

        pred_a, audio_feature = model_audio(wav)
        pred_v, video_feature = model_video(visual_inputs)
        pred_m = model_multi(audio_feature, video_feature)
        pred_classes_a = torch.max(pred_a, dim=1)[1]
        pred_classes_v = torch.max(pred_v, dim=1)[1]
        pred_classes_m = torch.max(pred_m, dim=1)[1]

        accu_num[0] += torch.eq(pred_classes_a, label).sum()
        accu_num[1] += torch.eq(pred_classes_v, label).sum()
        accu_num[2] += torch.eq(pred_classes_m, label).sum()
        recall_pos[0] += torch.sum((pred_classes_a == 1) & (label == 1))
        recall_pos[1] += torch.sum((pred_classes_v == 1) & (label == 1))
        recall_pos[2] += torch.sum((pred_classes_m == 1) & (label == 1))
        loss_a = loss_f(pred_a, label)
        loss_v = loss_f(pred_v, label)
        loss_m = loss_f(pred_m, label)
        loss_al = 0.5 * loss_m + 0.3 * loss_v + 0.2 * loss_a

        train_loss[0] += loss_a.detach()
        train_loss[1] += loss_v.detach()
        train_loss[2] += loss_m.detach()
        train_loss[3] += loss_al.detach()

        A_acc = accu_num[0].item() / sample_num
        A_recall = recall_pos[0].item() / recall_num
        A_f1 = F1_score(accu_num[0].item() / sample_num, recall_pos[0].item() / recall_num)
        V_acc = accu_num[1].item() / sample_num
        V_recall = recall_pos[1].item() / recall_num
        V_f1 = F1_score(accu_num[1].item() / sample_num, recall_pos[1].item() / recall_num)
        M_acc = accu_num[2].item() / sample_num
        M_recall = recall_pos[2].item() / recall_num
        M_f1 = F1_score(accu_num[2].item() / sample_num, recall_pos[2].item() / recall_num)
        loss_al = train_loss[3].item() / (step + 1)

        data_loader.desc = "[eval{}],Loss:{:.3f}" \
                           "A_acc:{:.3f},A_rec:{:.3f},A_F1:{:.3f};" \
                           "V_acc:{:.3f},V_rec:{:.3f},V_F1:{:.3f};" \
                           "M_acc:{:.3f},M_rec:{:.3f},M_F1:{:.3f}".format(epoch, loss_al, A_acc, A_recall, A_f1, V_acc,
                                                                          V_recall, V_f1, M_acc, M_recall, M_f1)

    return loss_al, A_acc, A_recall, A_f1, V_acc, V_recall, V_f1, M_acc, M_recall, M_f1
