import sys

from M_utils import MyDataset
from Model import multimodalcnn
import torch
import numpy as np
import random
from torchsummary import summary
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
import os
import argparse
from torchvision import transforms
import transforms as Trans
import torch.utils.data as data
import model_load
from train_eval_test import train_one_epoch_AV,eval_one_epoch_AV
from torch.utils.tensorboard import SummaryWriter
from Model import AudioNet1
from Model import multimodalcnn

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_train', type=str, default='', help='训练数据的路径')
parser.add_argument('--data_eval', type=str, default='', help='验证数据集的存放路径')
parser.add_argument('--pretrain_path', type=str, default='', help='官方预训练的模型保存的路径')
parser.add_argument('--n_class', type=int, default=2, help='')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight Decay')
parser.add_argument('--lr_patience', default=20, type=int,
                    help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
parser.add_argument('--resume', type=bool, default='False', help='是否从半路开始恢复训练，一般这种情况是服务器不小心断连了，但又没必要重新训练，')
parser.add_argument('--resume_path', type=str, default='', help='需要开始继续训练的模型参数保存的地址')
parser.add_argument('--video_norm_value', default=255, type=int,  # 说实话，这个参数是干嘛的，也不是很清楚
                    help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')

parser.add_argument('--learning_rate', type=int, default=0.05, help='')
parser.add_argument('--model', type=str, default='audiovisual', help='audio | visual | audiovisual')
# parser.add_argument('--data-class', type=str, default='p', help='p | n | nt')
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--batch_size', type=int, default='32', help='')


def main(opts, flod):
    device = torch.device(opts.device if torch.cuda.is_available() else 'cpu')
    print(f'1-->Using Gpu:{device}')
    nw = min([os.cpu_count(), opts.batch_size if opts.batch_size > 1 else 0, 8])
    print(f'2-->Using {nw} dataloader workers every process')
    print(opts)

    tb_writer = SummaryWriter(
        '/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s1/runs/1/flod_{:d}'.format(flod))

    if opts.model == 'audio':

        train_dataset = MyDataset.Multi_dataset(opts.data_train,
                                                video_transform=None,
                                                audio_transform=True,
                                                data_type='audio')
        eval_dataset = MyDataset.Multi_dataset(opts.data_eval,
                                               video_transform=None,
                                               audio_transform=True,
                                               data_type='audio')
    elif opts.model == 'visual':
        video_transform_train = Trans.Compose([
            Trans.RandomHorizontalFlip(),
            Trans.RandomRotate(),
            Trans.ToTensor(opts.video_norm_value)])
        video_transform_eval = Trans.Compose([
            Trans.ToTensor(opts.video_norm_value)])

        train_dataset = MyDataset.Multi_dataset(opts.data_train,
                                                video_transform=video_transform_train,
                                                audio_transform=None,
                                                data_type='video')
        eval_dataset = MyDataset.Multi_dataset(opts.data_eval,
                                               video_transform=video_transform_eval,
                                               audio_transform=None,
                                               data_type='video')
    elif opts.model == 'audiovisual':
        video_transform_train = Trans.Compose([
            Trans.RandomHorizontalFlip(),
            Trans.RandomRotate(),
            Trans.ToTensor(opts.video_norm_value)])
        video_transform_eval = Trans.Compose([
            Trans.ToTensor(opts.video_norm_value)])
        train_dataset = MyDataset.Multi_dataset(opts.data_train,
                                                video_transform=video_transform_train,
                                                audio_transform=True,
                                                data_type='audiovisual')
        eval_dataset = MyDataset.Multi_dataset(opts.data_eval,
                                               video_transform=video_transform_eval,
                                               audio_transform=None,
                                               data_type='audiovisual')

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=opts.batch_size,
                                   shuffle=True,
                                   num_workers=nw,
                                   pin_memory=True,
                                   )
    eval_loader = data.DataLoader(dataset=eval_dataset,
                                  batch_size=opts.batch_size * 2,
                                  shuffle=True,
                                  num_workers=nw,
                                  pin_memory=True,
                                  )
    model_audio =AudioNet1.Audio_net1(device=device, in_feature=1, hidden_feature=64, out_feature=128,
                                      input_size=18, num_layers=2, hidden_size=18, num_class=2).to(device)
    model_video=multimodalcnn.EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], 2, 18).to(device)

    # model = model_load.generate_model(opts)
    # print(model)
    # summary(model,(3,224,224))
    # print(model)
    # 获取模型的参数列表
    # params = list(model.parameters())
    # total_params = sum(p.numel() for p in params)
    # print('3--Total number of parameters:', total_params)

    loss_f = nn.CrossEntropyLoss()
    loss_f = loss_f.to(device)

    optimizer_a = optim.SGD(
        model_audio.parameters(),
        lr=opts.learning_rate,
        momentum=opts.momentum,
        dampening=opts.dampening,  # 阻尼参数
        weight_decay=opts.weight_decay,  # 权重衰减系数
        nesterov=False)
    optimizer_v = optim.SGD(
        model_video.parameters(),
        lr=opts.learning_rate,
        momentum=opts.momentum,
        dampening=opts.dampening,  # 阻尼参数
        weight_decay=opts.weight_decay,  # 权重衰减系数
        nesterov=False)
    # 学习率策略
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', patience=opts.lr_patience)  # 当loss不再减小时，且超过了10个批次，进行学习率的下降gamma倍，gamma默认参数的值为 0.1

    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_a, [15, 30, 45, 60,75,90], gamma=0.5,
                                                     last_epoch=-1)  # 在这个节点处，将学习率乘以0.5
    scheduler_v = torch.optim.lr_scheduler.MultiStepLR(optimizer_a, [15, 30, 45, 60, 75, 90], gamma=0.5,
                                                       last_epoch=-1)  # 在这个节点处，将学习率乘以0.5
    start_epoch = 0
    if opts.resume:
        pass
    max_acc_a = 0.0
    max_acc_v = 0.0
    max_epoch_a = 0
    max_epoch_v = 0
    for epoch in range(start_epoch,opts.epochs):
        # train_loss, train_acc, train_recall, train_f1 = train_one_epoch(model, device, train_loader, loss_f, optimizer,epoch)
        # eval_loss, eval_acc, eval_recall, eval_f1 = eval_one_epoch(model, eval_loader, device, loss_f, epoch)
        A_acc,A_recall,A_f1,V_acc,V_recall,V_f1=train_one_epoch_AV(device,train_loader,loss_f,optimizer_a,optimizer_v,epoch,model_audio,model_video)
        Ae_acc,Ae_recall,Ae_f1,Ve_acc,Ve_recall,Ve_f1=eval_one_epoch_AV(device,eval_loader,loss_f,epoch,model_audio,model_video)

        scheduler_a.step()
        scheduler_v.step()
        tb_writer.add_scalar('Ae_acc', Ae_acc, epoch)
        tb_writer.add_scalar('Ae_recall', Ae_recall, epoch)
        tb_writer.add_scalar('Ae_f1', Ae_f1, epoch)
        tb_writer.add_scalar('Ve_acc', Ve_acc, epoch)
        tb_writer.add_scalar('Ve_recall', Ve_recall, epoch)
        tb_writer.add_scalar('Ve_f1',Ve_f1,epoch)
        tb_writer.add_scalar('lr', optimizer_a.param_groups[0]['lr'], epoch)
        print('lr', optimizer_a.param_groups[0]["lr"])
        checkpoint_a = {'state_dict': model_audio.state_dict(),
                      # 'optimizer': optimizer.state_dict(),
                      'epoch': epoch}
        checkpoint_v={'state_dict':model_video.state_dict(),
        }
        torch.save(checkpoint_a,
                   '/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/txt_files/mm3s/checkpoint/5/audio_flod_{}.pth'.format(flod))
        torch.save(checkpoint_v,
                   '/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/txt_files/mm3s/checkpoint/5/video_flod_{}.pth'.format(flod))
        # if max_acc_a < Ae_acc && max_acc_v <:
        #     max_acc = eval_acc
        #     max_epoch = epoch
        #     print("\033[0;32;40mAcc_max is updating,which is {}\033[0m".format(max_acc.item()))
        #     torch.save(checkpoint,
        #                '/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/txt_files/mm3s/checkpoint/4/flod_{}_best{}_acc{:1f}.pth'.format(flod,epoch,eval_acc.item()))

        # yml_wri(args.batch_size, optimizer.param_groups[0]["lr"], information, flod, epoch,
        #         train_loss, train_acc, train_recall.tolist(), train_f1,
        #         eval_loss, eval_acc, eval_recall.tolist(), eval_f1)
        if (epoch - max_epoch_a > 40 & epoch-max_epoch_v>40):
            print('acc is never up during 40 epoch,so,end it early!')
            break


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    torch.cuda.set_device(1)
    seed = 5
    set_seed(seed)
    door = input('你把模型的参数修改好了吗:')
    if door == 'yes':
        for i in range(0,10):
            opts = parser.parse_args()
            print('Runing in flod:{:d}'.format(i))
            opts.data_train = '/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/txt_files/mm3s/crop3s_shift3s/tain_3s_{}.txt'.format(i)
            opts.data_eval = '/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/txt_files/mm3s/crop3s_shift3s/eval_3s_{}.txt'.format(i)
            print(opts.data_eval)
            opts.pretrain_path = '/home/liuzhengyu/duanw/python_code/LiuP100/30-VIT_316/Checkpoint/EfficientFace_Trained_on_AffectNet7.pth.tar'
            main(opts, i)
    else:
        sys.exit()
