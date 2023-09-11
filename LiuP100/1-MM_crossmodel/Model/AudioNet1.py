'''
2023-4-6
重新为多模态体征提取网络写一个单模态语音特征提取网络。
'''
from torch import nn
import torch


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2, 1))


class time_freq(nn.Module):
    def __init__(self, in_feature=1, hidden_feature=64, out_feature=128):
        super(time_freq, self).__init__()
        # 时间卷积
        self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=hidden_feature, kernel_size=(1, 4),
                               stride=(1, 1), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(hidden_feature)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        # 频率卷积
        self.conv2 = nn.Conv2d(in_channels=hidden_feature, out_channels=out_feature, kernel_size=(64, 1),
                               stride=1)
        self.bn2 = nn.BatchNorm2d(out_feature)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out1 = self.pool1(out)

        out = self.conv2(out1)
        out2 = self.pool2(self.relu(self.bn2(out)))
        return out2


class seqLstm(nn.Module):
    def __init__(self, device, input_size=18, num_layers=2, hidden_size=18):
        super(seqLstm, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=False)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        return out


class Audio_net1(nn.Module):

    def __init__(self, device='cpu', in_feature=1, hidden_feature=64, out_feature=128,
                 input_size=18, num_layers=2, hidden_size=18, num_class=2):
        super(Audio_net1, self).__init__()
        self.time_fre_cnn = time_freq(in_feature, hidden_feature, out_feature)
        self.seq_lstm = seqLstm(device=device, input_size=input_size, num_layers=num_layers,
                                hidden_size=hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64,num_class)
        )

        self.conv1d_0 = conv1d_block_audio(128, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)

        self.audio_feature=None

    def forward_feature(self, x):
        # 通过cnn和LSTM之后得到的语音特征。
        x = x.unsqueeze(1)
        x = self.time_fre_cnn(x)
        x = x.squeeze(2)  # 这里的permute(0, 2, 1)有问题
        out = self.seq_lstm(x)
        return out

    def forward_stage1(self, x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        # x1 = self.fc(x)
        x1 = self.classifier(x)
        return x1

    def forward(self, x):
        x = self.forward_feature(x)
        self.audio_feature=x
        audio_feature=self.audio_feature
        # x = self.forward_stage1(x)
        # x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x,audio_feature


if __name__ == '__main__':
    data_in = torch.randn([4, 64, 149])
    model = Audio_net1()
    out = model(data_in)
    print(out.shape)
    # data_out, hn, cn = model(data_in)
    # print('data_out:', data_out.shape)
    # print('hn:', hn.shape)
    # print('cn:', cn.shape)
    # print(model)
