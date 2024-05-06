import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import pdb
import random

class Self_Conv(nn.Module):
    def __init__(self, out_ch):
        super(Self_Conv, self).__init__()
        self.conv1 = nn.Conv2d(out_ch, 32, [3, 1], [1, 1], padding=[1, 0])
        self.conv2 = nn.Conv2d(32, 1, [3, 1], [1, 1], padding=[1, 0])
        self.cnn = nn.Sequential(
            self.conv1,
            nn.Sigmoid(),
            self.conv2,
            nn.Sigmoid()
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)  # , mean=0.0, std=0.0001)

    def forward(self, x):
        layer = self.cnn(x)

        return x * layer


class Self_Attention(nn.Module):
    def __init__(self):
        super(Self_Attention, self).__init__()

    def forward(self, query, key, value):
        '''
        :param query: [N, C, H, W]
        :param key: [N, K, C, H, W]
        :param value: [N, K, C, H, W]
        :return:
        '''
        N, K, C, H, W = key.shape
        query = torch.permute(query, [0, 2, 3, 1]).reshape(N, H * W, C)
        query = query.repeat_interleave(repeats=K, dim=0)  # [NK, HW, C]
        key = torch.permute(key, [0, 1, 3, 4, 2]).reshape(N * K, H * W, C)
        value = torch.permute(value, [0, 1, 3, 4, 2]).reshape(N * K, H * W, C)

        corr = torch.bmm(query, key.transpose(1, 2))  # [NK, HW, HW]
        norm = torch.nn.functional.softmax(corr, dim=2)
        out = torch.bmm(norm, value)  # [NK, HW, C]
        out = torch.permute(out, [0, 2, 1])

        return out.reshape(N, K, C, H, W)


class Attention(nn.Module):
    def __init__(self, out_ch):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(out_ch * 2, 128, [30, 1], [1, 1], padding=[14, 0])
        self.avgpool = nn.AvgPool2d([3, 1], [3, 1], [1, 0])
        self.conv2 = nn.Conv2d(128, 256, [15, 1], [1, 1], padding=[7, 0])
        self.cnn = nn.Sequential(
            self.conv1,
            nn.ELU(),
            self.avgpool,
            self.conv2,
            nn.ELU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ELU(),
            nn.Linear(256, 100, bias=True),
            nn.ELU(),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)  # , mean=0.0, std=0.0001)

    def forward(self, sup, que):
        '''
        :param sup: [B, K, ch, H, W]
        :param que: [B, ch, H, W]
        :return: prototype
        '''
        que = torch.unsqueeze(que, dim=1).repeat(1, sup.shape[1], 1, 1, 1)
        x = torch.cat((sup, que), dim=2)
        x_size = x.shape
        x = torch.reshape(x, (x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4]))
        layer = self.cnn(x)
        layer = torch.mean(layer, dim=(-2, -1))  # Global average pooling
        layer = self.fc(layer)
        attention_score = torch.reshape(layer, (x_size[0], x_size[1]))
        attention_sum = torch.sum(attention_score, dim=1)
        attention_norm = torch.div(attention_score, torch.unsqueeze(attention_sum, -1))
        attention_norm = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(attention_norm, -1), -1), -1)
        proto = sup * attention_norm
        proto = torch.sum(proto, dim=1)

        return proto