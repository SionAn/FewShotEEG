import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import pdb

class HS_CNN(nn.Module):
    def __init__(self):
        super(HS_CNN, self).__init__()
        self.conv11 = nn.Conv2d(1, 10, [45, 1], [3, 1], padding=[22, 0])
        torch.nn.init.normal_(self.conv11.weight, mean=0.0, std=0.0001)
        self.conv11_ = nn.Sequential(
            self.conv11,
            nn.ELU(),
            nn.Conv2d(10, 10, [1, 3], [1, 1], padding=0),
            nn.ELU(),
            nn.MaxPool2d([6, 1], [6, 1])
        )
        self.conv12 = nn.Conv2d(1, 10, [65, 1], [3, 1], padding=[33, 0])
        torch.nn.init.normal_(self.conv12.weight, mean=0.0, std=0.0001)
        self.conv12_ = nn.Sequential(
            self.conv12,
            nn.ELU(),
            nn.Conv2d(10, 10, [1, 3], [1, 1], padding=0),
            nn.ELU(),
            nn.MaxPool2d([6, 1], [6, 1])
        )
        self.conv13 = nn.Conv2d(1, 10, [85, 1], [3, 1], padding=[44, 0])
        torch.nn.init.normal_(self.conv13.weight, mean=0.0, std=0.0001)
        self.conv13_ = nn.Sequential(
            self.conv13,
            nn.ELU(),
            nn.Conv2d(10, 10, [1, 3], [1, 1], padding=0),
            nn.ELU(),
            nn.MaxPool2d([6, 1], [6, 1])
        )
        self.conv21 = nn.Conv2d(1, 10, [45, 1], [3, 1], padding=[22, 0])
        torch.nn.init.normal_(self.conv21.weight, mean=0.0, std=0.0001)
        self.conv21_ = nn.Sequential(
            self.conv21,
            nn.ELU(),
            nn.Conv2d(10, 10, [1, 3], [1, 1], padding=0),
            nn.ELU(),
            nn.MaxPool2d([6, 1], [6, 1])
        )
        self.conv22 = nn.Conv2d(1, 10, [65, 1], [3, 1], padding=[33, 0])
        torch.nn.init.normal_(self.conv22.weight, mean=0.0, std=0.0001)
        self.conv22_ = nn.Sequential(
            self.conv22,
            nn.ELU(),
            nn.Conv2d(10, 10, [1, 3], [1, 1], padding=0),
            nn.ELU(),
            nn.MaxPool2d([6, 1], [6, 1])
        )
        self.conv23 = nn.Conv2d(1, 10, [85, 1], [3, 1], padding=[44, 0])
        torch.nn.init.normal_(self.conv23.weight, mean=0.0, std=0.0001)
        self.conv23_ = nn.Sequential(
            self.conv23,
            nn.ELU(),
            nn.Conv2d(10, 10, [1, 3], [1, 1], padding=0),
            nn.ELU(),
            nn.MaxPool2d([6, 1], [6, 1])
        )
        self.conv31 = nn.Conv2d(1, 10, [45, 1], [3, 1], padding=[22, 0])
        torch.nn.init.normal_(self.conv31.weight, mean=0.0, std=0.0001)
        self.conv31_ = nn.Sequential(
            self.conv31,
            nn.ELU(),
            nn.Conv2d(10, 10, [1, 3], [1, 1], padding=0),
            nn.ELU(),
            nn.MaxPool2d([6, 1], [6, 1])
        )
        self.conv32 = nn.Conv2d(1, 10, [65, 1], [3, 1], padding=[33, 0])
        torch.nn.init.normal_(self.conv32.weight, mean=0.0, std=0.0001)
        self.conv32_ = nn.Sequential(
            self.conv32,
            nn.ELU(),
            nn.Conv2d(10, 10, [1, 3], [1, 1], padding=0),
            nn.ELU(),
            nn.MaxPool2d([6, 1], [6, 1])
        )
        self.conv33 = nn.Conv2d(1, 10, [85, 1], [3, 1], padding=[44, 0])
        torch.nn.init.normal_(self.conv33.weight, mean=0.0, std=0.0001)
        self.conv33_ = nn.Sequential(
            self.conv33,
            nn.ELU(),
            nn.Conv2d(10, 10, [1, 3], [1, 1], padding=0),
            nn.ELU(),
            nn.MaxPool2d([6, 1], [6, 1])
        )
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(4320, 2, bias=True)
        # torch.nn.init.normal_(self.fc.weight, mean=0.0, std=0.0001)
        # self.softmax = nn.Softmax(1)

    def forward(self, x):  # input shape: B, 3, 875, 3
        self.input = x
        layer = []
        layer47_1 = self.conv11_(self.input[:, 0:1])
        layer47_2 = self.conv12_(self.input[:, 0:1])
        layer47_3 = self.conv13_(self.input[:, 0:1])
        layer813_1 = self.conv21_(self.input[:, 1:2])
        layer813_2 = self.conv22_(self.input[:, 1:2])
        layer813_3 = self.conv23_(self.input[:, 1:2])
        layer1332_1 = self.conv31_(self.input[:, 2:3])
        layer1332_2 = self.conv32_(self.input[:, 2:3])
        layer1332_3 = self.conv33_(self.input[:, 2:3])
        # layer47_1 = self.flatten(layer47_1)
        # layer47_2 = self.flatten(layer47_2)
        # layer47_3 = self.flatten(layer47_3)
        # layer813_1 = self.flatten(layer813_1)
        # layer813_2 = self.flatten(layer813_2)
        # layer813_3 = self.flatten(layer813_3)
        # layer1332_1 = self.flatten(layer1332_1)
        # layer1332_2 = self.flatten(layer1332_2)
        # layer1332_3 = self.flatten(layer1332_3)
        layer.append(layer47_1)
        layer.append(layer47_2)
        layer.append(layer47_3)
        layer.append(layer813_1)
        layer.append(layer813_2)
        layer.append(layer813_3)
        layer.append(layer1332_1)
        layer.append(layer1332_2)
        layer.append(layer1332_3)

        layer = torch.cat(layer, dim=1)
        # layer = torch.cat(layer, dim=-1)
        # layer = self.fc(layer)
        # output = self.softmax(layer)

        return layer  # output


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, [125, 1], [1, 1], padding=[62, 0])
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, [1, 3], [1, 1], padding=[0, 0])
        self.bn2 = nn.BatchNorm2d(16)
        self.blck1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.conv2,
            self.bn2,
            nn.ELU(),
            nn.AvgPool2d([4, 1], [2, 1], padding=[2, 0]),
            nn.Dropout2d(0.25)
        )
        self.conv3 = nn.Conv2d(16, 16, [16, 1], [1, 1], padding=[7, 0])
        self.bn3 = nn.BatchNorm2d(16)
        self.blck2 = nn.Sequential(
            self.conv3,
            self.bn3,
            nn.ELU(),
            nn.AvgPool2d([8, 1], [4, 1], padding=[4, 0]),
            nn.Dropout2d(0.25)
        )

        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(1760, 2, bias=True)
        # self.softmax = nn.Softmax(1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.0001)

    def forward(self, x):
        output = self.blck1(x)
        output = self.blck2(output)
        # output = self.flatten(output)
        # output = self.fc(output)
        # output = self.softmax(output)

        return output


class DeepconvNet(nn.Module):
    def __init__(self):
        super(DeepconvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, [10, 1], [1, 1], padding=[0, 0])
        self.conv2 = nn.Conv2d(25, 25, [1, 3], [1, 1], padding=[0, 0])
        self.bn1 = nn.BatchNorm2d(25)
        self.blck1 = nn.Sequential(
            self.conv1,
            self.conv2,
            self.bn1,
            nn.ELU(),
            nn.Dropout2d(0.5)
        )
        self.conv3 = nn.Conv2d(25, 50, [10, 1], [1, 1], padding=[0, 0])
        self.bn2 = nn.BatchNorm2d(50)
        self.blck2 = nn.Sequential(
            self.conv3,
            self.bn2,
            nn.ELU(),
            nn.MaxPool2d([3, 1], [3, 1], padding=[0, 0]),
            nn.Dropout2d(0.5)
        )
        self.conv4 = nn.Conv2d(50, 100, [10, 1], [1, 1], padding=[0, 0])
        self.bn3 = nn.BatchNorm2d(100)
        self.blck3 = nn.Sequential(
            self.conv4,
            self.bn3,
            nn.ELU(),
            nn.Dropout2d(0.5)
        )
        self.conv5 = nn.Conv2d(100, 200, [10, 1], [1, 1], padding=[0, 0])
        self.bn4 = nn.BatchNorm2d(200)
        self.blck4 = nn.Sequential(
            self.conv5,
            self.bn4,
            nn.ELU(),
            nn.MaxPool2d([3, 1], [3, 1], padding=[0, 0]),
            nn.Dropout2d(0.5)
        )
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(17800, 2, bias=True)
        # self.softmax = nn.Softmax(1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.0001)

    def forward(self, x):
        output = self.blck1(x)
        output = self.blck2(output)
        output = self.blck3(output)
        output = self.blck4(output)
        # output = self.flatten(output)
        # output = self.fc(output)
        # output = self.softmax(output)

        return output

if __name__ == '__main__':
    x = torch.rand((1, 1, 875, 3))
    model = DeepconvNet()
    out = model(x)
