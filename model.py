import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import pdb
import embedding
import auxiliary as aux

class Meta(nn.Module):
    def __init__(self, args, pretrain=None):
        super(Meta, self).__init__()
        self.args = args
        if self.args.model == 'HS_CNN':
            out_ch = 90
            out_h = 47
            out_vec = 256 * 16
            self.embedding = embedding.HS_CNN()
        elif self.args.model == 'EEGNet':
            out_ch = 16
            out_h = 109
            out_vec = 256 * 37
            self.embedding = embedding.EEGNet()
        elif self.args.model == 'DeepconvNet':
            out_ch = 200
            out_h = 88
            out_vec = 256 * 30
            self.embedding = embedding.DeepconvNet()
        else:
            print('Check model name...')

        self.out_ch = out_ch
        self.self_attention = aux.Self_Attention()
        self.aggregation = self.args.aggregation
        self.aggregation_ = None

        if 'att' in self.aggregation:
            self.aggregation_ = aux.Attention(out_ch)
            if 'dual_att' in self.aggregation:
                self.self_conv = aux.Self_Conv(out_ch)

        if 'RN' in self.args.base:
            self.conv1 = nn.Conv2d(out_ch * 2, 128, [30, 1], [1, 1], padding=[14, 0])
            self.conv2 = nn.Conv2d(128, 256, [15, 1], [1, 1], padding=[7, 0])
            self.avgp = nn.AvgPool2d([3, 1], [3, 1], [1, 0])
            self.cnn = nn.Sequential(
                self.conv1,
                nn.ELU(),
                self.avgp,
                self.conv2,
                nn.ELU(),
                nn.Dropout(p=0.3)
            )
            self.flatten = nn.Flatten()
            self.fc = nn.Sequential(
                nn.Linear(out_vec, 512, bias=True),
                nn.ELU(),
                nn.Linear(512, 256, bias=True),
                nn.ELU(),
                nn.Linear(256, 64, bias=True),
                nn.ELU(),
                nn.Linear(64, 1, bias=True)
            )

        if 'PTN' in self.args.base:
            self.cosine = nn.CosineSimilarity()

        self._init_weight()

        if pretrain:
            pretrained = torch.load(pretrain, map_location=torch.device('cpu'))
            pretrained_ = {}
            for k, v in pretrained.items():
                if 'fc' not in k:
                    pretrained_[k.split('embedding.')[1]] = v

            self.embedding.load_state_dict(pretrained_)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)  # , mean=0.0, std=0.0001)

    def forward(self, sup, que):
        '''
        :param sup: support set [B, K, ch, 875, 3, N]
        :param que: query set [B, ch, 875, 3]
        :return: relation score for each N [B, N]
        '''

        sup_size = sup.shape

        sup = torch.reshape(sup, [sup_size[0] * sup_size[1], sup_size[2], sup_size[3], sup_size[4], sup_size[5]])
        self.que_feature = self.embedding(que)

        if 'dual_att' in self.aggregation:
            self.que_feature = self.self_conv(self.que_feature)

        pred = []
        self.prototype = []
        for i in range(sup_size[-1]):
            sup_feature = self.embedding(sup[:, :, :, :, i])
            feature_size = sup_feature.shape
            sup_feature = torch.reshape(sup_feature,
                                        [sup_size[0], sup_size[1], self.out_ch, feature_size[2], feature_size[3]])

            if 'dual_att' in self.aggregation:
                sup_feature = torch.reshape(sup_feature,
                                            [sup_size[0] * sup_size[1], self.out_ch, feature_size[2], feature_size[3]])
                sup_feature = self.self_conv(sup_feature)
                sup_feature = torch.reshape(sup_feature,
                                            [sup_size[0], sup_size[1], self.out_ch, feature_size[2], feature_size[3]])

            if 'PTN' in self.args.base:
                sup_feature = self.self_attention(self.que_feature, sup_feature, sup_feature)

            if not self.aggregation_:
                proto = torch.mean(sup_feature, 1)
            else:
                proto = self.aggregation_(sup_feature, self.que_feature)
            self.prototype.append(proto)

            if 'RN' in self.args.base:
                score = self.cnn(torch.cat((proto, self.que_feature), dim=1))
                score = self.flatten(score)
                score = self.fc(score)
            if 'PTN' in self.args.base:
                score = -torch.cdist(self.que_feature.reshape(self.que_feature.shape[0], -1), proto.reshape(proto.shape[0], -1))
                score = torch.diagonal(score) / (self.out_ch*feature_size[2]*feature_size[3]) * 10
                score = torch.unsqueeze(score, dim=-1)
            pred.append(score)

        pred = torch.cat(pred, dim=-1)
        output = pred

        return output


if __name__ == '__main__':
    x = torch.rand((1, 1, 875, 3))
    model = DeepconvNet()
    out = model(x)
