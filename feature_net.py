import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer


def conv3x3x3_ft(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        FeatureWiseTransformation2d_fw(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

#
class FuzzyFusion(nn.Module):
    def __init__(self, channel):
        super(FuzzyFusion, self).__init__()
        self.attn_conv = nn.Conv3d(channel, 1, kernel_size=1)  # 空间注意力权重生成
        self.dropout = nn.Dropout(0.1)

    def fuzzy_sigmoid(self, x):
        """自定义模糊sigmoid函数"""
        alpha = 2.0  # 控制斜率
        beta = 1.0  # 控制偏移
        return 1 / (1 + torch.exp(-alpha * (x - beta)))

    def forward(self, x1, x3):
        # 特征融合路径
        combined = x1 + x3
        fuzzy = self.fuzzy_sigmoid(combined)

        # 注意力权重生成
        attn_map = self.attn_conv(combined * fuzzy)  # 利用模糊门控特征
        attn_weights = torch.sigmoid(self.dropout(attn_map))

        # 模糊融合
        return attn_weights * x1 + (1 - attn_weights) * x3


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

        self.fusion = FuzzyFusion(out_channel)  # 模糊融合模块
    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = self.conv3(x2)

        fused = self.fusion(x1, x3)  # 模糊融合代替直接相加
        return F.relu(fused, inplace=True)


def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)


class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    feature_augment = True

    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum,
                                                             track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1, 1) * 0.3)
            self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1, 1) * 0.5)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

        # apply feature-wise transformation
        if self.feature_augment and self.training:
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, 1, dtype=self.gamma.dtype,
                                     device=self.gamma.device) * softplus(self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, 1, dtype=self.beta.dtype,
                                device=self.beta.device) * softplus(self.beta)).expand_as(out)
            out = gamma * out + beta
        return out

#########################原来的#############################
class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, patch_size, emb_size):
        super(D_Res_3d_CNN, self).__init__()
        self.in_channel = in_channel
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.block1 = residual_block(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        self.block2 = residual_block(out_channel1, out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(in_channels=out_channel2, out_channels=32, kernel_size=3, bias=False)

        self.layer_second = nn.Sequential(nn.Linear(in_features=self._get_layer_size()[0],
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))

        self.layer_last = nn.Sequential(nn.Linear(in_features=self._get_layer_size()[1],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, 100,
                             self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            _, t, c, w, h = x.size()
            s1 = t * c * w * h
            x = self.conv(x)
            x = x.view(x.shape[0], -1)
            s2 = x.size()[1]
        return s1, s2

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        inter = x
        inter = inter.view(inter.shape[0], -1)
        inter = self.layer_second(inter)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.layer_last(x)
        out = []
        out.append(inter)
        out.append(x)
        return out

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


TAR_INPUT_DIMENSION, N_DIMENSION = 103, 100
SRC_INPUT_DIMENSION, N_DIMENSION = 128, 100
class Network(nn.Module):
    def __init__(self, patch_size, emb_size):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1, 8, 16, patch_size, emb_size)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, x, domain='source'):
        if domain == 'target':
            x = self.target_mapping(x)
        elif domain == 'source':
            x = self.source_mapping(x)
        feature = self.feature_encoder(x)
        return feature

