# -*- coding:utf-8 -*-
# @FileName  :endoir_model
# @Time      :2023/11/4 10:09
# @Author    :CJL
from .endoir_blocks import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math


from .endoir_blocks import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

class ConvolutionalBlock(nn.Module):
    """
    卷积模块,由卷积层, BN归一化层, 激活层构成.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :参数 in_channels: 输入通道数
        :参数 out_channels: 输出通道数
        :参数 kernel_size: 核大小
        :参数 stride: 步长
        :参数 batch_norm: 是否包含BN层
        :参数 activation: 激活层类型; 如果没有则为None
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # 层列表
        layers = list()

        # 1个卷积层
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # 1个BN归一化层
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # 1个激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # 合并层
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        前向传播

        :参数 input: 输入图像集，张量表示，大小为 (N, in_channels, w, h)
        :返回: 输出图像集，张量表示，大小为(N, out_channels, w, h)
        """
        output = self.conv_block(input)

        return output
class DoubleConv(nn.Module):
    """
    卷积模块
    (convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class deepWBnet(nn.Module):
    def __init__(self):
        super(deepWBnet, self).__init__()
        self.n_channels = 3
        self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
        self.encoder_down1 = DownBlock(24, 48)
        self.encoder_down2 = DownBlock(48, 96)
        self.encoder_down3 = DownBlock(96, 192)
        self.encoder_bridge_down = BridgeDown(192, 384)
        self.decoder_bridge_up = BridgeUP(384, 192)
        self.decoder_up1 = UpBlock(192, 96)
        self.decoder_up2 = UpBlock(96, 48)
        self.decoder_up3 = UpBlock(48, 24)
        self.decoder_out = OutputBlock(24, self.n_channels)


    def forward(self, x):
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        x = self.decoder_bridge_up(x5)
        x = self.decoder_up1(x, x4)
        x = self.decoder_up2(x, x3)
        x = self.decoder_up3(x, x2)
        out = self.decoder_out(x, x1)
        return out
class Generator(nn.Module):
    """
    生成器模型，其结构与SRResNet完全一致.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        参数 large_kernel_size：第一层和最后一层卷积核大小
        参数 small_kernel_size：中间层卷积核大小
        参数 n_channels：中间层卷积通道数
        参数 n_blocks: 残差模块数量
        参数 scaling_factor: 放大比例
        """
        super(Generator, self).__init__()
        self.net=deepWBNet()
        #self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
        #                    n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)

    def forward(self, lr_imgs):
        """
        前向传播.

        参数 lr_imgs: 低精度图像 (N, 3, w, h)
        返回: 超分重建图像 (N, 3, w * scaling factor, h * scaling factor)
        """
        sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return sr_imgs


class Discriminator(nn.Module):
    """
    判别器
    """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        """
        参数 kernel_size: 所有卷积层的核大小
        参数 n_channels: 初始卷积层输出通道数, 后面每隔一个卷积层通道数翻倍
        参数 n_blocks: 卷积块数量
        参数 fc_size: 全连接层连接数
        """
        super(Discriminator, self).__init__()

        in_channels = 3

        # 卷积系列，参照论文SRGAN进行设计
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # 固定输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)

        # 最后不需要添加sigmoid层，因为PyTorch的nn.BCEWithLogitsLoss()已经包含了这个步骤

    def forward(self, imgs):
        """
        前向传播.

        参数 imgs: 用于作判别的原始高清图或超分重建图，张量表示，大小为(N, 3, w * scaling factor, h * scaling factor)
        返回: 一个评分值， 用于判断一副图像是否是高清图, 张量表示，大小为 (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit

class TruncatedVGG19(nn.Module):
    """
    truncated VGG19网络，用于计算VGG特征空间的MSE损失
    """

    def __init__(self, i, j):
        """
        :参数 i: 第 i 个池化层
        :参数 j: 第 j 个卷积层
        """
        super(TruncatedVGG19, self).__init__()

        # 加载预训练的VGG模型
        vgg19 = torchvision.models.vgg19(pretrained=True)  # C:\Users\Administrator/.cache\torch\checkpoints\vgg19-dcbb9e9d.pth

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # 迭代搜索
        for layer in vgg19.features.children():
            truncate_at += 1

            # 统计
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # 截断位置在第(i-1)个池化层之后（第 i 个池化层之前）的第 j 个卷积层
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # 检查是否满足条件
        assert maxpool_counter == i - 1 and conv_counter == j, "当前 i=%d 、 j=%d 不满足 VGG19 模型结构" % (
            i, j)

        # 截取网络
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        前向传播
        参数 input: 高清原始图或超分重建图，张量表示，大小为 (N, 3, w * scaling factor, h * scaling factor)
        返回: VGG19特征图，张量表示，大小为 (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output
