import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from network.covbnrelu import ConvBNReLU

# 使用之前定义的 ConvBNReLU 结构
class DenoiseNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DenoiseNet, self).__init__()
        # 假设去噪网络有四层卷积
        self.model = ConvBNReLU(input_channels, output_channels, num_layers=4)

    def forward(self, x):
        return self.model(x)

