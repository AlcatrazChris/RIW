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
        self.conv1 = ConvBNReLU(input_channels, 32)
        self.conv2 = ConvBNReLU(32, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv3 = ConvBNReLU(64, 128)
        self.conv4 = ConvBNReLU(128, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBNReLU(64, 32)
        self.up2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv6 = ConvBNReLU(32, output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, indices1 = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x, indices2 = self.pool2(x)
        x = self.up1(x, indices2)
        x = self.conv5(x)
        x = self.up2(x, indices1)
        x = self.conv6(x)
        return x

