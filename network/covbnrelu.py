import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=4, bias=True):
        super(ConvBNReLU, self).__init__()
        self.layers = nn.Sequential()

        # 设置中间层的输出通道数
        intermediate_channels = 32

        # 添加初始卷积层
        self.layers.add_module("conv0",
                               nn.Conv2d(input_channels, intermediate_channels, kernel_size=3, stride=1, padding=1,
                                         bias=bias))
        self.layers.add_module("bn0", nn.BatchNorm2d(intermediate_channels))
        self.layers.add_module("relu0", nn.ReLU(inplace=True))

        # 添加中间层
        for i in range(1, num_layers - 1):
            self.layers.add_module(f"conv{i}",
                                   nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1,
                                             padding=1, bias=bias))
            self.layers.add_module(f"bn{i}", nn.BatchNorm2d(intermediate_channels))
            self.layers.add_module(f"relu{i}", nn.ReLU(inplace=True))

        # 添加最后一个卷积层，输出尺寸为 output_channels
        self.layers.add_module(f"conv{num_layers - 1}",
                               nn.Conv2d(intermediate_channels, output_channels, kernel_size=3, stride=1, padding=1,
                                         bias=bias))
        self.layers.add_module(f"bn{num_layers - 1}", nn.BatchNorm2d(output_channels))
        self.layers.add_module(f"relu{num_layers - 1}", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)
