import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, mean=0., std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):

        if self.training:  # 仅在模型训练时添加噪声
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x

