import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class LabLoss(nn.Module):
    def __init__(self):
        super(LabLoss, self).__init__()

    def forward(self, input, target):
        # 假设 input 和 target 已经转换为 Lab 色彩空间且是 [n, 3, w, h]
        # 计算 MSE 损失，使用 'sum' 来聚合误差
        return F.mse_loss(input, target, reduction='sum')


# 示例使用
if __name__ == "__main__":
    input_tensor = torch.rand(10, 3, 256, 256)  # 假设这是批次大小为10的Lab图像
    target_tensor = torch.rand(10, 3, 256, 256)

    loss_model = LabLoss()
    loss = loss_model(input_tensor, target_tensor)
    print(f"Loss (sum of squared errors): {loss.item()}")
