import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG16_Weights
from torchvision.transforms import Normalize

class VGGColorLoss(nn.Module):
    def __init__(self):
        super(VGGColorLoss, self).__init__()
        # 使用新的权重参数方法
        weights = VGG16_Weights.DEFAULT
        vgg = models.vgg16(weights=weights)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:10])  # 取前10层
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # 冻结模型参数

    def forward(self, input, target):
        # 预处理，确保输入符合VGG的预期
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input = normalize(input)
        target = normalize(target)

        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)

        # 计算特征之间的L2损失
        loss = torch.nn.functional.mse_loss(input_features, target_features)
        return loss

if __name__ == "__main__":
    loss_model = VGGColorLoss()
    input_tensor = torch.rand(10, 3, 224, 224)
    target_tensor = torch.rand(10, 3, 224, 224)
    loss = loss_model(input_tensor, target_tensor)
    print(f"Loss: {loss.item()}")

