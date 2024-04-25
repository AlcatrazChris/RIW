import torch
import torch.nn as nn
import torchvision.transforms as transforms

class RotateImage(nn.Module):
    def __init__(self, degrees):
        super(RotateImage, self).__init__()
        self.degrees = degrees
        # RandomRotation将在指定的度数范围内随机选择一个角度进行旋转
        self.rotate = transforms.RandomRotation(degrees=self.degrees)

    def forward(self, x):
        if self.training:  # 仅在模型训练时旋转图片
            # 逐个处理batch中的每个图像
            batch_size = x.size(0)
            rotated_images = []
            for i in range(batch_size):
                # 将单个图像从Tensor转换为PIL图像进行旋转
                x_pil = transforms.functional.to_pil_image(x[i])
                x_rotated_pil = self.rotate(x_pil)
                # 将旋转后的PIL图像转换回Tensor
                x_rotated = transforms.functional.to_tensor(x_rotated_pil)
                rotated_images.append(x_rotated)

            # 将处理过的图像列表转换为一个Tensor
            return torch.stack(rotated_images)
        return x
