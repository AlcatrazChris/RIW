import torch
import torch.nn as nn
import random

class Noiser(nn.Module):
    def __init__(self, mode = 'none', layer_num=2):
        super(Noiser, self).__init__()
        self.mode = mode
        self.layer_num = layer_num
        self.noise_layers = nn.ModuleList([])

    def add_noise_layer(self, layer):
        self.noise_layers.append(layer)

    def forward(self, x):
        n, c, h, w = x.shape
        processed_images = []

        # 遍历每个图像
        for i in range(n):
            image = x[i, :, :, :].unsqueeze(0)  # 保持维度为[1, C, H, W]

            # 获取层的索引
            indices = list(range(len(self.noise_layers)))

            if self.mode == 'random':
                # 如果是random模式，随机选择指定数量的层
                if len(indices) >= self.layer_num:
                    chosen_indices = random.sample(indices, self.layer_num)
                else:
                    chosen_indices = indices  # 如果层的数量少于指定数量，则选择所有层
            else:
                # 如果不是random模式，随机排序所有层
                random.shuffle(indices)
                chosen_indices = indices

            # 应用选择的噪声层
            for idx in chosen_indices:
                image = self.noise_layers[idx](image)

            processed_images.append(image)

        # 将处理后的图像列表重新组合为一个tensor
        processed_images_tensor = torch.cat(processed_images, dim=0)

        return processed_images_tensor
