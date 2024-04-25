import torch
import torch.nn as nn
import numpy as np

class Dropout(nn.Module):
    def __init__(self, keep_ratio_range = (0.5,0.8)):
        super(Dropout, self).__init__()
        self.keep_min = keep_ratio_range[0]
        self.keep_max = keep_ratio_range[1]

    def forward(self, steg):

        noised_image = steg
        cover_image = torch.zeros_like(noised_image)
        batch_size = cover_image.size(0)
        random_index = torch.randint(0, batch_size, (1,)).item()
        cover_image = cover_image[random_index].unsqueeze(0)
        mask_percent = np.random.uniform(self.keep_min, self.keep_max)

        mask = np.random.choice([0.0, 1.0], noised_image.shape[2:], p=[1 - mask_percent, mask_percent])
        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
        mask_tensor = mask_tensor.expand_as(noised_image)
        noised_image = noised_image * mask_tensor + cover_image * (1-mask_tensor)
        return noised_image