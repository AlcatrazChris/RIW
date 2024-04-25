import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from noise.crop import random_float

class Resize(nn.Module):
    """
    Resize the image to a random scale and then pad it to maintain the original size.
    The target size is original size * resize_ratio, and padding is applied to compensate
    for size reductions.
    """
    def __init__(self, resize_ratio_range, interpolation_method='nearest', padding_value=0):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method
        self.padding_value = padding_value

    def forward(self, steg):
        # Calculate random resize ratio
        resize_ratio = torch.rand(1).item() * (self.resize_ratio_max - self.resize_ratio_min) + self.resize_ratio_min

        # Resize the image
        resized_image = F.interpolate(steg, scale_factor=(resize_ratio, resize_ratio), mode=self.interpolation_method)

        # Calculate padding to restore original dimensions
        original_height, original_width = steg.shape[2], steg.shape[3]
        resized_height, resized_width = resized_image.shape[2], resized_image.shape[3]

        # Padding sizes
        pad_height = (original_height - resized_height + 1) // 2
        pad_width = (original_width - resized_width + 1) // 2

        # Apply padding
        padded_image = F.pad(resized_image, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=self.padding_value)

        # If necessary, correct for any off-by-one dimensions due to uneven padding requirements
        if padded_image.shape[2] != original_height or padded_image.shape[3] != original_width:
            padded_image = F.pad(padded_image,
                                 (0, original_width - padded_image.shape[3], 0, original_height - padded_image.shape[2]),
                                 mode='constant', value=self.padding_value)
        # print(padded_image.shape)

        return padded_image