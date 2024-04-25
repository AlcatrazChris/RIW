import torch
import torch.nn as nn
import numpy as np

class Cropout(nn.Module):
    def __init__(self, height_ratio_range=(0.1, 0.3), width_ratio_range=(0.1, 0.2), num_blocks=1):
        super(Cropout, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range
        self.num_blocks = num_blocks

    def get_random_rectangle_inside(self, image, height_ratio_range, width_ratio_range):
        batch_size, _, height, width = image.size()
        rectangles = []
        for _ in range(self.num_blocks):
            h_start = np.random.randint(0, height * (1 - height_ratio_range[1]))
            h_end = np.random.randint(h_start + height * height_ratio_range[0], height)
            w_start = np.random.randint(0, width * (1 - width_ratio_range[1]))
            w_end = np.random.randint(w_start + width * width_ratio_range[0], width)
            rectangles.append((h_start, h_end, w_start, w_end))
        return rectangles

    def forward(self, noised_image):
        cropout_mask = torch.zeros_like(noised_image)
        rectangles = self.get_random_rectangle_inside(image=noised_image,
                                                      height_ratio_range=self.height_ratio_range,
                                                      width_ratio_range=self.width_ratio_range)

        for h_start, h_end, w_start, w_end in rectangles:
            cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        result_image = noised_image * (1 - cropout_mask)
        return result_image
