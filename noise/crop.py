import torch
import torch.nn as nn

def random_float(min, max):
    """
    Return a random number using PyTorch
    """
    return torch.rand(1).item() * (max - min) + min

def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image using PyTorch operations.
    """
    _, _, image_height, image_width = image.size()

    remaining_height_ratio = random_float(height_ratio_range, height_ratio_range)
    remaining_width_ratio = random_float(width_ratio_range, width_ratio_range)

    remaining_height = int(torch.round(torch.tensor(remaining_height_ratio) * image_height).item())
    remaining_width = int(torch.round(torch.tensor(remaining_width_ratio) * image_width).item())

    height_start = torch.randint(0, image_height - remaining_height + 1, (1,)).item()
    width_start = torch.randint(0, image_width - remaining_width + 1, (1,)).item()

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width

class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right.
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        super(Crop, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)

        # Crop the image
        cropped_image = noised_image[:, :, h_start:h_end, w_start:w_end]

        # It's more common in PyTorch to return a new tensor rather than modifying in place
        noised_and_cover = cropped_image

        return noised_and_cover
