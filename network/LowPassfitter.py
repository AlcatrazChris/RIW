import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from torchvision import transforms,utils
from PIL import Image


# Define a function to load an image and convert it to a tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

class LowpassFilter(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0, device=None):
        super(LowpassFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create and register Gaussian kernel as a parameter
        self.weight = self.create_gauss_kernel(kernel_size, sigma)
        self.weight = nn.Parameter(self.weight)  # Register as a Parameter
        self.to(self.device)  # Ensure the entire module is on the correct device

    def create_gauss_kernel(self, kernel_size, sigma):
        # Generate a Gaussian kernel
        intervals = torch.linspace(-1.5*sigma, 1.5*sigma, steps=kernel_size)
        x = intervals.unsqueeze(1)
        y = intervals.unsqueeze(0)
        kernel_2d = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel_2d /= kernel_2d.sum()
        return kernel_2d.view(1, 1, kernel_size, kernel_size)  # Shape the kernel for conv2d

    def forward(self, x):
        # Apply the filter to input
        weight = self.weight.repeat(x.size(1), 1, 1, 1)  # Adapt weight for input channels
        return F.conv2d(x, weight, padding=self.padding, groups=x.size(1))


