import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from torchvision import transforms,utils
from PIL import Image


# Define a function to load an image and convert it to a tensor
def image_to_tensor(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Define the transformation to convert the image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Apply the transformation to the image
    image_tensor = transform(image)

    # Add a batch dimension [n, c, w, h]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor




class LowpassFilter(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
        super(LowpassFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        # Create a Gaussian kernel
        self.weight = self.create_gauss_kernel(kernel_size, sigma)

    def create_gauss_kernel(self, kernel_size, sigma):
        # Create a Gaussian kernel
        intervals = torch.linspace(-1.5*sigma, 1.5*sigma, steps=kernel_size)
        x = intervals.unsqueeze(1)
        y = intervals.unsqueeze(0)
        kernel_2d = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel_2d = kernel_2d / torch.sum(kernel_2d)
        return nn.Parameter(kernel_2d.view(1, 1, kernel_size, kernel_size), requires_grad=False)

    def forward(self, x):
        # Repeat the weight for each input channel and apply the filter
        weight = self.weight.repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, weight, padding=self.padding, groups=x.size(1))

# Example usage
if __name__ == "__main__":
    # Instantiate the Gaussian filter
    gaussian_filter = GaussianFilter(kernel_size=5, sigma=2.0)

    # Create a dummy tensor that simulates an image batch [n, c, w, h]
    n, c, w, h = 1, 3, 1024, 1024  # Example dimensions
    dummy_image = torch.rand(n, c, w, h)  # Random tensor simulating image data
    image_tensor = image_to_tensor('../result/val_images/steg/steg_7_000.png')
    # Apply the Gaussian filter to the tensor
    filtered_image = gaussian_filter(image_tensor)
    utils.save_image(filtered_image, '../result/val_images/lp.png')
    print("Filtered output shape:", filtered_image.shape)
