import torch
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from network.LowPassfitter import LowpassFilter

def extract_texture(image):
    # 定义Sobel算子核
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 保证输入图像和核在同一个设备上
    sobel_kernel_x = sobel_kernel_x.to(image.device)
    sobel_kernel_y = sobel_kernel_y.to(image.device)

    # 检查图像通道数，并应用Sobel核
    if image.size(1) == 1:  # 单通道图像
        edge_x = F.conv2d(image, sobel_kernel_x, padding=1, groups=1)
        edge_y = F.conv2d(image, sobel_kernel_y, padding=1, groups=1)
    else:  # 多通道图像
        edge_x = F.conv2d(image, sobel_kernel_x.repeat(image.size(1), 1, 1, 1), padding=1, groups=image.size(1))
        edge_y = F.conv2d(image, sobel_kernel_y.repeat(image.size(1), 1, 1, 1), padding=1, groups=image.size(1))

    # 计算梯度幅度
    texture = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
    return texture


class TextureLoss(nn.Module):
    def __init__(self):
        super(TextureLoss, self).__init__()

    def forward(self, input_image, target_image):
        # 提取输入和目标图像的纹理
        input_texture = extract_texture(input_image)
        target_texture = extract_texture(target_image)

        # 计算纹理图像间的MSE损失
        mse_loss = F.mse_loss(input_texture, target_texture, reduction='sum')

        return mse_loss

def display_images(image_path1, image_path2):
    lfp = LowpassFilter(device='cpu')
    # 加载图像
    rgb_tensor1 = load_image(image_path1)
    rgb_tensor2 = load_image(image_path2)

    # 提取纹理
    # texture1 = extract_texture(rgb_tensor1)
    # texture2 = extract_texture(rgb_tensor2)
    texture1 = lfp(rgb_tensor1)
    texture2 = lfp(rgb_tensor2)

    # 计算原始图像差异
    image_diff = torch.abs(rgb_tensor1 - rgb_tensor2)

    # 计算纹理差异
    texture_diff = torch.abs(texture1 - texture2)
    # 准备绘图
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    axes = axes.ravel()

    axes[0].imshow(rgb_tensor1.squeeze(0).permute(1, 2, 0))
    axes[0].set_title('Original Image 1')
    axes[0].axis('off')

    axes[1].imshow(rgb_tensor2.squeeze(0).permute(1, 2, 0))
    axes[1].set_title('Original Image 2')
    axes[1].axis('off')

    axes[2].imshow(texture1.squeeze(0).permute(1, 2, 0).detach().numpy(), cmap='gray')
    axes[2].set_title('Texture Image 1')
    axes[2].axis('off')

    axes[3].imshow(texture2.squeeze(0).permute(1, 2, 0).detach().numpy(), cmap='gray')
    axes[3].set_title('Texture Image 2')
    axes[3].axis('off')
    # 显示差异图像
    axes[4].imshow(image_diff.squeeze(0).permute(1, 2, 0))
    axes[4].set_title('Difference Image')
    axes[4].axis('off')

    axes[5].imshow(texture_diff.squeeze(0).permute(1, 2, 0).detach().numpy(), cmap='gray')
    axes[5].set_title('Texture Difference Image')
    axes[5].axis('off')

    plt.show()

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    return image_tensor

# image_path1 = '../runs/val_20240508_165638/cover/cover_1_000.png'
# image_path2 = "../runs/val_20240508_165638/steg/steg_2_000.png"
#
# display_images(image_path1, image_path2)
#
# # 实例化损失函数
# texture_loss_fn = TextureLoss()
# # 假设 input_rgb 和 target_rgb 已经是加载和预处理后的张量
# loss = texture_loss_fn(rgb_tensor1, rgb_tensor2)
# print("Computed Texture Loss:", loss.item())
