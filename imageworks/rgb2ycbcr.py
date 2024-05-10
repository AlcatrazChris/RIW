import torch
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def rgb2ycbcr(rgb_image):
    # 校验输入图片的shape是否为[1, C, W, H]
    if rgb_image.dim() != 4 or rgb_image.size(1) != 3:
        raise ValueError("输入的tensor格式应为[1, 3, W, H]")

    # 定义转换矩阵，按ITU标准
    matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ]).to(rgb_image.device)

    # 偏移量
    shift = torch.tensor([0, 128, 128]).to(rgb_image.device)

    # 调整输入格式以符合矩阵乘法的要求，[1, W, H, 3]
    rgb_image = rgb_image.permute(0, 2, 3, 1)

    # 应用矩阵变换
    ycbcr_image = torch.tensordot(rgb_image, matrix, dims=([-1], [1]))

    # 添加偏移量
    ycbcr_image += shift.view(1, 1, 1, -1)

    # 调整输出格式为[1, C, W, H]
    ycbcr_image = ycbcr_image.permute(0, 3, 1, 2)

    # 确保输出tensor格式正确
    return ycbcr_image

class ColorWeightedMSELoss(nn.Module):
    def __init__(self, g_function):
        super(ColorWeightedMSELoss, self).__init__()
        self.g_function = g_function

    def forward(self, input_rgb, target_rgb):
        # 转换到YCbCr色彩空间
        input_ycbcr = rgb2ycbcr(input_rgb)
        target_ycbcr = rgb2ycbcr(target_rgb)

        # 提取Cr分量
        I_cr = input_ycbcr[:, 2, :, :]

        # 归一化Cr分量
        I_cr_normalized = self.g_function(I_cr)

        # 计算色度权重
        A = torch.ones_like(I_cr_normalized)
        M_color = A - torch.exp(-I_cr_normalized)

        # 计算MSE损失
        mse_loss = F.mse_loss(input_rgb, target_rgb, reduction='none')
        color_weighted_loss = mse_loss * M_color.unsqueeze(1)  # 广播权重到所有颜色通道

        # 返回加权后的总损失
        return color_weighted_loss.sum()



def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    return image_tensor

def plot_images(rgb_tensor, ycbcr_tensor):
    # 配置matplotlib
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    # 展示RGB图片
    ax[0][0].imshow(rgb_tensor.squeeze(0).permute(1, 2, 0))
    ax[0][0].set_title("Original RGB Image")
    ax[0][0].axis('off')

    # 展示YCbCr的三个通道
    labels = ['Y Channel', 'Cb Channel', 'Cr Channel']
    for i in range(3):
        if i == 0:
            ax[0][1].imshow(ycbcr_tensor.squeeze(0)[i])
            ax[0][1].set_title(labels[i])
            ax[0][1].axis('off')
        elif i == 1:
            ax[1][0].imshow(ycbcr_tensor.squeeze(0)[i])
            ax[1][0].set_title(labels[i])
            ax[1][0].axis('off')
        elif i == 2:
            ax[1][1].imshow(ycbcr_tensor.squeeze(0)[i])
            ax[1][1].set_title(labels[i])
            ax[1][1].axis('off')
    plt.show()


def normalize_cr(cr_channel):
    cr_min = cr_channel.min()
    cr_max = cr_channel.max()
    return (cr_channel - cr_min) / (cr_max - cr_min)
#
# # 载入两张图像进行测试
# image_path1 = '../runs/val_20240428_010957/cover/cover_1_000.png'
# image_path2 = "../runs/val_20240428_010957/steg/steg_1_000.png"
#
# # 读取和处理图像
# rgb_tensor1 = load_image(image_path1)
# rgb_tensor2 = load_image(image_path2)
#
# # 计算
# loss_fn = ColorWeightedMSELoss(g_function=normalize_cr)
#
# loss = loss_fn(rgb_tensor1, rgb_tensor2)
# print("Computed Loss:", loss.item())
# # image_path = "../dataset/train/images/0003.png"
# # rgb_tensor = load_image(image_path)
# # ycbcr_tensor = rgb2ycbcr(rgb_tensor)
# plot_images(rgb_tensor, ycbcr_tensor)
