import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from network.denoiser import DenoiseNet
from noise.noiser import Noiser
from noise.dropout import Dropout
from noise.guassian import GaussianNoise
from noise.jpeg_compression import JpegCompression
from network.LowPassfitter import LowpassFilter
from torchvision.utils import save_image
from metrics import Metrics
from tqdm import tqdm
from PIL import Image

# 设置超参数和设备
batch_size = 16
epochs = 500
lr = 0.01
save_model_dir = '../model/pretrain'
save_image_dir = '../runs/pretrain'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保图像保存路径存在
os.makedirs(save_image_dir, exist_ok=True)

# 数据预处理和加载
def load_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    train_images = datasets.ImageFolder('../dataset/train/', transform)
    train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
    return train_loader

# 可视化和指标评估
def visualize_and_metrics(epoch, noisy_data, output):
    # 确保张量在转换为NumPy数组前已经移至CPU
    noisy_img = noisy_data[0].cpu().detach().clamp(0, 1).numpy().transpose(1, 2, 0)
    output_img = output[0].cpu().detach().clamp(0, 1).numpy().transpose(1, 2, 0)

    # 绘制并保存图像而不是显示
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(noisy_img)
    axs[0].set_title('Noisy Image')
    axs[0].axis('off')

    axs[1].imshow(output_img)
    axs[1].set_title('Denoised Image')
    axs[1].axis('off')

    # 图像保存路径
    plt.savefig(os.path.join(save_image_dir, f'comparison_epoch_{epoch}.png'))
    plt.close()

    # 指标计算
    noisy_pil = Image.fromarray((noisy_img * 255).astype(np.uint8))
    output_pil = Image.fromarray((output_img * 255).astype(np.uint8))
    metrics = Metrics(noisy_pil, output_pil)
    psnr = metrics.psnr()
    ssim = metrics.ssim()
    ber = metrics.ber()
    print(f'Epoch {epoch} - PSNR: {psnr}, SSIM: {ssim}, BER: {ber}')


# 训练循环
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
        data = data.to(device)
        noise = Noiser('random')
        if epoch % 3 == 0 and epoch != 0:
            pass
        else:
            noise.add_noise_layer(layer=GaussianNoise())
            noise.add_noise_layer(layer=JpegCompression(device))
            noise.add_noise_layer(layer=Dropout(keep_ratio_range=(0.4, 0.6)))
            noise.add_noise_layer(layer=LowpassFilter(kernel_size=3))
        noisy_data = noise(data)
        target = data

        optimizer.zero_grad()
        output = model(noisy_data)
        loss = nn.MSELoss(reduction='sum')(output, target)
        loss.backward()
        optimizer.step()

        # 每5个epoch并且是epoch中的第一个batch，进行图像保存和指标评估
        if epoch % 5 == 0 and batch_idx == 0:
            visualize_and_metrics(epoch, noisy_data, output)

    # 保存模型状态
    torch.save(model.state_dict(), os.path.join(save_model_dir, f'model_epoch_{epoch}.pth'))


# 主函数
if __name__ == "__main__":
    train_loader = load_data()
    model = DenoiseNet(input_channels=3, output_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

