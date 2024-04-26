import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from network.denoiser import DenoiseNet
from noise.noiser import Noiser
from noise.dropout import Dropout
from noise.cropout import Cropout
from noise.guassian import GaussianNoise
from noise.jpeg_compression import JpegCompression
from network.LowPassfitter import LowpassFilter
from noise.rotate import RotateImage
from noise.resize import Resize
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from metrics import Metrics
import numpy as np

# 数据预处理：添加随机噪声
def noisy_data(data):
    noise = torch.randn_like(data) * 0.4  # 调整噪声强度
    noisy_data = data + noise
    return noisy_data.clamp(0, 1)

# 数据加载
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

train_images = datasets.ImageFolder('../dataset/train/', transform)
train_loader = DataLoader(train_images, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化去噪网络
denoise_model = DenoiseNet(input_channels=3, output_channels=3).to(device)
optimizer = optim.Adam(denoise_model.parameters(), lr=0.001)
criterion = nn.MSELoss(reduction='sum')

save_model_dir = "model/pretrain"

# 训练循环
def train(model, device, train_loader, optimizer, epoch, log_interval=100):

    model.train()
    PSNR = []
    SSIM = []
    BER = []
    for batch_idx, (data, target) in enumerate(train_loader):

        # 添加噪声到输入数据
        noise = Noiser('random')
        noise.add_noise_layer(layer=GaussianNoise())
        noise.add_noise_layer(layer=JpegCompression(device))
        noise.add_noise_layer(layer=Dropout(keep_ratio_range=(0.4, 0.6)))
        noise.add_noise_layer(layer=LowpassFilter(kernel_size=3))
        # noise.add_noise_layer(layer=Cropout(0.3,0.7))
        # noise.add_noise_layer(layer=Resize((0.5,1.5)))
        data = noise(data)
        noisy_data = data.to(device)
        target = data.to(device)  # 原始干净图像作为目标

        optimizer.zero_grad()
        output = model(noisy_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            if batch_idx % log_interval == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                # 保存模型
                torch.save(model.state_dict(), f'{save_model_dir}/model_epoch_{epoch}_batch_{batch_idx}.pth')

                # 可视化第一个样本的输入和输出
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                # 显示输入图像
                axs[0].imshow(noisy_data[0].cpu().detach().numpy().transpose(1, 2, 0))
                axs[0].set_title('Input Image')
                axs[0].axis('off')
                # 显示输出图像
                axs[1].imshow(output[0].cpu().detach().numpy().transpose(1, 2, 0))
                axs[1].set_title('Output Image')
                axs[1].axis('off')
                plt.show()

                # 评估指标
                save_image(noisy_data[0], 'temp_input.png')
                save_image(output[0], 'temp_output.png')
                metrics = Metrics(Image.open('temp_input.png'), Image.open('temp_output.png'))
                psnr = metrics.psnr()
                PSNR.append(psnr)
                ssim = metrics.ssim()
                SSIM.append(ssim)
                ber = metrics.ber()
                BER.append(ber)
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            print(
                f'Average PSNR: {np.mean(PSNR):.2f}, Average SSIM: {np.mean(SSIM):.4f}, Average BER: {np.mean(BER):.4f}')


# 运行训练
for epoch in range(1, 11):
    train(denoise_model, device, train_loader, optimizer, epoch)
