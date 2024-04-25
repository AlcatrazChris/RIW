import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from PIL import Image

# 载入图像
def load_image(path):
    # 以灰度模式读取图像
    image = Image.open(path).convert('L')
    # 转换为numpy数组
    image_np = np.array(image)
    return image_np

# 将图像转换为频域图像
def image_to_frequency(image_np):
    # 对图像应用傅里叶变换
    frequency_transform = fft2(image_np)
    # 将DC分量移到频谱中心
    frequency_shifted = fftshift(frequency_transform)
    # 计算幅度谱
    magnitude_spectrum = np.log(np.abs(frequency_shifted) + 1)
    return magnitude_spectrum

# 显示图像和它的频域表示
def display_images(original, frequency):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Frequency Domain Image')
    plt.imshow(frequency, cmap='gray')
    plt.axis('off')

    plt.show()

# 主程序
def main(image_path):
    # 载入图像
    original_image = load_image(image_path)
    # 转换到频域
    frequency_image = image_to_frequency(original_image)
    # 展示图像
    display_images(original_image, frequency_image)

# 这里替换成你的图像路径
image_path = '../result/val_images/steg/steg_2_001.png'
main(image_path)
