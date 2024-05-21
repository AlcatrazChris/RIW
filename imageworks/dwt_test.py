import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from network.waveletTrans import DWT# 假设你已经有了这个模块


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min)


def save_combined_image(images, filename):
    # 假设所有图像的大小相同
    height, width, _ = images[0].shape
    combined_image = np.zeros((height * 2, width * 2, 3))  # 创建足够大的图像以容纳所有拼接的图像

    # 将每个图像放在它们应该在的位置
    combined_image[0:height, 0:width, :] = normalize_image(images[0])
    combined_image[0:height, width:width * 2, :] = normalize_image(images[1])
    combined_image[height:height * 2, 0:width, :] = normalize_image(images[2])
    combined_image[height:height * 2, width:width * 2, :] = normalize_image(images[3])

    plt.imsave(filename, combined_image)


# 加载和转换图像
output_dir = 'dwt_result/'
os.makedirs(output_dir, exist_ok=True)
image_path = 'lena_mini.jpg'  # 替换为你的图片路径
image = Image.open(image_path)  # 不转换为灰度图
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
img_tensor = transform(image).unsqueeze(0)  # 添加批次维度


# 应用 DWT
dwt = DWT()
transformed = dwt(img_tensor)

# 假设 transformed 是一个批次大小为1，通道数为C的输出，每个通道代表不同的变换结果
C = transformed.size(1)
transformed_imgs = transformed.squeeze(0).detach().numpy()  # 移除批次维度并转换为NumPy数组
transformed_imgs = transformed_imgs.reshape(4, 3, transformed_imgs.shape[1], transformed_imgs.shape[2])

# 显示图像和变换结果
fig, ax = plt.subplots(2, 3, figsize=(24, 16))  # 你可以调整figsize以更好地填满屏幕
ax[0, 0].imshow(image)
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

titles = ['LL', 'HL', 'LH', 'HH']  # 根据实际情况调整
positions = [(0, 1), (0, 2), (1, 1), (1, 2)]  # 指定的位置

for i in range(4):
    img = transformed_imgs[i].transpose(1, 2, 0)

    # 对于RGB图像进行归一化并保存
    img_normalized = normalize_image(img)
    plt.imsave(os.path.join(output_dir, f'{titles[i]}_RGB.png'), img_normalized)

    # 对于单通道图像，同样进行归一化
    for channel, color in enumerate(['R', 'G', 'B']):
        single_channel_img = img[:, :, channel]
        single_channel_img_normalized = normalize_image(single_channel_img)
        plt.imsave(os.path.join(output_dir, f'{titles[i]}_{color}.png'), single_channel_img_normalized)

print("所有图像已保存在dwt_result文件夹中。")
