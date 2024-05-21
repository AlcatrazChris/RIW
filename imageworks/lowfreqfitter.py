from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# 图像路径
image_path = 'D:/05_Learning_file/_毕业设计/结果/水印.png'

# 读取图像
try:
    img = Image.open(image_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot open/read image file at: {image_path}")

# 应用盒滤波器（模糊效果）
img_blur = img.filter(ImageFilter.BoxBlur(2))  # 参数越大模糊效果越强

# 显示原图像和模糊后的图像
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(img_blur)
axes[1].set_title('Blurred Image')
axes[1].axis('off')

plt.show()

# 保存处理后的图像
img_blur.save('D:/05_Learning_file/_毕业设计/结果/3_提取.png')
print('Processed image saved.')
