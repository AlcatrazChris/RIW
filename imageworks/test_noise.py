import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from noise.jpeg_compression import JpegCompression
from noise.guassian import GaussianNoise
from noise.resize import Resize
from noise.cropout import Cropout
from noise.dropout import Dropout
from noise.rotate import RotateImage
from network.LowPassfitter import LowpassFilter

def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension

# Save a tensor to an image file
def save_image(tensor, filename):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    transform = transforms.ToPILImage()
    pil_image = transform(tensor)
    pil_image.save(filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
jpeg_compression = JpegCompression(device)
guassian = GaussianNoise(mean=0.,std=0.1)
resize = Resize((0.5,0.7))
cropout  =Cropout()
dropout = Dropout(keep_ratio_range=(0.93, 0.95))
rotate = RotateImage(30)
lfp = LowpassFilter(kernel_size=5)

image_path = 'D:/05_Learning_file/_毕业设计/结果/3_提取.png'
image_tensor = load_image(image_path).to(device)

compressed_image =dropout(image_tensor)
output_path = 'noise/JPEG.jpg'
save_image(compressed_image.cpu(), output_path)

original_img = Image.open(image_path)
compressed_img = Image.open(output_path)

plt.figure(figsize=(7, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Noised Image")
plt.imshow(compressed_img)
plt.axis('off')
plt.show()