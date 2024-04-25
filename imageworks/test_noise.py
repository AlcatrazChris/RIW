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
dropout = Dropout(keep_ratio_range=(0.5, 0.6))
rotate = RotateImage(30)

image_path = '../dataset/test/images/000.png'
image_tensor = load_image(image_path).to(device)

compressed_image =resize(image_tensor)
output_path = '../noise/dropout_test.png'
save_image(compressed_image.cpu(), output_path)

original_img = Image.open(image_path)
compressed_img = Image.open(output_path)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Resize Image")
plt.imshow(compressed_img)
plt.axis('off')
plt.show()