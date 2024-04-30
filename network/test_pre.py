import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from network.denoiser import DenoiseNet
from noise.noiser import Noiser
from noise.dropout import Dropout
from noise.guassian import GaussianNoise
from noise.jpeg_compression import JpegCompression
from network.LowPassfitter import LowpassFilter
from metrics import Metrics
import sys

sys.path.append('network/')

def load_model(model_path, input_channels, output_channels, device):
    model = DenoiseNet(input_channels, output_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),  # Resize the image to fit the model input
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def denoise_and_evaluate(model, noisy_image, device):
    noisy_image = noisy_image.to(device)
    with torch.no_grad():
        output = model(noisy_image)
    return output

def save_and_display_images(original, noisy, denoised, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    imgs = [original.squeeze().cpu().detach().numpy(), noisy.squeeze().cpu().detach().numpy(), denoised.squeeze().cpu().detach().numpy()]
    titles = ['Original', 'Noisy', 'Denoised']
    for ax, img, title in zip(axs, imgs, titles):
        ax.imshow(np.transpose(img, (1, 2, 0)))  # Convert from CHW to HWC format for visualization
        ax.set_title(title)
        ax.axis('off')
    plt.savefig(save_path)
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '../model/pretrain/model_epoch_100.pth'
    image_path = '../dataset/test/images/000.png'
    save_image_path = '../runs/pretrain/comparison_image.png'

    # Load the model
    model = load_model(model_path, input_channels=3, output_channels=3, device=device)

    # Load and preprocess the image
    original_image = preprocess_image(image_path).to(device)

    # Assume you add noise here for testing or load a noisy version directly
    noisy_image = original_image.clone()  # Simulating a noisy image, replace with actual noisy image if available

    # Denoise the image
    noise = Noiser()
    noise.add_noise_layer(layer=GaussianNoise(mean=0., std= 0.1))
    noise.add_noise_layer(layer=JpegCompression(device))
    noise.add_noise_layer(layer=Dropout(keep_ratio_range=(0.3, 0.5)))
    # noise.add_noise_layer(layer=LowpassFilter(kernel_size=3))
    noisy_data = noise(noisy_image)
    denoised_image = denoise_and_evaluate(model, noisy_data, device)

    # Save and display images
    save_and_display_images(original_image, noisy_data, denoised_image, save_image_path)

    # Calculate and print PSNR
    original_pil = Image.fromarray((255 * original_image.squeeze().cpu().detach().permute(1, 2, 0).numpy()).astype(np.uint8))
    noisy_pil = Image.fromarray((255 * noisy_data.squeeze().cpu().detach().permute(1, 2, 0).numpy()).astype(np.uint8))
    denoised_pil = Image.fromarray((255 * denoised_image.squeeze().cpu().detach().permute(1, 2, 0).cpu().numpy()).astype(np.uint8))
    metrics = Metrics(original_pil, denoised_pil)
    metrics1 = Metrics(original_pil, noisy_pil)
    psnr = metrics.psnr()
    psnr1 = metrics1.psnr()
    print(f'PSNR_noised: {psnr1},PSNR_denoised: {psnr}')

if __name__ == "__main__":
    main()
