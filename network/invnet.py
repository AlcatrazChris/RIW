from network.invblock import INV_block
import torch.nn as nn
from network.VAE import VariationalAutoencoder
from network.denoiser import DenoiseNet
from network.selfAttention import SelfAttentionLayer

class DenoisingModule(nn.Module):
    def __init__(self):
        super(DenoisingModule, self).__init__()
        self.conv1 = nn.Conv2d(24, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 24, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

class InvNet(nn.Module):
    def __init__(self, nLayer, input_dim = 576):
        super(InvNet, self).__init__()
        self.inv_blocks = nn.ModuleList([INV_block() for _ in range(nLayer)])
        # self.denoiser = DenoisingModule()
        # self.denoiser = DenoiseNet(24,24)
        # self.vae = VariationalAutoencoder( hidden_dim=16,latent_dim=64)
        # self.attention_forward = SelfAttentionLayer(embed_dim=24, num_heads=8)
        # self.attention_backward = SelfAttentionLayer(embed_dim=24, num_heads=8)

    def forward(self, x, rev=False):
        if not rev:
            for i, block in enumerate(self.inv_blocks):
                x = block(x)

        else:
            # x = self.denoiser(x)
            for i, block in enumerate(reversed(self.inv_blocks)):
                x = block(x, rev=True)
        return x

