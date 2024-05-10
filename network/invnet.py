from network.invblock import INV_block
import torch.nn as nn
from network.VAE import VariationalAutoencoder
from network.denoiser import DenoiseNet
from network.selfAttention import SelfAttentionLayer

class InvNet(nn.Module):
    def __init__(self, nLayer, input_dim = 576):
        super(InvNet, self).__init__()
        self.inv_blocks = nn.ModuleList([INV_block() for _ in range(nLayer)])
        # self.denoiser = DenoiseNet(24,24)
        # self.vae = VariationalAutoencoder( hidden_dim=16,latent_dim=64)
        self.attention_forward = SelfAttentionLayer(embed_dim=24, num_heads=8)
        self.attention_backward = SelfAttentionLayer(embed_dim=24, num_heads=8)

    def forward(self, x, rev=False):
        if not rev:
            for i, block in enumerate(self.inv_blocks):
                x = block(x)
                if i == len(self.inv_blocks) // 2:  # 在中间层后添加自注意力
                    x = self.attention_forward(x)
        else:
            for i, block in enumerate(reversed(self.inv_blocks)):
                x = block(x, rev=True)
                if i == len(self.inv_blocks) // 2:  # 在中间层后添加自注意力
                    x = self.attention_backward(x)
        return x

# Example usage
