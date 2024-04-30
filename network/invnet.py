from network.invblock import INV_block
import torch.nn as nn
from network.DNAutoencoer import DenoisingAutoencoder
from network.denoiser import DenoiseNet

class InvNet(nn.Module):
    def __init__(self, nLayer, input_dim):
        super(InvNet, self).__init__()
        self.inv_blocks = nn.ModuleList([INV_block() for _ in range(nLayer)])
        self.denoiser = DenoiseNet(24,24)

    def forward(self, x, rev=False):
        if not rev:
            for block in self.inv_blocks:
                x = block(x)
        else:
            for block in reversed(self.inv_blocks):
                x = block(x, rev=True)
        return x

# Example usage
