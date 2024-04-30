import torch.nn as nn
from network.invnet import InvNet


class Model(nn.Module):
    def __init__(self, nLayer = 16):
        super(Model, self).__init__()

        self.model = InvNet(nLayer,input_dim=256)

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out


