from model import *
from network.invblock import INV_block


class InvNet(nn.Module):

    def __init__(self, nLayer):
        super(InvNet, self).__init__()
        self.inv = nn.ModuleList([INV_block() for _ in range(nLayer)])

    def forward(self, x, rev=False):
        out = x
        if not rev:
            for inv in self.inv:
                out = inv(out)
        else:
            for inv in reversed(self.inv):
                out = inv(out, rev=True)
        return out


