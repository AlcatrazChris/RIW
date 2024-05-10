import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        # 根据卷积输出调整输入尺寸
        self.fc1 = nn.Linear(24 * 128 * 128, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # 平整化卷积输出
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        reconstruction = torch.sigmoid(self.fc2(h))
        reconstruction = reconstruction.view(-1, 24, 128, 128)  # 重塑为原始尺寸
        return reconstruction

class VariationalAutoencoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, 24 * 128 * 128)  # 假设解码器输出尺寸与输入尺寸相同

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

def vae_loss(reconstruction, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div
