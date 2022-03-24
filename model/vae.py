import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.encode_nn = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128)
        )

        self.z_mean_generater = nn.Linear(in_features=128, out_features=latent_dim)
        self.z_logvar_generater = nn.Linear(in_features=128, out_features=latent_dim)

        self.decode_nn = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=28*28)
        )

    def sample_z_from_distribution(self, z_mean, z_logvar):  # z is sampled from lantent dim Joint Gaussian distribution
        epsilon = torch.rand_like(z_mean)  # sample e from a normal distribution
        return z_mean + epsilon * torch.exp(z_logvar/2.)  # trick to simplify computation

    def encoder(self, x):
        x = self.encode_nn(x)
        z_mean = self.z_mean_generater(x)
        z_logvar = self.z_logvar_generater(x)
        return z_mean, z_logvar

    def decoder(self, x):
        return self.decode_nn(x)

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z_sampled = self.sample_z_from_distribution(z_mean, z_logvar)
        decoder_out = self.decoder(z_sampled)
        return z_mean, z_logvar, decoder_out
