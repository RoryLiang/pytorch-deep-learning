import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_height, input_width, input_channel, latent_dim=15) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(input_height*input_width*input_channel), out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_height, input_width, input_channel, latent_dim=15) -> None:
        super().__init__()
        self.input_height = input_height
        self.input_width = input_height
        self.input_channel = input_channel
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=(input_height*input_width*input_channel))
        )

    def forward(self, x):
        return torch.reshape(self.decoder(x), (x.size(0),
                                               self.input_channel,
                                               self.input_height,
                                               self.input_width))


class Discriminator(nn.Module):
    def __init__(self, latent_dim=15) -> None:
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)
