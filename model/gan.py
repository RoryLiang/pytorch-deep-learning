from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_height=28, img_width=28, img_channel=1) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, img_height*img_width*img_channel),
            nn.Tanh()
        )

    def forward(self, z):
        img_vec = self.generator(z)
        return img_vec.view(z.size(0), self.img_channel, self.img_height, self.img_width)


class Discriminator(nn.Module):
    def __init__(self, latent_dim=100, img_height=28, img_width=28, img_channel=1) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_height*img_width*img_channel, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        pred_label = self.discriminator(img)
        return pred_label
