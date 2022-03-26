import itertools
import torch
import logging
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from model.aae import Encoder, Decoder, Discriminator
from utils.logging_ import init_logger


logger = logging.getLogger(__name__)


class VAEoptimizer():
    def __init__(self, args) -> None:
        self.args = args

    def optimize(self):
        train_dataset = datasets.MNIST(
            root=self.args.data_dir,
            train=True,
            transform=ToTensor(),
            download=True
        )
        valid_dataset = datasets.MNIST(
            root=self.args.data_dir,
            train=False,
            transform=ToTensor(),
            download=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )

        encoder = Encoder()
        decoder = Decoder()
        discriminator = Discriminator()
        device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
        init_logger(rank=0, filenmae=self.args.output_dir+"/default.log")
        logger.info(f"training on {device}")
        encoder.to(device)
        decoder.to(device)
        discriminator.to(device)
        writer = SummaryWriter("/".join([self.args.output_dir, "tb"]))

        recon_loss_fn = nn.MSELoss()
        gan_loss_fn = nn.BCELoss()
        recon_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=self.args.learning_rate)
        gan_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.args.learning_rate)

        for epoch in range(1, self.args.epoch_num+1):

            recon_epoch_loss = 0
            dis_epoch_loss = 0
            iteration_num = 0
            for batch_idx, (img, label) in enumerate(train_dataloader):

                # ae reconstruct phase
                z = encoder(img)
                z_label = torch.zeros(self.args.batch_size, device=device)
                recon_img = decoder(z)
                recon_loss = recon_loss_fn(img, recon_img)
                recon_epoch_loss += recon_loss

                recon_optimizer.zero_grad()
                recon_loss.backward()
                recon_optimizer.step()

                # train discriminator
                z_sampled = torch.randn(self.args.batch_size, z.size(1), device=device)
                z_sampled_label = torch.ones(self.args.batch_size, device=device)

                real_dis_loss = gan_loss_fn(discriminator(z_sampled), z_sampled_label)
                fake_dis_loss = gan_loss_fn(discriminator(z), z_label)

                dis_loss = (real_dis_loss + fake_dis_loss) / 2
                dis_epoch_loss += dis_loss

                gan_optimizer.zero_grad()
                dis_loss.backward
                gan_optimizer.step()

                iteration_num = batch_idx
            recon_epoch_loss /= iteration_num
            dis_epoch_loss /= iteration_num

            logger.info(",".join([
                f"epoch={epoch:03d}",
                f"recon_loss={recon_epoch_loss:.4f}",
                f"dis_loss={dis_epoch_loss:.4f}"
            ]))

            writer.add_scalar("loss/training_loss", recon_epoch_loss, epoch)
            writer.add_scalar("loss/validation_loss(acc)", dis_epoch_loss, epoch)
