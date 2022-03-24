import torch
import logging
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from model.vae import VAE
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

        model = VAE()
        device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
        init_logger(rank=0, filenmae=self.args.output_dir+"/default.log")
        logger.info(f"training on {device}")
        model.to(device)
        writer = SummaryWriter("/".join([self.args.output_dir, "tb"]))

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        for epoch in range(1, self.args.epoch_num+1):

            avg_train_epoch_loss = 0
            train_iter_num = 0
            model.train(mode=True)

            for batch_idx, (img, labels) in enumerate(train_dataloader):

                img = img.to(device)
                z_mean, z_logvar, decoder_out = model(img)

                # computing loss
                kl_div = -0.5 * torch.sum(1 + z_logvar - z_mean**2 - torch.exp(z_logvar), axis=1)  # sum over latent dimension

                batchsize = kl_div.size(0)
                kl_div = kl_div.mean()  # average over batch dimension

                pixelwise = loss_fn(decoder_out, img, reduction='none')
                pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
                pixelwise = pixelwise.mean()  # average over batch dimension

                reconstruction_term_weight = 1
                loss = reconstruction_term_weight*pixelwise + kl_div

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_train_epoch_loss += loss.data
                train_iter_num = batch_idx
            avg_train_epoch_loss /= train_iter_num

            avg_valid_epoch_loss = 0
            valid_iter_num = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (img, labels) in enumerate(valid_dataloader):

                    img = img.to(device)
                    z_mean, z_logvar, decoder_out = model(img)

                    loss = loss_fn(decoder_out, img)

                    avg_valid_epoch_loss += loss.data
                    valid_iter_num = batch_idx
                avg_train_epoch_loss /= valid_iter_num

            logger.info(",".join([
                f"epoch={epoch:03d}",
                f"training_loss={avg_train_epoch_loss:.4f}",
                f"validation_loss={avg_valid_epoch_loss:.4f}"
            ]))

            writer.add_scalar("loss/training_loss", avg_train_epoch_loss, epoch)
            writer.add_scalar("loss/validation_loss(acc)", avg_valid_epoch_loss, epoch)
