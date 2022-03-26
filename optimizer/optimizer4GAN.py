import torch
import logging
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from model.gan import Generator, Discriminator
from utils.logging_ import init_logger


logger = logging.getLogger(__name__)


class GANoptimizer():
    def __init__(self, args) -> None:
        self.args = args

    def optimize(self):
        train_dataset = datasets.MNIST(
            root=self.args.data_dir,
            train=True,
            transform=ToTensor(),
            download=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )

        generator = Generator()
        discriminator = Discriminator()

        device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
        init_logger(rank=0, filenmae=self.args.output_dir+"/default.log")
        logger.info(f"training on {device}")
        generator.to(device)
        discriminator.to(device)
        writer = SummaryWriter("/".join([self.args.output_dir, "tb"]))

        loss_fn = nn.BCELoss()
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=self.args.learning_rate)
        dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.args.learning_rate)

        for epoch in range(1, self.args.epoch_num+1):

            gen_epoch_loss = 0
            dis_epoch_loss = 0
            iteration_num = 0
            for batch_idx, (real_img, label) in enumerate(train_dataloader):

                # train generator
                rand_noise_z = torch.randn(self.args.batch_size, 100, device=device)

                fake_img = generator(rand_noise_z)
                fake_label = torch.zeros(self.args.batch_size, device=device)
                gen_loss = loss_fn(discriminator(fake_img), fake_label)
                gen_epoch_loss += gen_loss

                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()

                # train discriminator
                real_img.to(device)
                real_label = torch.ones(self.args.batch_size, device=device)

                real_dis_loss = loss_fn(discriminator(real_img), real_label)
                fake_dis_loss = loss_fn(discriminator(fake_img), fake_label)

                dis_loss = (real_dis_loss + fake_dis_loss) / 2
                dis_epoch_loss += dis_loss

                dis_optimizer.zero_grad()
                dis_loss.backward
                dis_optimizer.step()

                iteration_num = batch_idx

            gen_epoch_loss /= iteration_num
            dis_epoch_loss /= iteration_num

            logger.info(",".join([
                f"epoch={epoch:03d}",
                f"gen_loss={gen_epoch_loss:.4f}",
                f"dis_loss={dis_epoch_loss:.4f}"
            ]))

            writer.add_scalar("loss/gen_loss", gen_epoch_loss, epoch)
            writer.add_scalar("loss/dis_loss", dis_epoch_loss, epoch)
