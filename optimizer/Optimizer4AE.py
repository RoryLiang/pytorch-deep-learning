import torch
import logging
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from model.ae import AutoEncoder
from utils.logging_ import init_logger


logger = logging.getLogger(__name__)


class AEoptimizer():
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

        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=True)

        model = AutoEncoder()
        device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")
        init_logger(rank=0, filenmae=self.args.output_dir/"default.log")
        logger.info(f"training on {device}")
        model.to(device)
        writer = SummaryWriter("/".join([self.args.output_dir, "tb"]))

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        for epoch in range(1, self.args.epoch_num+1):

            train_epoch_loss = 0
            train_iter = 0
            for batch in train_dataloader:
                model.train(mode=True)
                batch[0].to(device)
                out = model(batch[0])
                loss = loss_fn(out, batch[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.data
                train_iter += 1
            train_epoch_loss /= train_iter

            valid_epoch_loss = 0
            valid_iter = 0
            for batch in valid_dataloader:
                model.eval()
                batch[0].to(device)
                out = model(batch[0])
                loss = loss_fn(out, batch[0])

                valid_epoch_loss += loss.data
                valid_iter += 1
            train_epoch_loss /= train_iter

            logger.info(",".join([
                f"epoch={epoch:03d}",
                f"training_loss={train_epoch_loss:.4f}",
                f"validation_loss={valid_epoch_loss:.4f}"
            ]))

            writer.add_scalar("loss/training_loss", train_epoch_loss, epoch)
            writer.add_scalar("loss/validation_loss", valid_epoch_loss, epoch)
