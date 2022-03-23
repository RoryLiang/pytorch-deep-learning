import torch
import logging
from icecream import ic
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from model.ae import AutoEncoder


logger = logging.getLogger(__name__)


class AEoptimizer():
    def __init__(self, args) -> None:
        self.args = args

    def optimize(self):
        train_dataset = datasets.MNIST(
            root="../data",
            train=True,
            transform=ToTensor(),
            download=True
        )
        valid_dataset = datasets.MNIST(
            root="../data",
            train=False,
            transform=ToTensor(),
            download=True
        )

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

        model = AutoEncoder()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"train on {device}")
        model.to(device)
        output_path = "../output"
        writer = SummaryWriter("/".join([output_path, "tb"]))

        epoch_num = 100
        learning_rate = 1e-3
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, epoch_num+1):

            train_epoch_loss = 0
            valid_epoch_loss = 0

            for batch in train_dataloader:
                out = model(batch[0])
                loss = loss_fn(out, batch[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.data

            for batch in valid_dataloader:
                out = model(batch[0])
                loss = loss_fn(out, batch[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                valid_epoch_loss += loss.data

            logger.info(",".join([
                f"epoch={epoch}",
                f"valid_loss={train_epoch_loss}",
                f"valid_loss={valid_epoch_loss}"
            ]))

            writer.add_scalar("training_loss", train_epoch_loss, epoch)
            writer.add_scalar("validation_loss", valid_epoch_loss, epoch)
