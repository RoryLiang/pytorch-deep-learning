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

        model = AutoEncoder()
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
                model_out = model(img)
                loss = loss_fn(model_out, img)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_train_epoch_loss += loss.item()
                train_iter_num = batch_idx
            avg_train_epoch_loss /= train_iter_num

            avg_valid_epoch_loss = 0
            valid_iter_num = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (img, labels) in valid_dataloader:

                    img = img.to(device)
                    out = model(img)
                    loss = loss_fn(out, img)

                    avg_valid_epoch_loss += loss.item()
                    valid_iter_num = batch_idx
                avg_train_epoch_loss /= valid_iter_num

            logger.info(",".join([
                f"epoch={epoch:03d}",
                f"training_loss={avg_train_epoch_loss:.4f}",
                f"validation_loss={avg_valid_epoch_loss:.4f}"
            ]))

            writer.add_scalar("loss/training_loss", avg_train_epoch_loss, epoch)
            writer.add_scalar("loss/validation_loss(acc)", avg_valid_epoch_loss, epoch)
