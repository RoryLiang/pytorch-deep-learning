import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from model.ae import AutoEncoder
from torch import nn
import matplotlib.pyplot as plt


class AEoptimizer():
    def __init__(self, args) -> None:
        self.args = args

    def optimize():
        train_dataset = datasets.MNIST(
            root="../data",
            train=True,
            download=True
        )
        valid_dataset = datasets.MNIST(
            root="../data",
            train=False,
            download=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=True
        )
        train_features, train_labels = next(iter(train_dataloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0].squeeze()
        label = train_labels[0]
        plt.imshow(img, cmap="gray")
        plt.show()
        print(f"Label: {label}")
'''
        model = AutoEncoder()
        epoch_num = 100
        learning_rate = 1e-3
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, epoch_num+1):
            for batch in train_dataloader:
                out = model(batch)
'''