import torch
import torch.nn as nn


class Dense(nn.Module):
    """
    Fully Connected feed forward network.

    `depth` is the total number of linear layers.
    """

    def __init__(self, in_dim, out_dim, width, depth):

        super().__init__()

        layers = [
            nn.Linear(in_features=in_dim, out_features=width, bias=True),
            nn.ReLU()
        ]
        for _ in range(1, depth - 1):
            layers.append(nn.Linear(in_features=width,
                                    out_features=width, bias=True))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=width,
                                out_features=out_dim, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)
