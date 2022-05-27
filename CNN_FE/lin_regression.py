from turtle import forward
import torch
from torch import nn

class LinPredictor(torch.nn.Module):

    def __init__(self, max_length):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = torch.nn.Sequential(
            # nn.Flatten(),
            # nn.LazyLinear(8192),
            # nn.ReLU(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(1024)
            # nn.ReLU(),
            # nn.LazyLinear(256)
        )

        self.regression = nn.Linear(1024,1)

    def forward(self, x):
        flattened = self.flatten(x)
        encoded = self.encoder(flattened)
        output = self.regression(encoded)
        #print(output.shape)
        return output

