import torch
from torch import nn


class ConvLinAutoEnc(torch.nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            nn.LazyConv1d(32,3, stride=2),
            nn.ReLU(),
            nn.LazyConv1d(64,3, stride=2),
            nn.ReLU(),
            nn.LazyConv1d(64,3, stride=2)
        )

        self.fully_connected_1 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(64*61),
            nn.ReLU(),
            nn.LazyLinear(1024)
        )

        self.fully_connected_2 = nn.Sequential(
            nn.ReLU(),
            nn.LazyLinear(64*61)
        )

        self.decoder = torch.nn.Sequential(
            
            nn.LazyConvTranspose1d(64,3, stride=2),
            nn.ReLU(),
            nn.LazyConvTranspose1d(64,3, stride=2),
            nn.ReLU(),
            nn.LazyConvTranspose1d(32,3, stride=2),
            nn.ReLU(),
            nn.LazyConvTranspose1d(27,3),
            nn.ReLU(),
            nn.LazyLinear(max_length)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        tgt_shape = encoded.shape
        # print(encoded.shape)
        features = self.fully_connected_1(encoded)
        # print(features.shape)
        f_2 = self.fully_connected_2(features)
        decoded = self.decoder(f_2.reshape(tgt_shape))
        # print(decoded.shape)
        return decoded