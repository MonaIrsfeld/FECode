import torch
from torch import nn

class LinAutoEnc(torch.nn.Module):

    def __init__(self, max_length):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            nn.Flatten(),
            # nn.LazyLinear(8192),
            # nn.ReLU(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(128)
        )

        self.decoder = torch.nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            # nn.LazyLinear(8192),
            # nn.ReLU(),
            # nn.LazyLinear(250),
            # nn.ReLU(),
            # nn.LazyLinear(2048),
            # nn.ReLU(),
            nn.LazyLinear(max_length*27)
            # nn.ReLU()
        )

    def forward(self, x):
        org_shape = x.shape
        # print(torch.min(x, dim=2).values.shape)
        # print(torch.std(x, dim=2).shape)
        std = torch.std(x, dim=2)
        # print(std[0])
        std = std.unsqueeze(2).expand(org_shape)
        # print(std[0,0])
        # random = std.repeat((1,1,256))
        # print(random.shape)
        x = x+0.25*torch.randn(org_shape)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.reshape(org_shape)
