import torch
from torch import nn

class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(27),
            nn.Conv1d(27, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(256, 512, kernel_size=3, stride=1),
            nn.ReLU()
        )

    
    def forward(self, x):
        #print(x.shape)
        x = self.conv(x)
        f = nn.Flatten()
        flattened = f(x)
        # print(flattened.shape)
        dense = nn.Sequential(
            nn.Linear(flattened.shape[1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        return dense(flattened)