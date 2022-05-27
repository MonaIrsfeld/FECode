from turtle import forward
import torch
from torch import nn

class LSTMAutoEnc(nn.Module):

    def __init__(self, max_length):
        super().__init__()

        self.encoder = nn.LSTM(input_size = max_length, hidden_size = 150, batch_first=True)
        self.decoder = nn.LSTM(input_size = 150, hidden_size = max_length, batch_first=True)
    
    def forward(self, x):
        encoded = self.encoder(x)[0]
        return self.decoder(encoded)[0]
