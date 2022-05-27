from turtle import forward
import torch
from torch import dropout, nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=1),
            #nn.BatchNorm1d(out_ch),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=1),
            #nn.BatchNorm1d(out_ch),
            nn.Tanh()
        )
        self.pool = nn.MaxPool1d(2, stride=2)
    
    def forward(self,x):
        return self.conv2(self.conv1(x))

class DecoderBlock(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, kernel_size):
        super().__init__()
        self.block = nn.Sequential(
                        nn.Conv1d(n_ch_in, n_ch_in, kernel_size, padding=1),
                        #    nn.BatchNorm1d(n_ch_in),
                        nn.Tanh(),
                                   
                           nn.ConvTranspose1d(n_ch_in, n_ch_out, 2, stride=2),
                           #nn.BatchNorm1d(n_ch_out),
                           nn.Tanh()
                          )
    
    def forward(self,x):
        return self.block(x)
    

class ConvAutoEnc(torch.nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std
        #print(self.std.shape)
        kernel_size = 3
        self.encoder = torch.nn.Sequential(
            nn.BatchNorm1d(27),
            nn.Conv1d(27,32,kernel_size, stride=2, padding=1),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(32,64,kernel_size, stride=2,padding=1),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128,kernel_size, stride=2,padding=1),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Conv1d(128,256, kernel_size, padding=1),
            # nn.ReLU(),
            #nn.Dropout(0.5),
            # nn.Conv1d(256,512, kernel_size),
            # nn.ReLU(),
            # nn.Conv1d(512,256, kernel_size),
            # nn.ReLU(),
            # nn.Conv1d(256,128, kernel_size, padding=1),
            # nn.ReLU(),
            nn.Conv1d(128,256,kernel_size, stride=2,padding=1),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Conv1d(256,512,kernel_size, stride=2,padding=1),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(512,1024,kernel_size, stride=2,padding=1),
            #nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(1024,512,kernel_size, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv1d(512,512,kernel_size,stride=2, padding=1),
            nn.ReLU(),
        )
        self.linear1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1024, 512),

        )

        self.linear2 = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(512, 2))
        )

        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose1d(512,512,kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(512,1024,kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(1024,512,kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(512,256,kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256,128,kernel_size, stride=2, padding=1),
            nn.ReLU(),
            # nn.ConvTranspose1d(128,256, kernel_size, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose1d(256,512, kernel_size),
            # nn.ReLU(),
            # nn.ConvTranspose1d(512,256, kernel_size),
            # nn.ReLU(),
            # nn.ConvTranspose1d(256,128, kernel_size, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose1d(128,128,kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128,64,kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64,32, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32,27, kernel_size, stride=2, padding=1),
            #nn.BatchNorm1d(27),
            #nn.ReLU(),
        )

    def forward(self, x):
        # print(x.shape)
        encoded = self.linear1(self.encoder(x))#+0.1*self.std*torch.randn(x.shape))
        #print(encoded.shape)
        decoded = self.decoder(self.linear2(encoded))
        #print(decoded.shape)
        return decoded