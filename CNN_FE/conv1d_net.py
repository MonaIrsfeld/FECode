from turtle import forward
import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_ch_in, n_ch_out,kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(n_ch_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_ch_out, n_ch_out, kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(n_ch_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.MaxPool1d(2, stride=2)
    
    def forward(self, x):
        y = self.conv1(x)
        stacked = torch.cat([y,self.conv2(y)], dim=1)
        #print(stacked.shape)
        return self.pool(stacked)
    

class ConvBlock(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_ch_in, n_ch_out, kernel_size, padding=1),
            nn.BatchNorm1d(n_ch_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_ch_out, n_ch_out, kernel_size, padding=1),
            nn.BatchNorm1d(n_ch_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.MaxPool1d(2, stride=2)
    
    def forward(self, x):
        y = self.conv1(x)
        #print(stacked.shape)
        return self.pool(self.conv2(y))




class ConvRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.ReLU()
        kernel_size=3
        self.conv = nn.Sequential(
            #nn.InstanceNorm1d(27),
            nn.BatchNorm1d(27),
            ResBlock(27, 32, kernel_size, 0.8),
            ResBlock(64, 64, kernel_size, 0.8),
            ResBlock(128, 128, kernel_size, 0.8),
            ResBlock(256, 256, kernel_size, 0.8),
            ResBlock(512, 512, kernel_size, 0.8),
            #ResBlock(1024, 1024, kernel_size, 0.8),
            #ResBlock(2048, 1024, kernel_size, 0.8),
            #ResBlock(2048, 512, kernel_size, 0.8),
            ResBlock(1024, 256, kernel_size, 0.8),
            ResBlock(512, 128, kernel_size, 0.8),
            nn.AdaptiveAvgPool1d(1)
        )

        # self.conv = nn.Sequential(
        #     nn.BatchNorm1d(27),
        #     ConvBlock(27,32,kernel_size,0.8),
        #     ConvBlock(32,64,kernel_size,0.8),
        #     ConvBlock(64,128,kernel_size,0.8),
        #     ConvBlock(128,256,kernel_size,0.8),
        #     ConvBlock(256,512,kernel_size,0.8),
        #     ConvBlock(512,1024,kernel_size,0.8),
        #     ConvBlock(1024,1024,kernel_size,0.8),
        #     ConvBlock(1024,512,kernel_size,0.8),
        #     ConvBlock(512,512,kernel_size,0.8),
        # )

        self.lin = nn.Sequential(
            nn.Flatten(),
            #nn.Dropout(0.5),
            #nn.LazyLinear(512),
            #nn.Tanhshrink(),
            nn.Dropout(0.8),
            nn.Linear(256,1)
        )
    
    def forward(self, x):
        #print(x.shape)
        x_new = self.conv(x)
        #print(x_new.shape)
        return self.lin(x_new)
