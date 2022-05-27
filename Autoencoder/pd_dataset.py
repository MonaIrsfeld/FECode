import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from scipy import signal
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

class PDDataset(Dataset):
    def __init__(self, file_names):
        super().__init__()
        temp_data = []
        names = []
        for name in file_names:
            
            file_data = pd.read_csv(name, sep='\t')
            column_names = file_data.columns
            length = file_data.shape[0]
            if length<512:
                continue
            name = name[13:22]
            temp_result = torch.Tensor(file_data.values)
            temp_result = temp_result.unfold(0,512,128)
            for i in range(temp_result.shape[0]):
                names.append(name)
                #print(temp_result[i,:,:].shape)
                temp_data.append(temp_result[i,:,:])
        temp_data =pad_sequence([torch.Tensor(ele).transpose(0,1) for ele in temp_data]).transpose(0,1).transpose(1,2)
        #print('temp data: ', temp_data.shape)
        self.data = scaling(temp_data)
        self.file_names = names
        print('data: ', self.data.shape)
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]
    
    def get_size(self, dim=None):
        return self.data.size(dim)
    
    def get_cts_name(self, index):
        return self.file_names[index]


def filtering(data):
    for row in data:
        # moving average filtering (N=5)
        row = np.convolve(row, np.ones(5), 'valid')/5
        # butterworth filter with cutoff frequency = 15Hz
        sos = signal.butter(3, 12, 'hp', fs=100, output='sos')
        row = signal.sosfilt(sos, row)
        #row = (row-np.min(row))/(np.max(row)-np.min(row))
    return data


def scaling(data):
    for i in range(27):
        # min = torch.min(data[:,i,])
        # max = torch.max(data[:,i,])
        # data[:,i,:] = (data[:,i,]-min)/(max-min)
        #data[:,i,:] = torch.from_numpy(RobustScaler(quantile_range=(2,98)).fit_transform(data[:,i,:]))
        data[:,i,:] = torch.from_numpy(MinMaxScaler(feature_range=(0,1)).fit_transform(data[:,i,:]))
    return data