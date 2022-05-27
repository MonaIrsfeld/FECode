from matplotlib.pyplot import axis
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
from numpy import dtype, random, fft

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from scipy import signal

class PDDataset(Dataset):
    def __init__(self, file_names, train=True):
        super().__init__()
        temp_data = []
        labels = []
        names = []
        for name in file_names:
            temp_result = []
            file_data = pd.read_csv(name, sep='\t')
            label_data = pd.read_csv('./out_subset.csv')
            index = np.where(label_data['id']==name[13:22])[0][0]
            label = float(label_data['tmt_b_minus_a'][index])
            length = file_data.shape[0]
            if label==999 or length<512:
                continue
            
            #temp_result = np.repeat(file_data.values, 5, 0)[:5000,:]
            temp_result = torch.Tensor(file_data.values)
            temp_result = temp_result.unfold(0,512,128)
            for i in range(temp_result.shape[0]):
                names.append(name)
                labels.append(label)
                #print(temp_result[i,:,:].shape)
                temp_data.append(temp_result[i,:,:])
            #temp_result = np.transpose(temp_result)
            #temp_result = fft.fft(temp_result)[:,:1000]
            #temp_result = np.split(temp_result, [500,1000,1500,2000], axis=1)[:3]
            #labels.extend([label, label, label])
            #labels.append(label)
            #temp_result = preprocessing(temp_result)
            #temp_result=MinMaxScaler().fit_transform(temp_result)
            #temp_data.append(temp_result)
        temp_data =pad_sequence([torch.Tensor(ele).transpose(0,1) for ele in temp_data]).transpose(0,1).transpose(1,2)
        #temp_data = scaling(temp_data)
        print('temp data: ', temp_data.shape)
        self.data = temp_data
        self.file_names = names
        self.labels = torch.Tensor(labels)
        #self.labels = torch.Tensor(MinMaxScaler().fit_transform(np.array(labels).reshape(-1,1))).squeeze()
        #self.labels = torch.Tensor(robust_scale(labels, quantile_range=(0,90)))
        #print(self.labels)
        # self.data = torch.transpose(temp_data, 1,2)
        #print('data: ', self.data.shape)
    
    def get_labels(self):
        return self.labels
    
    def __getitem__(self, index):
        #start = random.randint(100)
        #start=0
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]
    
    def get_data_with_fn(self, index):
        return self.file_names[index], self.data[index], self.labels[index]



def preprocessing(data):
    for row in data:
        # moving average filtering (N=5)
        row = np.convolve(row, np.ones(3), 'valid')/3
        # butterworth filter with cutoff frequency = 12Hz
        sos = signal.butter(3, 12, 'hp', fs=100, output='sos')
        row = signal.sosfilt(sos, row)
        #row = (row-np.min(row))/(np.max(row)-np.min(row))
    return data

def scaling(data):
    for i in range(data.shape[1]):
        scaler = MinMaxScaler(feature_range=(0,1)).fit(data[:,i,:])
        data[:,i,:] = torch.from_numpy(scaler.transform(data[:,i,:]))
    return data