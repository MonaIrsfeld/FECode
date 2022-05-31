import setuptools
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import glob
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import csv
from lin_regression import LinPredictor
from conv1d_net import ConvRegressor
from pd_dataset import PDDataset
#import seaborn as sns
#import pytorch_forecasting as pfc
#from pytorch_forecasting.metrics import RMSE
#from timeseries_data import make_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 512




def get_file_names():
    file_names = glob.glob("./00_raw/sub-COKI*/sub-COKI*_ses-T1_*_tracksys-imu.tsv")
    return [name for name in file_names if 'subtract' in name]


def get_data(file_names):
    result=[]
    for name in file_names:
        data = pd.read_csv(name, sep="\t")
        temp_result = []
        for col in data.columns:
            center = len(data[col])//2
            temp_result.append(data[col][:1000])
        result.append(np.array(temp_result))
    return result, data.columns

if __name__=='__main__':
    file_names = get_file_names()
    train_files, test_files = file_names[:38]+file_names[39:], [file_names[38]]
    print('Test:',test_files)
    train_data = PDDataset(train_files)
    test_data = PDDataset(test_files)
    
    
    _ , column_names = get_data(file_names) 

    #train_data, test_data = train_test_split(data, test_size=0.1)

    
    batch_size = len(train_data)+1
    epochs=400

    last_loss = 60000
    patience = 2

    dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    loss_per_epoch = torch.zeros(epochs)

    test_dl = DataLoader(test_data, batch_size=batch_size)
    val_loss = torch.zeros(epochs)

    # Model Initialization
    model = ConvRegressor()
    model = model.to(device)

    best_model = ConvRegressor()
    best_loss = np.infty
  
    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()
  
    # Using an Adam Optimizer 
    optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-2,
                             weight_decay = 0.1)
    
    outputs = []
    for epoch in range(epochs):
        model.train()
        for batch, label in dl:
            optimizer.zero_grad()
            predicted = model(batch.to(device))
            
        
            # Calculating the loss function
            loss = loss_function(predicted, label.unsqueeze(1).cuda())
      
        # the the gradient is computed and stored.
        # .step() performs parameter update
            loss.backward()
            optimizer.step()
            loss_per_epoch[epoch] += loss.item() / len(dl)
        # validation loss
        model.eval()
        with torch.no_grad():
            for batch,label in test_dl:
                val_loss[epoch]+=(loss_function(model(batch.cuda()).squeeze(),label.cuda()).item())/len(test_dl)
        
        
        # code snippet used for early stopping
        # if loss_per_epoch[epoch] > last_loss:
        #     trigger_times += 1
        #     print('Trigger Times:', trigger_times)

        #     if trigger_times >= patience:
        #         print('Early stopping!\nStart to test process.')
        #         break

        # else:
        #     print('trigger times: 0')
        #     trigger_times = 0

        # last_loss = loss_per_epoch[epoch]
            
        
        print('Epoch ',epoch+1, '/',epochs,': ',loss_per_epoch[epoch].item(),'('+str(val_loss[epoch].item())+')')

with torch.no_grad():
    for batch, label in test_dl:
        print('Predicted:\n', model(batch.cuda()).cpu().squeeze().detach().numpy().reshape(-1,1))
        print('Actual:\n', label.numpy().reshape(-1,1))
  
# Defining the Plot Style
# plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# indices = random.randint(len(data), size=10)
# for index in indices:
#     sensor_data, label = data[index]
#     sensor_data = sensor_data.unsqueeze(0)
#     print('Actual:', label.item(), 'Predicted:', model(sensor_data).item())

torch.save(model.state_dict(), 'regression_cnn.pkl')

# plt.semilogy([loss.detach().numpy() for loss in loss_per_epoch[:epoch]], label='Training loss')
# plt.semilogy(val_loss[:epoch], label='Validation loss')
# plt.legend()
# plt.show()

# write extracted features to csv file
f= open('features.csv', 'w')
writer = csv.writer(f)
data = PDDataset(file_names)
for i in range(len(data)):
    file_name, sensor_data, label = data.get_data_with_fn(i)
    row = np.concatenate([[file_name], model.conv(sensor_data.unsqueeze(0).cuda()).cpu().flatten().detach().numpy(), [label]])
    writer.writerow(row)
f.close()



