
from cProfile import label
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import torch
from torch.utils.data import DataLoader
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from convae import ConvAutoEnc
from convlinae import ConvLinAutoEnc
from linae import LinAutoEnc
from lstm import LSTMAutoEnc
from pd_dataset import PDDataset
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 1500
column_names = []


def get_file_names():
    file_names = glob.glob("./00_raw/sub-COKI*/sub-COKI*_ses-T1_*_tracksys-imu.tsv")
    return [name for name in file_names if 'normal' in name]


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

def scale_train(train_data):
    scalers = []
    for i in range(train_data.shape[1]):
        scaler = MinMaxScaler()
        scaler = scaler.fit(train_data[:,i,:])
        scalers.append(scaler)
        train_data[:,i,:] = torch.Tensor(scaler.transform(train_data[:,i,:]))
    return train_data, scalers

def scale_test(test_data, scalers):
    for i in range(len(scalers)):
        test_data[:,i,:] = torch.Tensor(scalers[i].transform(test_data[:,i,:]))
    return test_data

if __name__=='__main__':
    file_names = get_file_names()
    train_files, test_files = file_names[:-5], file_names[-5:]
    print('Test:',test_files)
    train_data = PDDataset(train_files)
    test_data = PDDataset(test_files)
    data = PDDataset(file_names)
    _ , column_names = get_data(file_names) 

    # train_data, test_data = train_test_split(data, test_size=0.05)
    
    # train_data = torch.stack(train_data)
    # test_data = torch.stack(test_data)
    train_data.data, scalers = scale_train(train_data.data)
    test_data.data = scale_test(test_data.data, scalers)

    
    std = torch.std(train_data.data, dim=(0,2))
    std = std.repeat((train_data.get_size(1),1)).transpose(0,1)
    
    batch_size = 1000
    epochs=250

    dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    loss_per_epoch = torch.zeros(epochs)

    test_dl = DataLoader(test_data, batch_size=batch_size)
    val_loss = torch.zeros(epochs)

    device='cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # Model Initialization
    model = ConvAutoEnc(std)
    model = model.to(device)
  
    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()
  
    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3,
                             weight_decay = 0)
    
    outputs = []
    for epoch in range(epochs):
        #model.train()
        for batch in dl:
            optimizer.zero_grad()
            # Output of Autoencoder
            # batch = batch.unsqueeze(1)
            reconstructed = model(batch.to(device))
            
        
            # Calculating the loss function
            if reconstructed.size()>batch.size():
                loss = loss_function(reconstructed[:,:,:batch.shape[2]], batch)
                
            else:
                loss = loss_function(reconstructed, batch[:,:,:reconstructed.shape[2]])
            
      
        # the the gradient is computed and stored.
        # .step() performs parameter update
            loss.backward()
            optimizer.step()
            # print(loss.item()/len(dl))
            loss_per_epoch[epoch] += loss.item() / len(dl)
        #model.eval()
        with torch.no_grad():
            for batch in test_dl:
                # batch=batch.unsqueeze(1)
                reconstructed = model(batch)
                if reconstructed.size()>batch.size():
                    reconstructed = reconstructed[:,:,:batch.shape[2]]
                else:
                    reconstructed = torch.nn.functional.pad(reconstructed, pad=(0,batch.shape[2]-reconstructed.shape[2]))
                val_loss[epoch]+=(loss_function(reconstructed,batch).item())/len(test_dl)
            
        
        print('Epoch ',epoch+1, '/',epochs,': ',loss_per_epoch[epoch].item(),'(',val_loss[epoch].item(),')')
        
        # Storing the losses in a list for plotting
        outputs.append((epochs, batch, reconstructed))
  
# Defining the Plot Style
plt.xlabel('Iterations')
plt.ylabel('Loss')
# Train loss
plt.semilogy([loss.detach().numpy() for loss in loss_per_epoch], label='Training loss')
# validation loss
plt.semilogy(val_loss, label='Validation loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'autoencoder.pkl')

seed = np.random.randint(len(test_data))
pred = model(test_data[seed].unsqueeze(0))
print(model.encoder(test_data[seed].unsqueeze(0)).size())
print(seed)
for i in range(27):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, num=column_names[i], sharex=True, sharey=True)
    org = test_data[seed][i]
    output = pred[0,i].detach().numpy()
    if len(org)<len(output):
        org = np.pad(org, (0,len(output)-len(org)))
    else:
        output = np.pad(output, (0,len(org)-len(output)))
    ax1.plot(org, color='green')
    ax1.set_title('Original (filtered) signal')
    ax2.plot(output)
    ax2.set_title('Network output')
    ax3.plot(np.abs(org-output))
plt.show()

f= open('features_auto.csv', 'w')
writer = csv.writer(f)
for i in range(len(data)):
    file_name, sensor_data = data.get_cts_name(i), data[i]
    row = np.concatenate([[file_name], model.linear1(model.encoder(sensor_data.unsqueeze(0))).flatten().detach().numpy()])
    writer.writerow(row)
f.close()



