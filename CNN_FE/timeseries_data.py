from pytorch_forecasting.data import TimeSeriesDataSet
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def make_dataset(file_names):
    column_names = ['group_id', 'time_idx',
                        'acc_x_lf','acc_y_lf','acc_z_lf', 'gyro_x_lf','gyro_y_lf','gyro_z_lf', 'mag_x_lf','mag_y_lf','mag_z_lf',
                        'acc_x_rf','acc_y_rf','acc_z_rf', 'gyro_x_rf','gyro_y_rf','gyro_z_rf', 'mag_x_rf','mag_y_rf','mag_z_rf',
                        'acc_x_hip','acc_y_hip','acc_z_hip', 'gyro_x_hip','gyro_y_hip','gyro_z_hip', 'mag_x_hip','mag_y_hip','mag_z_hip',
                        'label']
    df = pd.DataFrame(columns=column_names)
    labels = []
    for name in file_names:
        temp_result = []
        file_data = pd.read_csv(name, sep='\t')
        label_data = pd.read_csv('./out_subset.csv')
        index = np.where(label_data['id']==name[13:22])[0][0]
        file_data.insert(0, 'group_id', name[17:22])
        file_data.insert(1, 'time_idx', list(np.arange(file_data.shape[0], dtype=int)))
        label = float(label_data['tmt_b_minus_a'][index])
        if label==999 or 'normal' not in name or file_data.values.shape[0]<100 or file_data.values.shape[0]>15000:
            continue
        labels.append(label)
        file_data = file_data.assign(label_column = label)
        file_data.columns = column_names
        df = pd.concat([df, file_data], axis=0, ignore_index=True)
    df = df.astype({'time_idx':int})
    print(df)
    return TimeSeriesDataSet(df, time_idx = 'time_idx', group_ids=['group_id'], target='label')