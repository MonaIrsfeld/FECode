# import tensorflow as tf
# from keras import Sequential
# from keras.layers import Flatten, Dense, Dropout, BatchNormalization
# from keras.optimizers import adam_experimental
# from scipy import fft, signal
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_example_data(idx_number, task_id):
    idx_number = str(idx_number)
    if task_id==0:
        data = pd.read_csv("./data/sub-COKI"+idx_number+"/sub-COKI"+idx_number+"_ses-T1_task-straightwalknormalpace_tracksys-imu.tsv", sep="\t")
        return data.columns, data
    elif task_id==1:
        data = pd.read_csv("./data/sub-COKI"+idx_number+"/sub-COKI"+idx_number+"_ses-T1_task-straightwalkfastpace_tracksys-imu.tsv", sep="\t")
        return data.columns, data
    elif task_id==2:
        data = pd.read_csv("./data/sub-COKI"+idx_number+"/sub-COKI"+idx_number+"_ses-T1_task-dualtaskstraightwalkcheckingboxes_tracksys-imu.tsv", sep="\t")
        return data.columns, data
    elif task_id==3:
        data = pd.read_csv("./data/sub-COKI"+idx_number+"/sub-COKI"+idx_number+"_ses-T1_task-dualtaskstraightwalkserialsubtraction_tracksys-imu.tsv", sep="\t")
        return data.columns, data


def sizes_all_data():
    file_names = glob.glob("./data/sub-COKI"+"*"+"/sub-COKI"+"*"+"_ses-T1_task-"+"*"+"_tracksys-imu.tsv")[:10]
    column_length = 27
    for name in file_names:
        print(name)
        data = pd.read_csv(name, sep="\t")
        acc_x_keys = [x for x in data.columns if x[5:]=="acc_x"]
        plt.figure(name)
        for key in acc_x_keys:
            plt.plot(data[key])
        plt.show()


def data_with_labels(task_name):
    file_names = glob.glob("./00_raw/sub-COKI"+"*"+"/sub-COKI"+"*"+"_ses-T1_task-"+task_name+"_tracksys-imu.tsv")
    cts_indices = [list(map(int, re.findall('\d+', file_names[i])))[2] for i in range(len(file_names))]
    out_subset_data = pd.read_csv("./out_subset.csv")
    with_labels = np.unique([out_subset_data["id"][i] for i in range(len(out_subset_data["id"])) if out_subset_data["tmt_b_minus_a"][i]!=999
            and int(out_subset_data["id"][i][4:]) in cts_indices])
    return with_labels

if __name__=='__main__':
    # sizes_all_data()
    labelled_1 = data_with_labels("straightwalknormalpace")
    print("straightwalknormalpace: ", len(labelled_1))

    labelled_2 = data_with_labels("straightwalkfastpace")
    print("straightwalkfastpace: ", len(labelled_2))

    labelled_3 = data_with_labels("dualtaskstraightwalkcheckingboxes")
    print("dualtaskstraightwalkcheckingboxes: ", len(labelled_3))

    labelled_4 = data_with_labels("dualtaskstraightwalkserialsubtraction")
    print("dualtaskstraightwalkserialsubtraction: ", len(labelled_4))

    print("Task 1 and 2: ", len([item for item in labelled_1 if item in labelled_2]))
    print("Task 1 and 3: ", len([item for item in labelled_1 if item in labelled_3]))
    print("Task 1 and 4: ", len([item for item in labelled_1 if item in labelled_4]))
    print("Task 2 and 3: ", len([item for item in labelled_2 if item in labelled_3]))
    print("Task 2 and 4: ", len([item for item in labelled_2 if item in labelled_4]))
    print("Task 3 and 4: ", len([item for item in labelled_3 if item in labelled_4]))
    
    # fig, axs = plt.subplots(2,2)
    # for i in range(4):
    #     data_columns, data = load_example_data(10023, i)
    #     signal = data[data_columns[0]].to_numpy()
    #     yf = fft.rfft(signal)
    #     xf = fft.rfftfreq(len(signal), 1/100)
    #     idx_x = i//2
    #     idx_y = i%2
    #     axs[idx_x, idx_y].plot(xf[:2000],np.abs(yf[:2000]))

    # fig, axs = plt.subplots(2,2)
    # for i in range(4):
    #     data_columns, data = load_example_data(10040, i)
    #     signal = data[data_columns[0]].to_numpy()
    #     yf = fft.rfft(signal)
    #     xf = fft.rfftfreq(len(signal), 1/100)
    #     idx_x = i//2
    #     idx_y = i%2
    #     axs[idx_x, idx_y].plot(xf[:2000],np.abs(yf[:2000]))

    # plt.show()





