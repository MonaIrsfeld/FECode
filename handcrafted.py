from math import perm
import pandas as pd
import numpy as np
from numpy.random import permutation
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import RFE
from time import time
from sklearn.neighbors import KNeighborsRegressor
import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm

def shuffleInUnison(X,y):
    assert len(X) == len(y)
    permutation = list(np.random.permutation(len(y)).astype(int))
    X=np.array(X)
    y=np.array(y)
    return X[permutation], y[permutation]

def read_data():
    all_data = pd.read_csv('./out_subset.csv', na_values='999')
    # needed_columns = ['id','tmt_b_minus_a',"gait_speed","DoubleLimbSupportVariability_corrected","StepTime_corrected",
    # "StrideTime_corrected","StanceTime_corrected","SwingTime_corrected","DoubleLimbSupport_corrected",
    # "Asymmetry_corrected","StepTimeVariability_corrected"]
    needed_columns = ['id','tmt_b_minus_a',"DoubleLimbSupport","NumberOfSteps","StepTime","StrideTime",
    "StanceTime","SwingTime","DoubleLimbSupportVariability","Asymmetry",
    "StepTimeVariability","MeasurementCalculated","SingleTaskSubtraction","gait_speed","DoubleLimbSupportVariability_corrected","StepTime_corrected",
    "StrideTime_corrected","StanceTime_corrected","SwingTime_corrected","DoubleLimbSupport_corrected",
    "Asymmetry_corrected","StepTimeVariability_corrected"]
    #print(data[needed_columns])
    # print(len(np.unique([all_data['id'][i] for i in range(len(all_data['id'])) if all_data['tmt_b_minus_a'][i]!=999])))
    # print(len(np.unique([all_data['id'][i] for i in range(len(all_data['id']))])))
    not_task = [i for i in range(len(all_data['id'])) if all_data['task_numeric'][i]!=4]
    data = all_data.drop(not_task)
    not_needed_clms = [x for x in data.columns if x not in needed_columns]
    
    data = data.drop(not_needed_clms, axis=1)
    #print(data)
    
    #data = data.dropna()
    print(data)
    #data = data.drop('id')
    without_labels = data[data['tmt_b_minus_a'].isna()].index
    data = data.drop(without_labels)
    d_tmt = [float(x) for x in data["tmt_b_minus_a"][data.index]]
    #print(d_tmt)
    data = data.drop('tmt_b_minus_a', axis=1)
    
    cts_ids = [id for id in data['id']]
    data = data.drop('id', axis=1)
    columns = data.columns
    data = data.fillna(0).values
    # result = (-1)*np.ones((data.shape[0], data.shape[1]*4))
    
    # for i in range(data.shape[0]):
    #     id = cts_ids[i]
    #     #print(id)
    #     result[i, :data.shape[1]] = data[i]
    #     rows = all_data.iloc('id'==id)
    #     #print(rows)
    #     for row in rows:
    #         task_id = row['task_numeric']
    #         if task_id in [1]:
    #             continue
    #         #print(task_id)
    #         #print(row[columns])
    #         result[i, data.shape[1]*(task_id-1):data.shape[1]*(task_id)] = row[columns]
    return data, d_tmt
    #return result, d_tmt

def get_extracted_features_cnn():
    return pd.read_csv('./features.csv', header=None).values

def get_extracted_features_autoenc():
    extr_features = pd.read_csv('./features_auto.csv', header=None).values
    print(extr_features.shape)
    handcrafted_df = pd.read_csv('./out_subset.csv', na_values='999')
    handcrafted_df = handcrafted_df.fillna(value=0)
    not_task = [i for i in range(len(handcrafted_df['id'])) if handcrafted_df['task_numeric'][i]!=2 or handcrafted_df['tmt_b_minus_a'][i]==0]
    handcrafted_df = handcrafted_df.drop(not_task)
    handcrafted_df = handcrafted_df.drop(handcrafted_df.columns.difference(['id', 'tmt_b_minus_a']), axis=1)
    handcrafted_df = handcrafted_df.dropna().values
    print(handcrafted_df.shape)
    with_label = [i for i in range(len(extr_features)) if extr_features[i,0] in handcrafted_df[:,0]]
    X = extr_features[with_label]
    y = np.zeros(X.shape[0])
    for i in range(handcrafted_df.shape[0]):
        row = handcrafted_df[i]
        if row[0] not in extr_features[:,0]:
            continue
        index = np.where(row[0]==X[:,0])
        y[index] = row[1]
    print(X.shape)
    print(len(y))
    return X, y



def preprocessing(data):
    pca = PCA(n_components=100)
    lda = LinearDiscriminantAnalysis()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    #data = pca.fit_transform(data)
    return data

def evaluation(predictions, labels):
    return (mean_squared_error(labels,predictions)+mean_absolute_error(labels,predictions))/2

def loo_folds(X,y):
    X, y = shuffleInUnison(X,y)
    ids = np.unique(X[:,0])
    np.random.shuffle(ids)
    k = len(np.unique(X[:,0]))
    print(X[np.where(X[:,0]==ids[1]),0])
    output = dict()
    for i in range(k):
        output[i] = (X[np.where(X[:,0]==ids[i]),1:][0],y[np.where(X[:,0]==ids[i])])
    unique_labels = []
    for i in range(k):
        label = output[i][1][0]
        # print(label)
        # print(np.mean(output[i][1]))
        # assert np.mean(output[i][1])==label
        unique_labels.append(label)
    print('Standard deviation:',np.std(unique_labels))
    print(output[2][0].shape)
    perm = np.random.permutation(k)
    return output

def prepare_k_folds(X, y, k=3):
    X, y = shuffleInUnison(X,y)
    fold_size = np.size(y)//k
    output = dict()
    for i in range(k-1):
        output[i]=(X[i*fold_size:(i+1)*fold_size,:], y[i*fold_size:(i+1)*fold_size])
    output[k-1]= (X[(k-1)*fold_size:,:],y[(k-1)*fold_size:])
    print(output[0][0].shape)
    return output


#######################################################################################################################
# Function prepare_data_for_fold(fold_dictionary, fold_idx):
# Function to prepare the training and testing set for the fold fold_idx.
#
# Input arguments:
#   - [dict] fold_dictionary: dictionary containing {key, value} = {fold index, (X_fold, y_fold)}
#   - [int] fold_idx: index of the fold to use as testing set
# Output arguments:
#   - [np.array] training data, training labels, testing data, testing labels for fold fold_idx
#######################################################################################################################
def prepare_data_for_fold(fold_dictionary, fold_idx):
    # Testing data and labels
    X_test, y_test = fold_dictionary[fold_idx]
    # Initialising the training data and label array by computing their expected size
    num_features = X_test.shape[1]
    num_training_examples = 0
    for idx in fold_dictionary.keys():
        if idx != fold_idx:
            num_training_examples += fold_dictionary[idx][0].shape[0]
    X_train = np.zeros((num_training_examples,num_features))
    y_train = np.zeros(num_training_examples,dtype=float)
    # Filling out training data and label arrays
    current_idx = 0
    for idx in fold_dictionary.keys():
            if idx != fold_idx:
                current_num_examples = fold_dictionary[idx][0].shape[0]
                X_train[current_idx:current_idx+current_num_examples] = fold_dictionary[idx][0]
                y_train[current_idx:current_idx+current_num_examples] = fold_dictionary[idx][1]
                current_idx += current_num_examples
    
    return X_train, y_train, X_test, y_test


if __name__=='__main__':
    to_eval = float(input('Label to evaluate:'))
    #X, y = read_data()
    features = get_extracted_features_cnn()
    X = features[:,:-1]
    y = features[:,-1]
    #X, y = get_extracted_features_autoenc()
    # with open('tmp_data.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(X)
    print(X.shape)
    # X_new = np.zeros((X.shape[0], 100))
    # X_new = preprocessing(X[:,1:])
    # X = X[:,:101]
    # X[:,1:] = X_new
    #X[:,1:] = preprocessing(X[:,1:])

    #X[:,1:]=preprocessing(X[:,1:])
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=202)
    #X_train = preprocessing(X_train)
    start = time()
    regressor = RandomForestRegressor()
    # regressor = SVR(kernel='sigmoid',degree=10, epsilon=0.1, gamma='auto')
    #regressor = GradientBoostingRegressor()
    #regressor = SGDRegressor(max_iter=10000)
    #regressor = KNeighborsRegressor(n_neighbors=1)
    #regressor = Ridge()
    #rfe = RFE(regressor, n_features_to_select=100)
    #print('Training RFE')
    #X = rfe.fit_transform(X, y)
    print('RFE trained in %.2f seconds' % (time()-start))
    # k = X.shape[0]
    # folds = prepare_k_folds(X, y,k)
    k = len(np.unique(X[:,0]))
    folds = loo_folds(X,y)
    rmses = np.zeros(k) # Array to store the accuracies on each fold
    maes = np.zeros(k)
    r2_scores = np.zeros(k)

    x_axis = []
    y_axis = []
    
    for idx in range(k):
        print("Training regression model for fold %d/%d ..." % (idx+1,k))

        X_train, y_train, X_test, y_test = prepare_data_for_fold(folds, idx)
        regressor.fit(X_train, y_train)
        # predictions_train = classifier.predict(X_train)
        # for i in range(len(y_train)):
        #     print('Predicted:', predictions_train[i], 'Actual:', y_train[i])
        predictions_test = regressor.predict(X_test)
        if np.abs(np.min(y_test)-to_eval)<0.1:
            with open('eval.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                rows = np.zeros((len(predictions_test), 2))
                rows[:,0] = predictions_test
                rows[:,1] = y_test
                writer.writerows(rows)
        print('Predicted:', predictions_test, 'Actual:', np.mean(y_test))
        # for i in range(len(y_test)):
        #     print('Predicted:', predictions_test[i], 'Actual:', y_test[i])
        maes[idx] = mean_absolute_error(y_test, predictions_test)
        print("Mean absolute error: ",maes[idx])
        rmses[idx] = np.sqrt(mean_squared_error(y_test,predictions_test))
        print("Mean squared error: ", rmses[idx]**2)
        r2_scores[idx] = regressor.score(X[:,1:], y)
        x_axis.append(np.mean(y_test))
        y_axis.append(maes[idx])



    regr_oob = RandomForestRegressor(oob_score=True)
    regr_oob.fit(X[:,1:], y)
    print('OOB Score:', regr_oob.oob_score_) 
    r_squared = regr_oob.score(X[:,1:], y)
    print('R^2:', r_squared)
    print('Mean r2:', np.mean(r2_scores))
    print('F value:', r_squared/(1-r_squared)*(X.shape[0]-X.shape[1])/(X.shape[1]-1))
    end = time()
    print('All %d regression models trained in %.2f seconds' % (k,end-start))
    print('###################################################################')

    # Display overall evaluation metrics averaged on all folds
    print("All fold MAEs:")
    print(maes)
    print("MAE: %.2f +- %.2f" % (np.mean(maes),np.std(maes)))
    print('###################################################################')
    print("All fold RMSE:")
    print(rmses)
    print("RMSE: %.2f +- %.2f" % (np.mean(rmses),np.std(rmses)))
    print('###################################################################')
    
    indices = np.argsort(x_axis)
    plt.plot(np.array(x_axis)[indices], np.array(y_axis)[indices])
    plt.show()
    # svm = SVR(kernel='precomputed')
    # svm = svm.fit(data, labels)
    # pred = svm.predict(data)
    # for i in range(data.shape[0]):
    #     #print(labels[i])
    #     print(labels[i], 'but predicted', pred[i])
    # print(evaluation(pred, labels))

