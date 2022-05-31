import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import mannwhitneyu

if __name__=='__main__':
    values = pd.read_csv('eval.csv', header=None).values
    pred = values[:,0]
    labels = values[:,1]
    mae = mean_absolute_error(pred, labels)
    rmse = np.sqrt(mean_squared_error(pred, labels))
    corr = np.corrcoef(pred, labels)
    print('MAE:',mae)
    print('RMSE:', rmse)
    print('Correlation:', corr)
    errors_cnn = np.abs(pred-labels)
    values_hc = pd.read_csv('eval_hc.csv', header=None).values
    pred_hc = values_hc[:,0]
    labels_hc = values_hc[:,1]
    errors_hc = np.abs(pred_hc-labels_hc)
    print(np.mean(errors_hc))
    print(mannwhitneyu(errors_hc, errors_cnn, alternative='greater').pvalue)
    print(np.corrcoef(pred_hc, labels_hc))