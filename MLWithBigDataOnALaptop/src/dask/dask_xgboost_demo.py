import os

import numpy as np

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler # Scale and shift each feature to have zero mean and standard deviation equal to one.
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from dask.distributed import Client
from dask import dataframe as ddf
from dask_ml.wrappers import Incremental

import xgboost as xgb

if __name__ == '__main__':
    dask_client = Client(n_workers=1, threads_per_worker=1, memory_limit='8GiB')
    training_data = ddf.read_hdf('data/train.hdf', key='/train_*')
    X_data = training_data.iloc[:, 3:]
    y_data = training_data.faultNumber
    y_binary = (y_data > 0).astype(np.int8)
    scaler = Incremental(StandardScaler(), scoring='accuracy')
    scaler.fit(X_data, y_binary)
    X_norm = scaler.transform(X_data)
    clf = xgb.dask.DaskXGBClassifier(n_estimators=10, tree_method='hist')
    clf.client = dask_client
    clf.fit(X_norm, y_data, eval_set=[(X_norm, y_data)])
    testing_data = ddf.read_hdf('data/test.hdf', key='/test_*')
    X_data_test = testing_data.iloc[:, 3:]
    y_data_test = testing_data.faultNumber
    y_binary_test = (y_data_test > 0).astype(np.int32)
    X_norm_test = scaler.transform(X_data_test)
    y_pred_dask = clf.predict(X_norm_test)
    y_pred = y_pred_dask.astype(np.int64).compute()
    print(classification_report(y_data_test, y_pred))
    fig: plt.Figure = plt.figure(figsize=(6.4*4, 4.8*4))
    ConfusionMatrixDisplay.from_predictions(y_data_test, y_pred, ax=fig.add_subplot(1, 1, 1), normalize='true')
    plt.show()
    dask_client.shutdown()
    del dask_client