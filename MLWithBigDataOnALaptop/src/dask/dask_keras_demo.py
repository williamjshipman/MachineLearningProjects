import os

import numpy as np

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler # Scale and shift each feature to have zero mean and standard deviation equal to one.
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from dask.distributed import Client
from dask import dataframe as ddf
from dask.array import blockwise
from dask_ml.wrappers import Incremental

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

from scikeras.wrappers import KerasClassifier

def build_model():
    input_layer = keras.Input(shape=(52,))
    fc_1 = keras.layers.Dense(64, activation='tanh')(input_layer)
    fc_2 = keras.layers.Dense(64, activation='tanh')(fc_1)
    fc_3 = keras.layers.Dense(64, activation='tanh')(fc_2)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(fc_3)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

if __name__ == '__main__':
    print('Connecting to Dask scheduler...')
    dask_client = Client(n_workers=1, threads_per_worker=4, memory_limit='16GiB')
    print('Connecting to Dask scheduler...Done.')
    
    print('Loading training data...')
    training_data = ddf.read_hdf('data/train.hdf', key='/train_*')
    X_data = training_data.iloc[:, 3:]
    y_data = training_data.faultNumber
    y_binary = (y_data > 0).astype(np.int8)
    print('Loading training data...Done.')
    
    print('Scale to zero mean and unit variance...')
    scaler = Incremental(StandardScaler(), scoring='accuracy')
    scaler.fit(X_data, y_binary)
    X_norm = scaler.transform(X_data)
    print('Scale to zero mean and unit variance...Done.')

    print('Fitting Keras classifier...')
    clf = Incremental(KerasClassifier(build_fn=build_model, batch_size=1024, epochs=1, optimizer='adam', loss='binary_crossentropy'), scoring='accuracy')
    clf.fit(X_norm, y_binary, classes=list(range(2)))
    print('Fitting Keras classifier...Done')

    print('Loading test data...')
    testing_data = ddf.read_hdf('data/test.hdf', key='/test_*')
    X_data_test = testing_data.iloc[:, 3:]
    y_data_test = testing_data.faultNumber
    y_binary_test = (y_data_test > 0).astype(np.int32)
    print('Loading test data...Done')
    
    print('Scaling test data...')
    X_norm_test = scaler.transform(X_data_test)
    print('Scaling test data...Done.')
    
    print('Running classifier...')
#     clf.set_params(estimator__predict__batch_size=10240)
#     y_pred_dask = clf.predict(X_norm_test)
    y_pred = np.zeros((int(X_norm_test.shape[0]),), dtype=np.int32)
    for start in range(0, int(X_norm_test.shape[0]), 128):
        end = min(start + 128, int(X_norm_test.shape[0]))
        y_pred[start:end] = clf.predict(X_norm_test[start:end, :]).compute()
#     y_pred_dask = blockwise(lambda x: clf.predict(x), 'i', X_norm_test, 'ij', dtype=np.int64)
#     y_pred = y_pred_dask.astype(np.int64).compute()
    print('Running classifier...Done.')
    
    print(classification_report(y_binary_test, y_pred))
    fig: plt.Figure = plt.figure(figsize=(6.4*4, 4.8*4))
    ConfusionMatrixDisplay.from_predictions(y_binary_test, y_pred, ax=fig.add_subplot(1, 1, 1), normalize='true')
    plt.show()
    
    dask_client.shutdown()
    del dask_client