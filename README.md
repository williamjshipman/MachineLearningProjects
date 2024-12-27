# Machine Learning Projects

Copyright 2021, William Shipman.

This repository contains a collection of personal machine learning projects.

## Installation instructions

Create a new conda environment using the following (new version of environment):

    conda create -n ml -c conda-forge -c nvidia python=3 xlrd seaborn scikit-learn=1.6 scikit-image tensorboard numexpr statsmodels scipy matplotlib joblib ipykernel numpy pandas rpy2 tqdm nb_conda_kernels dask dask-ml dask-xgboost pyreadr gpy gpyopt sktime onnx skl2onnx cachey fastparquet xgboost xarray datashader numba pytorch::pytorch=2.5.1 scikit-learn-extra plotly bokeh

Create a new conda environment using the following (old version of environment):

    conda create -n ml -c conda-forge python=3 xlrd seaborn scikit-learn scikit-image tensorboard numexpr statsmodels scipy matplotlib joblib ipykernel numpy tensorflow=2.5 pandas rpy2 bcolz tqdm nb_conda_kernels dask dask-ml dask-xgboost orange3 orange3-imageanalytics orange3-network orange3-educational orange3-explain orange3-timeseries scikit-learn-extra pyreadr gpy gpyopt sktime onnx skl2onnx cachey fastparquet xgboost xarray datashader numba pytorch plotly bokeh

Some requirements are only available on PyPI, install them as follows:

    conda activate ml
    pip install scikeras kaggle opendatasets

Finally, you can create a Jupyter kernel for this environment:

    ipython kernel install --user --name=ML