FROM continuumio/miniconda3

RUN /bin/bash -c "/opt/conda/bin/conda create \
        -n ml -y --quiet -c conda-forge -c nvidia \
        python=3 \
        jupyter \
        xlrd \
        seaborn \
        scikit-learn=1.6 \
        scikit-image \
        tensorboard \
        numexpr \
        statsmodels \
        scipy \
        matplotlib \
        joblib \
        ipykernel \
        numpy \
        pandas \
        rpy2 \
        tqdm \
        nb_conda_kernels \
        dask \
        dask-ml \
        dask-xgboost \
        pyreadr \
        gpy \
        gpyopt \
        sktime \
        onnx \
        skl2onnx \
        cachey \
        fastparquet \
        xgboost \
        xarray \
        datashader \
        numba \
        pytorch::pytorch=2.5.1 \
        scikit-learn-extra \
        plotly \
        bokeh && \
    conda install -c conda-forge jupyterlab && \
    conda clean -t && \
    source activate base && \
    conda activate ml && \
    pip install kaggle opendatasets && \
    ipython kernel install --user --name=ML && \
    mkdir -p /opt/notebooks && \
    mkdir -p /opt/data"

ENTRYPOINT [ "/bin/bash", "-c", "jupyter lab \
    --notebook-dir=/opt/notebooks --ip='*' \
    --no-browser --allow-root" ]

CMD [ "--port=8888" ]