# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914


import os
import daal4py as d4p
import pandas as pd
import pickle
import boto3

from utils.storage import store
from sklearn.pipeline import Pipeline
from utils.logger import log

s3 = boto3.resource('s3')


def pred(data: pd.DataFrame, model_name: str, preprocessor_path: str, backend: str = "local", bucket: str = None, model_key: str = None, data_key: str = None, model_path: str = None):
    """Returns predictions based on the input `data` using the stored model.

    Parameters
    ----------
    data : pd.DataFrame
        receives a dataframe converted from json payload in serve.py function
    model_name : str
        name of model with extension
    preprocessor_path : str
        path where preprocessor config is stored
    backend : str, optional
        backend to be used for pulling model, "local" or "s3", by default "local"
    bucket : str, optional
        if backend is "s3" , this is the name of the s3 bucket
    model_path : str, optional
        if backend is "local, this is the path to folder containing model, by default None

    Returns
    -------
    str
        "True" for credit approval and "False" for denial
    """

    # loading preprocessor
    if backend == "local":
        preprocessor_file = open(os.path.join(preprocessor_path, "preprocessor.sav"), 'rb')
        preprocessor = pickle.load(preprocessor_file)
    elif backend == "s3":
        preprocessor = pickle.loads(s3.Object(bucket, data_key + "/preprocessor.sav").get()['Body'].read())
    
    # preprocessor transformations
    preprocess = Pipeline(steps=[("preprocessor", preprocessor)])
    data = preprocess.transform(data)
    data = pd.DataFrame(data)

    # loading model
    model_storage = store(backend=backend, bucket=bucket, key=model_key, path=model_path, model_name=model_name)
    daal_model = model_storage.get()

    # daal model inference
    log.info("Starting Daal4Py Inference")
    daal_prediction = (
        d4p.gbt_classification_prediction(
            nClasses=2, resultsToEvaluate="computeClassProbabilities"
        )
        .compute(data, daal_model)
        .probabilities[:, 1]
    )

    log.info("Inference Complete")
    
    log.info("Exporting Predictions")
    predictions = []
    for probability in daal_prediction:
        predicted_label, probability = 'True' if probability > 0.5 else 'False', probability
        predictions.append((predicted_label, probability))

    return predictions