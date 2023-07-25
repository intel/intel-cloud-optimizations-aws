# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914


import os
import io
import daal4py as d4p
import numpy as np
import pandas as pd
import xgboost as xgb
import boto3

from utils.base_model import AbstractModel
from utils.logger import log
from utils.storage import store

from sklearnex import patch_sklearn
from sklearn.metrics import classification_report, roc_auc_score
patch_sklearn()

np.random.seed(42)

s3_resource = boto3.resource('s3')
s3 = boto3.client('s3')


class Model(AbstractModel):
    
    def __init__(self, local_model_path: str, model_name: str, local_data_path: str, bucket: str=None, model_key: str=None, data_key: str=None, backend: str="local"):
        """Class to manage the training and validation of XGBoost and Daal4Py Models.

        Parameters
        ----------
        model_path : str
            if backend is "local", this is the path to model location
        data_path : str
            _description_
        bucket : str, optional
            _description_, by default None
        backend : str, optional
            _description_, by default "local"
        """
        self.local_model_path = local_model_path
        self.backend = backend
        self.local_data_path = local_data_path
        self.model_name = model_name
        self.model_key = model_key
        self.data_key = data_key
        self.bucket = bucket
        self.store = store(backend=backend, bucket=self.bucket, path=self.local_model_path, model_name=self.model_name, key=self.model_key)
        self.X_train = []
        self.y_train = []
        self.y_test = []
        self.X_test = []
        self.DMatrix = []
    
    def load_data(self):
        # load data
        if self.backend == "local":
            self.X_train = pd.read_csv(os.path.join(self.local_data_path, "Xtrain.csv"), header=None)
            self.y_train = pd.read_csv(os.path.join(self.local_data_path, "ytrain.csv"), header=None)
            self.y_test = pd.read_csv(os.path.join(self.local_data_path, "ytest.csv"), header=None)
            self.X_test = pd.read_csv(os.path.join(self.local_data_path, "Xtest.csv"), header=None)
            log.info('successfully loaded data from local disk')
        elif self.backend == "s3":
            Xtrain_obj = s3.get_object(Bucket=self.bucket, Key=self.data_key + '/Xtrain.csv')
            self.X_train = pd.read_csv(io.BytesIO(Xtrain_obj['Body'].read()), header=None)
            
            ytrain_obj = s3.get_object(Bucket=self.bucket, Key=self.data_key + '/ytrain.csv')
            self.y_train = pd.read_csv(io.BytesIO(ytrain_obj['Body'].read()), header=None)
            
            ytest_obj = s3.get_object(Bucket=self.bucket, Key=self.data_key + '/ytest.csv')
            self.y_test = pd.read_csv(io.BytesIO(ytest_obj['Body'].read()), header=None)
            
            Xtest_obj = s3.get_object(Bucket=self.bucket, Key=self.data_key + '/Xtest.csv')
            self.X_test = pd.read_csv(io.BytesIO(Xtest_obj['Body'].read()), header=None)
            log.info('successfully loaded data from s3')
        
        print(self.y_train)
        
        self.DMatrix = xgb.DMatrix(self.X_train.values, self.y_train.values)

    def train(self):
        # define model
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "nthread": 4,  # flags.num_cpu
            "tree_method": "hist",
            "learning_rate": 0.02,
            "max_depth": 10,
            "min_child_weight": 6,
            "n_jobs": 4,  # flags.num_cpu,
            "verbosity": 0,
            "silent": 1,
        }

        log.info("Training XGBoost model")
        self.clf = xgb.train(params, self.DMatrix, num_boost_round=500)
        self.clf = d4p.get_gbt_model_from_xgboost(self.clf)

    def validate(self):
        """Validate the model performance by computing fairness metrics and classification metrics.

        Returns
        -------
        dict
             A dictionary containing the following keys:
                'auc' (float): The area under the receiver operating characteristic curve (AUC).
                'precision' (dict): A dictionary of precision scores, with keys 'Non-Default' and 'Default'.
                'recall' (dict): A dictionary of recall scores, with keys 'Non-Default' and 'Default'.
                'f1-score' (dict): A dictionary of F1 scores, with keys 'Non-Default' and 'Default'.
                'support' (dict): A dictionary of support values, with keys 'Non-Default' and 'Default'.
                'parity' (dict): A dictionary of fairness metrics, such as demographic parity and equal opportunity.
        """
        daal_model = self.store.get()

        y_hat = (
            d4p.gbt_classification_prediction(
                nClasses=2, resultsToEvaluate="computeClassProbabilities"
            )
            .compute(self.X_test, daal_model)
            .probabilities[:, 1]
        )
        auc = roc_auc_score(self.y_test, y_hat)
        results = classification_report(
            self.y_test,
            y_hat > 0.5,
            target_names=["Non-Default", "Default"],
            output_dict=True,
        )
        results.update({"auc": auc})
        return results

    def save(self):
        log.info("Saving model")
        self.store.put(self.clf)
