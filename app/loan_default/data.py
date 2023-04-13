# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914


import os
import io
import boto3
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from utils.logger import log

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

s3_resource = boto3.resource('s3')
s3 = boto3.client('s3')

def synthetic_datagen(bucket: str, key: str, local_path: str, backend: str = 's3', size: int = 400000):
    """Generates additional synthetic data for benchmarking and testing purposes. Not recommended in production model development.

    Parameters
    ----------
    data : str
        raw data csv file path with file and extension
    size : int
        desired size of final dataset
    backend : str
        backend to be used for pulling raw data csv, "local" or "s3"
    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe with the original data or original plus synthetic augmentation.
    """
    
    if bucket and local_path is None:
        log.warning('You must provide a storage location')
        raise

    if backend == "local":
        data = pd.read_csv(local_path)
        log.info('successfully loaded data from local')
    elif backend == "s3":
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = pd.read_csv(io.BytesIO(obj['Body'].read()))
        log.info('successfully loaded data from s3')
    

    # number of rows to generate
    if size < data.shape[0]:
        pass
    else:
        log.info(f"Generating {size:,} rows of data")
        repeats = size // len(data)
        data = data.loc[np.repeat(data.index.values, repeats + 1)]
        data = data.iloc[:size]
        # perturbing all int/float columns
        person_age = data["person_age"].values + np.random.randint(
            -1, 1, size=len(data)
        )
        person_income = data["person_income"].values + np.random.normal(
            0, 10, size=len(data)
        )
        person_emp_length = data[
            "person_emp_length"
        ].values + np.random.randint(-1, 1, size=len(data))
        loan_amnt = data["loan_amnt"].values + np.random.normal(
            0, 5, size=len(data)
        )
        loan_int_rate = data["loan_int_rate"].values + np.random.normal(
            0, 0.2, size=len(data)
        )
        loan_percent_income = data["loan_percent_income"].values + (
            np.random.randint(0, 100, size=len(data)) / 1000
        )
        cb_person_cred_hist_length = data[
            "cb_person_cred_hist_length"
        ].values + np.random.randint(0, 2, size=len(data))
        # perturbing all binary columns
        perturb_idx = np.random.rand(len(data)) > 0.1
        random_values = np.random.choice(
            data["person_home_ownership"].unique(), len(data)
        )
        person_home_ownership = np.where(
            perturb_idx, data["person_home_ownership"], random_values
        )
        perturb_idx = np.random.rand(len(data)) > 0.1
        random_values = np.random.choice(
            data["loan_intent"].unique(), len(data)
        )
        loan_intent = np.where(perturb_idx, data["loan_intent"], random_values)
        perturb_idx = np.random.rand(len(data)) > 0.1
        random_values = np.random.choice(
            data["loan_grade"].unique(), len(data)
        )
        loan_grade = np.where(perturb_idx, data["loan_grade"], random_values)
        perturb_idx = np.random.rand(len(data)) > 0.1
        random_values = np.random.choice(
            data["cb_person_default_on_file"].unique(), len(data)
        )
        cb_person_default_on_file = np.where(
            perturb_idx, data["cb_person_default_on_file"], random_values
        )
        data = pd.DataFrame(
            list(
                zip(
                    person_age,
                    person_income,
                    person_home_ownership,
                    person_emp_length,
                    loan_intent,
                    loan_grade,
                    loan_amnt,
                    loan_int_rate,
                    data["loan_status"].values,
                    loan_percent_income,
                    cb_person_default_on_file,
                    cb_person_cred_hist_length,
                )
            ),
            columns=data.columns,
        )

        augmented_data = data.drop_duplicates()
        assert len(augmented_data) == size
        augmented_data.reset_index(drop=True)

        return augmented_data

    return data

def process_data(data: pd.DataFrame, target_path: str, bucket: str=None, key: str=None, backend: str="s3"):
    """Function to synthetically generate 4M (default) rows
    from loan default data.

    Parameters
    ----------
    data : pd.DataFrame
         data that has been processed augmented or loaded simply loaded by the synthetic datagen func.
    target_path : str
        path where data should be saved if "local" backend is selected. 
    bucket : str, optional
        if "s3" backend is selected this is the name of the bucket, by default None
    backend : str, optional
        select backend for object storage "local" or "s3", by default "local"
    """
    
    # train test split and apply data transformations
    log.info("Creating training and test sets")
    train, test = train_test_split(data, test_size=0.25, random_state=0)
    num_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    pow_transformer = PowerTransformer()
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                num_imputer,
                [
                    "loan_int_rate",
                    "person_emp_length",
                    "cb_person_cred_hist_length",
                ],
            ),
            (
                "pow",
                pow_transformer,
                ["person_age", "person_income", "loan_amnt", "loan_percent_income"],
            ),
            (
                "cat",
                cat_transformer,
                [
                    "person_home_ownership",
                    "loan_intent",
                    "loan_grade",
                    "cb_person_default_on_file",
                ],
            ),
        ],
        remainder="passthrough",
    )

    # data processing pipeline
    preprocess = Pipeline(steps=[("preprocessor", preprocessor)])
    X_train = train.drop(["loan_status"], axis=1)
    y_train = train["loan_status"]
    X_train = preprocess.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    log.info(f"preprocess named steps: {preprocess.named_steps['preprocessor']}, {type(preprocess.named_steps['preprocessor'])}")
    X_test = test.drop(["loan_status"], axis=1)
    y_test = test["loan_status"]
    X_test = preprocess.transform(X_test)
    X_test = pd.DataFrame(X_test)
    
    # export processing pipeline for inference data processing
    with open("preprocessor.sav", 'wb') as file:
            pickle.dump(preprocessor, file)

    # save processed data 
    if backend == "local":
        X_train.to_csv(os.path.join(target_path, "Xtrain.csv"), index=False, header=None)
        y_train.to_csv(os.path.join(target_path, "ytrain.csv"), index=False, header=None)
        y_test.to_csv(os.path.join(target_path, "ytest.csv"), index=False, header=None)
        X_test.to_csv(os.path.join(target_path, "Xtest.csv"), index=False, header=None)
        log.info('successfully saved data to local')
    elif backend == "s3":
        
        key_path = '/'.join(key.split('/')[0:-1])
        
        csv_buffer = io.StringIO()
        X_train.to_csv(csv_buffer, index=False, header=None)
        s3_resource.Object(bucket, key_path + '/Xtrain.csv').put(Body=csv_buffer.getvalue())
        
        csv_buffer = io.StringIO()
        y_train.to_csv(csv_buffer, index=False, header=None)
        s3_resource.Object(bucket, key_path + '/ytrain.csv').put(Body=csv_buffer.getvalue())
        
        csv_buffer = io.StringIO()
        y_test.to_csv(csv_buffer, index=False, header=None)
        s3_resource.Object(bucket, key_path + '/ytest.csv').put(Body=csv_buffer.getvalue())
        
        csv_buffer = io.StringIO()
        X_test.to_csv(csv_buffer, index=False, header=None)
        s3_resource.Object(bucket, key_path + '/Xtest.csv').put(Body=csv_buffer.getvalue())
        
        with open("preprocessor.sav", "rb") as file:
            s3.upload_fileobj(file, bucket, key_path + '/preprocessor.sav')
        
        log.info('successfully saved data to s3')
