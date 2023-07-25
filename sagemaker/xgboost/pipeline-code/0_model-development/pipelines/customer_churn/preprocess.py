#Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
#
# Modified by Eduardo Alvarez on 12/13/2022
"""Feature engineers the customer churn dataset."""
import argparse
import logging
import pathlib


import boto3
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    print(input_data)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/raw-data.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")

    # read in csv
    df = pd.read_csv(fn)

    # drop the "Phone" feature column
    df = df.drop(["Phone"], axis=1)

    # Change the data type of "Area Code"
    df["Area Code"] = df["Area Code"].astype(object)
    
    X, y = df.drop(['Churn?'], axis=1), df['Churn?']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.4)
    
    obj_cols = X_train.select_dtypes(include=["object"]).columns
    num_cols = X_train.select_dtypes(exclude=["object"]).columns
    
    one_hot = OneHotEncoder()
    s_scaler = StandardScaler()
    
    one_hot = one_hot.fit(X_train[obj_cols])
    s_scaler = s_scaler.fit(X_train[num_cols])
    # Convert categorical variables into dummy/indicator variables.
    a = one_hot.transform(X_train[obj_cols]).toarray()
    b = s_scaler.transform(X_train[num_cols])
    X_train = np.c_[a, b]
    X_val = np.c_[one_hot.transform(X_val[obj_cols]).toarray(), s_scaler.transform(X_val[num_cols])]
    X_test = np.c_[one_hot.transform(X_test[obj_cols]).toarray(), s_scaler.transform(X_test[num_cols])]
    
    lb = LabelBinarizer().fit(y_train)
    y_train = lb.transform(y_train)
    y_val = lb.transform(y_val)
    y_test = lb.transform(y_test)
    # Store Transformations
    trans = {
        'One_Hot': one_hot,
        'scaler': s_scaler,
        'label': lb,
        'obj_cols': list(obj_cols),
        'num_cols': list(num_cols)
    }

    # Split the data 
    pd.DataFrame(np.c_[y_train, X_train]).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(np.c_[y_val, X_val]).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(np.c_[y_test, X_test]).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
    with open("transformation.sav", 'wb') as file:
        pickle.dump(trans, file)
    prefix = 'cust-churn-model'
    trans_bucket = "sagemaker-us-east-1-000257663186"
    file_path = os.path.join(prefix, 'transformation','transformation.sav')
    RawData = boto3.Session().resource('s3').Bucket(trans_bucket).Object(file_path).upload_file('transformation.sav')
