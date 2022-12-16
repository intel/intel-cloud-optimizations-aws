import os
import sys
from pathlib import Path

import boto3
import joblib

class Store:
    def __init__(self, backend="disk", bucket=None, key=None, model_name=None, path=None):
        self.model = None
        self.backend = backend
        self.path = path # local path to folder containing model
        self.bucket = bucket
        self.model_name = model_name
        self.path = self.path / self.model_name #full local path to model file
        self.key = key #the folder on the s3 bucket. eg. the 's3folder' portion of s3://mybucket/s3folder
        self.key = key / self.model_name #the path after the s3 bucket. eg., s3://mybucket/s3pathtofile

    def _to_disk(self):
        joblib.dump(self.model, self.path)

    def _from_disk(self):
        return joblib.load(self.path)

    def _to_cloud(self):
        if self.backend == "s3":
            self._to_s3()

    def _from_cloud(self):
        if self.backend == "s3":
            return self._from_s3(self.bucket, self.model_name)

    def _to_s3(self):
        """
        Upload model binary to s3
            
        Example AWS S3 URI:
            s3://bucket/s3pathtofile
        
        """
        s3 = boto3.client("s3")
        with open(self.path, "rb") as fh:
            s3.upload_fileobj(fh, self.bucket, self.key)

    def _from_s3(self):
        """download model binary to s3 to local disk"""
        s3 = boto3.client("s3")
        local_folder_path = os.path.dirname(self.path)
        Path(local_folder_path).mkdir(parents=True, exist_ok=True) #need to create folder before downloading file
        with open(self.path, "wb") as fh:
            s3.download_fileobj(self.bucket, self.key, fh)
        return open(self.path, "rb")

    def get(self):
        """get model from disk or cloud."""
        if self.backend == "disk":
            return self._from_disk()
        elif self.backend == "s3":
            return self._from_cloud()
        else:
            print("storage backend not supported")  # TODO change to logging
            sys.exit(1)

    def put(self, model):
        """put model to disk or cloud."""
        self.model = model
        if self.backend == "disk":
            self._to_disk()
        elif self.backend == "s3":
            self._to_cloud()
        else:
            print("storage backend not supported")  # TODO change to logging
            sys.exit(1)
