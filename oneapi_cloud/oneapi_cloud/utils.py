import os
import sys
from pathlib import Path

import boto3
import joblib


class Store:
    def __init__(self, backend="disk", bucket=None, model_name=None, path=None):
        self.model = None
        self.backend = backend
        self.path = path
        self.bucket = bucket
        self.model_name = model_name
        self.path = self.path / self.model_name

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
        """upload model binary to s3."""
        s3 = boto3.client("s3")
        with open(self.path, "rb") as fh:
            s3.upload_fileobj(fh, self.bucket, self.model)

    def _from_s3(self):
        """upload model binary to s3."""
        s3 = boto3.client("s3")
        with open(self.path, "wb") as fh:
            s3.download_fileobj(self.bucket, self.model_name, fh)
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
