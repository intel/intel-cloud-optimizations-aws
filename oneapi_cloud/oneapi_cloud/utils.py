import os
from pathlib import Path
import sys

import boto3
from joblib import dump, load


class Store:
    def __init__(self, model=None, model_store="disk", bucket=None, path=None):
        self.model = model
        self.store_store = model_store
        self.path = path
        self.bucket = bucket
        self.model_name = "model.joblib"
        self.path = self.path / self.model_name

    def _to_disk(self):
        dump(self.model, self.path)

    def _from_disk(self):
        return load(self.path)

    def _to_cloud(self):
        if self.model_store == "s3":
            self._to_s3()

    def _from_cloud(self):
        if self.model_store == "s3":
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
        if self.model_store == "disk":
            return self._from_disk()
        elif self.model_store == "s3":
            return self._from_cloud()
        else:
            print("storage backend not supported")  # TODO change to logging
            sys.exit(1)

    def put(self):
        """put model to disk or cloud."""
        if self.model_store == "disk":
            self._to_disk()
        elif self.model_store == "s3":
            self._to_cloud()
        else:
            print("storage backend not supported")  # TODO change to logging
            sys.exit(1)
