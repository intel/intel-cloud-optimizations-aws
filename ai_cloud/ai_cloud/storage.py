import os
import tempfile
from pathlib import Path
from typing import Any

import boto3
import joblib


class S3Store:
    def __init__(
        self, model_name: str, path: str = None, bucket: str = None, key: str = None
    ):
        self.model_name = model_name
        self.path = Path(path) / self.model_name
        self.bucket = bucket
        self.key = Path(key) / self.model_name
        self._s3_client = boto3.client("s3")

    def put(self, model: Any) -> None:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as fh:
            joblib.dump(model, fh)
            temp_path = fh.name
        with open(temp_path, "rb") as fh:
            self._s3_client.upload_fileobj(fh, self.bucket, self.key)
        os.remove(temp_path)

    def get(self) -> Any:
        local_folder_path = os.path.dirname(self.path)
        Path(local_folder_path).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as fh:
            temp_path = fh.name
        self._s3_client.download_file(self.bucket, self.key, temp_path)
        with open(temp_path, "rb") as fh:
            model = joblib.load(fh)
        os.remove(temp_path)
        return model


class DiskStore:
    """
     Class for storing and retrieving models from disk.

    :param model_name: name of the model
    :type model_name: str
    :param path: local path to folder containing model
    :type path: str
    """

    def __init__(self, model_name: str, path: str = None):
        self.model_name = model_name
        self.path = Path(path) / self.model_name

    def put(self, model: Any) -> None:
        with open(self.path, "wb") as fh:
            joblib.dump(model, fh.name)

    def get(self) -> None:
        with open(self.path, "rb") as fh:
            return joblib.load(fh.name)


def store(
    backend: str, model_name: str, path: str = None, bucket: str = None, key: str = None
):
    """
    Selector function for storing and retrieving models.

    :param backend: storage backend to use. Can be 'disk' or 's3'
    :type backend: str
    :param model_name: name of the model
    :type model_name: str
    :param path: local path to folder containing model
    :type path: str
    :param bucket: S3 bucket name
    :type bucket: str
    :param key: the folder on the S3 bucket. eg. the 's3folder' portion of s3://mybucket/s3folder
    :type key: str
    """

    if backend == "s3":
        return S3Store(model_name, path, bucket, key)
    elif backend == "disk":
        return DiskStore(model_name, path)
    else:
        raise ValueError("invalid backend")


__all__ = ["store"]
