import os
import tempfile
from pathlib import Path
from typing import Any

import boto3
import joblib


class S3Store:
    def __init__(self, model_name: str, path: str = None, bucket: str = None, key: str = None):
        """S3 bucket save and pull objects
        Parameters
        ----------
        model_name : str
            name of model with extension
        path : str, optional
            path to model storage location on local disk, by default None
        bucket : str, optional
            name of S3 bucket, by default None
        key : str, optional
            key inside S3 bucket, by default None
        """
        self.model_name = model_name
        self.path = Path(path) / self.model_name
        self.bucket = bucket
        self.key = key + '/' + self.model_name
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
    def __init__(self, model_name: str, path: str = None):
        """Local disk save and pull objects

        Parameters
        ----------
        model_name : str
            name of model with extension
        path : str, optional
            path on local disk to the model location, by default None
        """
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
    """Storage handler function for S3 and Local Disk storage

    Parameters
    ----------
    backend : str
        backend used for storage of workflow objects, "local" or "s3"
    model_name : str
        name of model with extension
    path : str, optional
        path on local disk to the model location, by default None
    bucket : str, optional
        name of S3 bucket, by default None
    key : str, optional
        key inside S3 bucket, by default None

    Returns
    -------
    XGboost Model
        returns .joblib XGBoost Model

    Raises
    ------
    ValueError
        "invalid backend" when "s3" or "local" storage are not used.
    """

    if backend == "s3":
        return S3Store(model_name, path, bucket, key)
    elif backend == "local":
        return DiskStore(model_name, path)
    else:
        raise ValueError("invalid backend")


__all__ = ["store"]
