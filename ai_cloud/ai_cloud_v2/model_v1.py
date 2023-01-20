import os
from pathlib import Path

from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.datasets import load_iris
from sklearn.svm import SVC

from utils import Store


class Model:
    def __init__(
        self, backend="disk", bucket=None, path=Path(__file__).parent.resolve()
    ):
        self.model = SVC()
        self.path = path / "models"
        self.store = Store(
            backend=backend, bucket=bucket, path=self.path, model_name="model.joblib"
        )

    def dataset(self):
        """get iris dataset."""
        return load_iris(return_X_y=True)

    def train(self):
        """a simple svm classifier."""
        X, y = self.dataset()
        self.model.fit(X, y)
        print("training model.")
        return self.model

    def infer(self):
        X, _ = self.dataset()
        model = self.store.get()
        return model.predict(X[0:235])

    def save(self):
        """pickle model."""
        print("model saved")
        self.store.put(self.model)


if __name__ == "__main__":
    clf = Model("disk")
    clf.train()
    clf.save()
    print(clf.infer())
