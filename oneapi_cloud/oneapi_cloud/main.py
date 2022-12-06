import os
from pathlib import Path

from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.svm import SVC
from sklearn.datasets import load_iris

from utils import Store


class Model:
    def __init__(self, model_store="disk"):
        self.model = SVC()
        self.path = os.path.join(Path(__file__).parent.resolve(), "model.joblib")
        self.store = Store(model_store=model_store, path=self.path)

    @classmethod
    def dataset():
        """get iris dataset."""
        return load_iris(return_X_y=True)

    def train(self):
        """a simple svm classifier."""
        X, y = self.dataset()
        self.model.fit(X, y)
        return self.model

    def infer(self):
        model = self.store.get(self.path)
        X, _ = self.dataset()
        return model.predict(X[0:235])

    def save(self):
        """pickle model."""
        self.store.put()
