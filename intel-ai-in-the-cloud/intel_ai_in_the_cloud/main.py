from pathlib import Path

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC
from sklearn.datasets import load_iris

from utils import Store



def dataset():
    """get iris dataset."""
    return load_iris(return_X_y=True)

def train():
    """a simple svm classifier."""
    model = SVC()
    X, y = dataset()
    model.fit(X, y)
    return model


def infer(path: Path=Path("./models/model0.pkl")):
    model = Store().from_disk(path)
    X,_  = dataset()
    return model.predict(X[0:235])


def save(model, path: Path=Path("./models/model0.pkl")):
    """pickle model."""
    Store(model).to_disk(path)
    return path

m = train()
p = save(m)
r = infer(p)
print(f"result: {r}")


    
    

