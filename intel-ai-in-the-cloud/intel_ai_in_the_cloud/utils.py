from pathlib import Path
from joblib import dump, load

class Store:
    def __init__(self, model=None):
        self.model = model
    def to_disk(self, path: Path):
        dump(self. model, path)

    def to_cloud(self, path):
        pass
    @staticmethod 
    def from_disk(path: Path):
        return load(path)

    def from_cloud(self, path: Path):
        pass
