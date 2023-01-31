from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from logger import log
from credit_risk.xgb_model import Model

app = FastAPI()


class TrainData(BaseModel):
    path: Path = Path.cwd()
    num_samples: int = 400_00_00


@app.post("/train")
async def train(data:TrainData):
    """Train the model.

    This endpoint trains the model using the provided path for the data and number of samples to generate.
    The default number of samples is 400_00_00.

    Args:
        path (Path): Path to the data.
        num_samples (int, optional): Number of samples to generate. Defaults to 400_00_00.

    Returns:
        dict: A dictionary containing a success message and the validation scores.
    """
    clf = Model("disk", path=data.path)
    clf.generate_data(data.num_samples)
    clf.preprocess_data()
    log.info("Training model")
    clf.train()
    clf.save()
    log.info("Validating model, model saved to {path} / models")
    scores = clf.validate()
    return {"msg": "Model trained successfully","result": scores}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")

