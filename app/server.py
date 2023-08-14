import uvicorn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from utils.logger import log
from loan_default.model import Model
from loan_default.predict import pred
from loan_default.data import process_data, synthetic_datagen



app = FastAPI()

# payload models
class PreprocessingPayload(BaseModel):
    local_path: str=None
    target_path: str=None
    bucket: str=None
    key: str=None
    size: int = 4000000
    backend: str="s3"
class TrainPayload(BaseModel):
    local_data_path: str
    model_name: str
    local_model_path: str=None
    bucket: str=None
    model_key: str=None 
    data_key: str=None
    backend: str="s3"
class PredictionPayload(BaseModel):
    sample: list
    model_name: str
    preprocessor_path: str=None
    local_model_path: str=None
    bucket: str=None
    model_key: str=None
    data_key: str=None
    csv_payload: str=None
    backend: str="s3"


@app.get("/ping")
async def ping():
    """Ping server to determine status

    Returns
    -------
    API response
        response from server on health status
    """
    return {"message":"Server is Running"}

@app.post("/data")
async def train(payload:PreprocessingPayload):
    """Preprocess Credit Risk data
    This endpoint preprocess data and stores in data lake or in other structured format. In this codebase,
    it also handles the expansion of the dataset for benchmarking purposes.
    
    Parameters
    ----------
    payload : PreprocessingPayload
        Data endpoint payload model

    Returns
    -------
    API response
        response from server on data endpoint
    """
    augmented_data = synthetic_datagen(size=payload.size, bucket=payload.bucket, key=payload.key, local_path=payload.local_path, backend=payload.backend)
    process_data(data=augmented_data, target_path=payload.target_path, bucket=payload.bucket, key=payload.key, backend=payload.backend)

    return {"message": f"Data successfully processed and saved"}

@app.post("/train")
async def train(payload:TrainPayload):
    """Train the model.
    This endpoint trains the model using the provided path for the data.

    Parameters
    ----------
    payload : TrainPayload
        Data endpoint payload model

    Returns
    -------
    API response
        response from server on train endpoint
    """

    clf = Model(backend=payload.backend, model_name=payload.model_name, model_key=payload.model_key, data_key=payload.data_key, 
                bucket=payload.bucket, local_model_path=payload.local_model_path, local_data_path=payload.local_data_path)
    log.info("loading data")
    clf.load_data()
    log.info("Training model")
    clf.train()
    clf.save()
    log.info(f"Model saved to {payload.backend}")
    scores = clf.validate()
    log.info(f"Validation Scores: {scores}")
    return {"msg": "Model trained succesfully", "validation scores": scores}

@app.post("/predict")
async def predict(payload:PredictionPayload):
    """Prediction endpoint

    Parameters
    ----------
    payload : PredictionPayload
        Prediction endpoint payload model

    Returns
    -------
    API response
        response from server on predict endpoint
    """
    
    if payload.csv_payload:
        sample = pd.read_csv(payload.csv_payload)
    else:
        sample = pd.json_normalize(payload.sample)
    predictions = pred(data=sample, backend=payload.backend, bucket=payload.bucket, model_key=payload.model_key, data_key=payload.data_key, 
                       model_path=payload.local_model_path, model_name=payload.model_name,  preprocessor_path=payload.preprocessor_path)
    log.info(f'Prediction Output: {predictions}')
    return {"msg": "Model Inference Complete", "Prediction Output": predictions} 

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5000, log_level="info")