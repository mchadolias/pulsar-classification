from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import uvicorn
import pickle
import numpy as np
import pandas as pd
from typing import List

MODEL_PATH = "models/best_xgboost_model.pkl"

app = FastAPI(title="HTRU2 Pulsar Classifier API", version="1.0")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Use the exact feature names from the error message
FEATURE_NAMES = [
    "ip_mean",  # Integrated Profile Mean
    "ip_std",  # Integrated Profile Standard Deviation
    "ip_kurtosis",  # Integrated Profile Excess Kurtosis
    "ip_skewness",  # Integrated Profile Skewness
    "dm_mean",  # DM-SNR Curve Mean
    "dm_std",  # DM-SNR Curve Standard Deviation
    "dm_kurtosis",  # DM-SNR Curve Excess Kurtosis
    "dm_skewness",  # DM-SNR Curve Skewness
]


class HTRUInputSingle(BaseModel):
    features: List[float] = Field(
        ...,
        description="List of 8 numerical HTRU2 features for a single sample.",
    )

    @validator("features")
    def validate_features(cls, v):
        if len(v) != 8:
            raise ValueError("Exactly 8 features are required")
        return v


class HTRUInputBatch(BaseModel):
    samples: List[List[float]] = Field(
        ..., description="List of samples. Each sample contains 8 numerical HTRU2 features."
    )

    @validator("samples")
    def validate_samples(cls, v):
        for i, sample in enumerate(v):
            if len(sample) != 8:
                raise ValueError(f"Sample {i} must contain exactly 8 features, got {len(sample)}")
        return v


class PredictionResponse(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    is_pulsar: bool


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


def predict_single(features: List[float]) -> float:
    """Predict probability for a single sample"""
    # Convert to DataFrame with proper column names
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    probability = model.predict_proba(features_df)[0, 1]
    return float(probability)


def predict_batch(samples: List[List[float]]) -> List[float]:
    """Predict probabilities for multiple samples"""
    # Convert to DataFrame with proper column names
    samples_df = pd.DataFrame(samples, columns=FEATURE_NAMES)
    probabilities = model.predict_proba(samples_df)[:, 1]
    return [float(prob) for prob in probabilities]


@app.post("/predict", response_model=PredictionResponse)
def predict_single_sample(payload: HTRUInputSingle) -> PredictionResponse:
    """
    Predict if a single sample is a pulsar.

    Expected features (8 numerical values in this exact order):
    1. ip_mean - Mean of the integrated profile
    2. ip_std - Standard deviation of the integrated profile
    3. ip_kurtosis - Excess kurtosis of the integrated profile
    4. ip_skewness - Skewness of the integrated profile
    5. dm_mean - Mean of the DM-SNR curve
    6. dm_std - Standard deviation of the DM-SNR curve
    7. dm_kurtosis - Excess kurtosis of the DM-SNR curve
    8. dm_skewness - Skewness of the DM-SNR curve
    """
    try:
        probability = predict_single(payload.features)
        return PredictionResponse(probability=probability, is_pulsar=probability >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch_samples(payload: HTRUInputBatch) -> BatchPredictionResponse:
    """
    Predict multiple samples at once.

    Each sample should contain 8 numerical HTRU2 features in this exact order:
    1. ip_mean, 2. ip_std, 3. ip_kurtosis, 4. ip_skewness,
    5. dm_mean, 6. dm_std, 7. dm_kurtosis, 8. dm_skewness
    """
    try:
        probabilities = predict_batch(payload.samples)
        predictions = [
            PredictionResponse(probability=prob, is_pulsar=prob >= 0.5) for prob in probabilities
        ]
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "HTRU2 Pulsar Classification API", "version": "1.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.get("/features")
def get_feature_names():
    """Endpoint to check what feature names the model expects"""
    return {
        "feature_names": FEATURE_NAMES,
        "descriptions": {
            "ip_mean": "Mean of the integrated profile",
            "ip_std": "Standard deviation of the integrated profile",
            "ip_kurtosis": "Excess kurtosis of the integrated profile",
            "ip_skewness": "Skewness of the integrated profile",
            "dm_mean": "Mean of the DM-SNR curve",
            "dm_std": "Standard deviation of the DM-SNR curve",
            "dm_kurtosis": "Excess kurtosis of the DM-SNR curve",
            "dm_skewness": "Skewness of the DM-SNR curve",
        },
    }


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696)
