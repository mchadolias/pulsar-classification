from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import pickle
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime

MODEL_PATH = "models/best_xgboost_model.pkl"

app = FastAPI(
    title="🌌 Pulsar Star Classification API",
    description="""
    ## Pulsar Detection Machine Learning API
    
    🔭 **Identify pulsar stars from radio telescope data**
    
    ### Model Performance:
    - **ROC-AUC**: 0.9768 - Excellent discrimination
    - **Recall**: 86.3% - Detects 86.3% of actual pulsars
    - **F1-Score**: 89.3% - Balanced performance metric
    - **Precision**: 92.5% - 92.5% of predicted pulsars are real
    
    ### Dataset Characteristics:
    - **Class Distribution**: 90.8% non-pulsars / 9.2% pulsars
    - **Best Model**: XGBoost Classifier
    - **Training Data**: HTRU2 Pulsar Dataset (17,898 samples)
    """,
    version="1.1.0",
    contact={
        "name": "Michael Chadolias",
        "url": "https://github.com/mchadolias/pulsar_classification",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

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


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "performance": {
            "roc_auc": 0.9768,
            "recall": 0.8628,
            "f1_score": 0.8927,
            "precision": 0.925,  # Calculated from your confusion matrix
        },
        "dataset_info": {
            "total_samples": 17898,
            "class_distribution": {"non_pulsars": 0.9084, "pulsars": 0.0916},
            "imbalance_ratio": "9.9:1",
        },
    }


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


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌌 Pulsar Classification API</title>
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                color: white;
            }
            .header { 
                text-align: center; 
                padding: 40px 0; 
                background: rgba(255,255,255,0.1); 
                border-radius: 15px;
                margin-bottom: 30px;
            }
            .card { 
                background: rgba(255,255,255,0.1); 
                padding: 20px; 
                margin: 15px 0; 
                border-radius: 10px; 
                border-left: 4px solid #4CAF50;
            }
            .endpoint { 
                background: rgba(0,0,0,0.3); 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 8px; 
                font-family: monospace;
            }
            .btn { 
                display: inline-block; 
                padding: 10px 20px; 
                margin: 5px; 
                background: #4CAF50; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
            }
            .metrics { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 15px; 
                margin: 20px 0; 
            }
            .metric-card { 
                background: rgba(255,255,255,0.15); 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌌 Pulsar Star Classification API</h1>
            <p>Machine Learning API for detecting pulsar stars with >92% precision!</p>
        </div>
        
        <div class="card">
            <h2>🚀 Quick Start</h2>
            <p>Test the API with these endpoints:</p>
            <a class="btn" href="/docs">API Documentation</a>
            <a class="btn" href="/redoc">Alternative Docs</a>
            <a class="btn" href="/health">Health Check</a>
        </div>
        
        <div class="card">
            <h2>📊 Model Performance</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>ROC-AUC</h3>
                    <p>0.9768</p>
                    <small>Area Under Curve</small>
                </div>
                <div class="metric-card">
                    <h3>Recall</h3>
                    <p>86.3%</p>
                    <small>Pulsars Detected</small>
                </div>
                <div class="metric-card">
                    <h3>F1-Score</h3>
                    <p>89.3%</p>
                    <small>Balance Measure</small>
                </div>
                <div class="metric-card">
                    <h3>Precision</h3>
                    <p>92.5%</p>
                    <small>Correct Pulsar IDs</small>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🔌 API Endpoints</h2>
            <div class="endpoint">
                <strong>POST</strong> /predict - Single prediction
            </div>
            <div class="endpoint">
                <strong>POST</strong> /predict_batch - Batch predictions
            </div>
            <div class="endpoint">
                <strong>GET</strong> /features - Feature specifications
            </div>
            <div class="endpoint">
                <strong>GET</strong> /health - Service status
            </div>
        </div>
        
        <div class="card">
            <h2>🔬 Technical Details</h2>
            <p><strong>Model:</strong> XGBoost Classifier</p>
            <p><strong>Dataset:</strong> HTRU2 Pulsar Dataset (17,898 samples)</p>
            <p><strong>Features:</strong> 8 radio telescope measurements</p>
            <p><strong>Deployment:</strong> Docker + FastAPI</p>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696)
