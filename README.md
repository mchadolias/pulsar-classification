# 🌌 Pulsar Star Classification Project

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
  - [Business Problem](#-business-problem)
  - [Dataset](#-dataset)
  - [Dataset Features](#dataset-features)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#-features)
- [Configuration](#configuration)
- [Model Performance](#-model-performance)
- [Model Deployment](#-model-deployment)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)
- [License](#-license)
- [Contributing](#-contributing)

---

## 📋 Project Overview

This project focuses on classifying pulsar stars using the HTRU2 dataset from Kaggle. Pulsars are rare and valuable astronomical objects, and accurate classification is crucial for astronomical research. The project implements a complete machine learning pipeline from data acquisition to model deployment.

### 🎯 Business Problem

Pulsars are rare neutron stars that produce valuable scientific data. Manual classification is time-consuming and prone to error. This project aims to automate pulsar classification using machine learning to assist astronomers in identifying genuine pulsar signals from noise.

### 📊 Dataset

- **Source**: [HTRU2 Dataset from Kaggle](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2)
- **Samples**: 17,898 total instances
- **Features**: 8 numerical features derived from pulsar candidate profiles
- **Target**: Binary classification (0 = non-pulsar, 1 = pulsar)
- **Class Distribution**: Highly imbalanced (~90.84% non-pulsars, ~9.16% pulsars)

### Dataset Features

The HTRU2 dataset contains 8 features derived from the integrated pulse profile and DM-SNR curve:

1. **Integrated Profile Features**:
   - `ip_mean`: Mean of the integrated profile
   - `ip_std`: Standard deviation of the integrated profile
   - `ip_kurtosis`: Kurtosis of the integrated profile
   - `ip_skewness`: Skewness of the integrated profile

2. **DM-SNR Curve Features**:
   - `dm_mean`: Mean of the DM-SNR curve
   - `dm_std`: Standard deviation of the DM-SNR curve
   - `dm_kurtosis`: Kurtosis of the DM-SNR curve
   - `dm_skewness`: Skewness of the DM-SNR curve

3. **Target**:
   - `signal`: Class label (0: noise, 1: pulsar)

![](outputs/screenshot/heatmap.png)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- UV package manager
- Kaggle API credentials

### Installation

1. **Clone the repository**

   ```bash
   git clone git@github.com:mchadolias/pulsar-classification.git
   cd pulsar-classification
   ```

2. **Set up Kaggle API**

   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Install dependencies**

   ```bash
    # Install only core dependencies
    uv sync

    # Install with dev dependencies
    uv sync --extra dev

    # Install with training dependencies  
    uv sync --extra training

    # Install with both dev and training (all dependecies)
    uv sync --extra dev --extra training
   ```

   **Note**: This is a new addition to have optional dependencies depending on the usage of the experiment. For new users, it is recommended to use all dependecies at once not to have issues with reproducibility.

4. **Set up directories**

   ```bash
   uv run python scripts/setup_directories.py
   ```

### Usage

Run the complete pipeline:

```bash
uv run python scripts/main.py
```

## Project Structure

```markdown
pulsar-classification/
├── notebooks/
|   ├── 01_data_exploration.ipynb  # Initial EDA notebook
|   └── 02_kaggle_submission.ipynb # Kaggle posted notebook to dataset
├── docs/
│   ├── DEPLOYMENT.md              # Deployment instructions
│   └── MODEL_PERFOMANCE.md        # Model perfomance report
├── deployment/
│   ├── client.py                  # API request script for the user
│   ├── predict.py                 # FastAPI app initialization 
│   └── examples/
│       ├── test_cases.json        # Prediction cases for all options
│       ├── single_prediction.json # Single prediction case
│       └── batch_prediction.json  # Batch prediction case
├── scripts/
│   ├── main.py                    # Main pipeline execution script
│   ├── config.py                  # Configuration management
│   ├── data_handler.py            # Data download, loading, and preprocessing
│   ├── training.py                # Model training and evaluation
│   └── setup_directories.py       # Directory setup utility
├── outputs/
│   ├── screenshots/               # Saved demo screenshots
│   ├── models/                    # Saved trained models
│   ├── metrics/                   # Evaluation metrics and results
│   └── predictions/               # Prediction outputs
├── logs/                          # Execution logs with timestamps
├── data/
│   ├── external/                  # Raw downloaded data
│   └── processed/                 # Processed and split data
├── model_config.toml              # Hyperparameter configuration
├── .dockerignore
├── .python-version
├── requirements.txt
├── uv.lock
├── setup.cfg
├── README.md
├── LICENSE
└── Dockerfile
```

## 🔧 Features

### Data Pipeline

- **Automated Download**: Fetches dataset from Kaggle API
- **Data Validation**: Checks for missing values and data quality
- **Preprocessing**: Column renaming, numerical rounding, standardization
- **Stratified Splitting**: Maintains class distribution across splits

### Model Training

- **Multiple Algorithms**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Automated Evaluation**: ROC-AUC, Recall, F1-Score, Confusion Matrix
- **Feature Importance**: Model interpretability analysis

### Logging & Monitoring

- **Comprehensive Logging**: All operations logged to file and terminal
- **Timestamped Logs**: Automatic log file creation with timestamps
- **Error Handling**: Detailed error messages with stack traces
- **Progress Tracking**: Real-time progress updates

### Output Management

- **Model Persistence**: Saved models in pickle format
- **Metrics Export**: JSON and CSV formats for all evaluation metrics
- **Prediction Storage**: Test predictions with probabilities
- **Configuration Backup**: Training configuration preserved

## Configuration

### Model Configuration (`model_config.toml`)

The project uses TOML configuration for model hyperparameters:

```toml
[logistic_regression]
max_iter = 500
solver = "lbfgs"
penalty = "l2"

[random_forest]
n_estimators = 200
max_depth = 10
class_weight = "balanced"

# ... additional model configurations
```

### Data Configuration (`scripts/config.py`)

- Data directory structure
- Split ratios (train/validation/test)
- Random seeds for reproducibility

## 📈 Model Performance

### Performance Summary

**Best Model: XGBoost**

- **ROC-AUC**: 0.9768
- **Recall**: 0.8628 (86.3% of pulsars correctly identified)
- **F1-Score**: 0.8927
- **Precision**: 0.925 (92.5% of predicted pulsars are correct)

### Key Metrics (Test Set)

- **True Positives**: 283 pulsars correctly identified
- **True Negatives**: 3229 non-pulsars correctly identified
- **False Positives**: 23 non-pulsars misclassified
- **False Negatives**: 45 pulsars missed

### Top Features

1. **ip_kurtosis** (43.7% importance) - Integrated Profile Kurtosis
2. **ip_skewness** (35.2% importance) - Integrated Profile Skewness
3. **dm_std** (6.8% importance) - DM-SNR Curve Standard Deviation

**Full Performance Report**: See detailed analysis in [MODEL_PERFORMANCE.md](docs/MODEL_PERFOMANCE.md)

## 🐳 Model Deployment

### Quick API Deployment

```bash
# Build and run Docker container
docker build -t pulsar-classification-api:latest .
docker run -it -p 9696:9696 pulsar-classification-api:latest
```

### API Endpoints

- `POST /predict` - Single sample prediction
- `POST /predict_batch` - Batch prediction
- `GET /health` - Service health check
- `GET /features` - Expected feature names

### Deployment Options

- **Docker**: Local container deployment
- **Fly.io**: Cloud deployment (demonstrated)
- **Hugging Face Spaces**: [mchadolias/pulsar-classification-htru2](https://huggingface.co/spaces/mchadolias/pulsar-classification-htru2/)

**Full Deployment Guide**: See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions.

## 🔧 Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   docker run -it -p 9698:9696 pulsar-classification-api:latest
   ```

2. **Model file not found**
   - Ensure `best_xgboost_model.pkl` exists in `outputs/models/`

3. **Feature validation errors**
   - Verify exactly 8 features are provided
   - Check feature order matches `/features` endpoint

4. **Docker build failures**
   - Keep in mind that building issues have been observed when using a VPN, so rebuild it after you are disconnected.

   ```bash
   docker build --no-cache -t pulsar-classification-api:latest .
   ```

5. **Kaggle API Errors**
   - Verify `kaggle.json` credentials
   - Check internet connection
   - Ensure dataset is publicly accessible

6. **Dependency Conflicts**
   - Use UV for isolated environments
   - Check Python version compatibility
   - Review dependency versions in `pyproject.toml`

### Monitoring

```bash
# Check container status
docker ps

# View logs
docker logs <container_id>

# Health check
curl http://localhost:9696/health
```

### Performance Tips

- The pipeline is optimized for efficiency, completing in under 3 minutes
- For larger datasets, consider reducing cross-validation folds
- Model training can be parallelized by adjusting `n_jobs` parameter

## 📚 References

### Academic & Dataset References

1. [HTRU2 Dataset Repository (UC Irvine)](https://archive.ics.uci.edu/dataset/372/htru2)
2. [Lyon, R. J., Stappers, B. W., et al. - Pulsar Classification](https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.1104L)
3. [Kaggle Pulsar Dataset](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2)

### Key Library Documentation

4. [FastAPI Documentation](https://fastapi.tiangolo.com/) - Modern web framework for building APIs
5. [XGBoost Documentation](https://xgboost.readthedocs.io/) - Scalable and accurate gradient boosting
6. [Scikit-learn Documentation](https://scikit-learn.org/stable/) - Machine learning in Python
7. [Pydantic Documentation](https://docs.pydantic.dev/) - Data validation using Python type annotations
8. [UV Documentation](https://docs.astral.sh/uv/) - Fast Python package and project manager
9. [Docker Documentation](https://docs.docker.com/) - Containerization platform
10. [Uvicorn Documentation](https://www.uvicorn.org/) - ASGI web server implementation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

*Last Updated: 2025-11-26*  
*Last Pipeline Execution: 2025-11-18 01:33:06*  
*Best Model: XGBoost (ROC-AUC: 0.9768)*
