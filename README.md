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
  - [Performance Summary](#performance-summary)
  - [Best Model Selection](#best-model-selection)
  - [Feature Importance](#feature-importance)
- [Key Features Implementation](#-key-features-implementation)
- [Model Interpretation](#-model-interpretation)
- [Model Deployment](#-model-deployment)
  - [Quick Deployment](#quick-deployment)
- [API Usage](#-api-usage)
  - [Available Endpoints](#available-endpoints)
  - [Feature Specifications](#feature-specifications)
  - [Making Predictions](#making-predictions)
  - [Python Client Usage](#-python-client-usage)
  - [Interpretation Guide](#interpretation-guide)
- [Technical Details](#-technical-details)
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
  
## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- UV package manager
- Kaggle API credentials

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd pulsar_classification
   ```

2. **Set up Kaggle API**

   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Install dependencies**

   ```bash
   uv sync
   ```

4. **Set up directories**

   ```bash
   uv run python scripts/setup_directories.py
   ```

### Usage

Run the complete pipeline:

```bash
uv run python scripts/main.py
```

## 🏗️ Project Structure {#project-structure}

```markdown
pulsar_classification/
├── notebooks/                 # Jupyter Notebooks
├── scripts/
│   ├── main.py                # Main pipeline execution script
│   ├── config.py              # Configuration management
│   ├── data_handler.py        # Data download, loading, and preprocessing
│   ├── training.py            # Model training and evaluation
│   └── setup_directories.py   # Directory setup utility
├── outputs/
│   ├── models/                # Saved trained models
│   ├── metrics/               # Evaluation metrics and results
│   └── predictions/           # Prediction outputs
├── logs/                      # Execution logs with timestamps
├── data/
│   ├── external/              # Raw downloaded data
│   └── processed/             # Processed and split data
└── model_config.toml          # Hyperparameter configuration
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

## ⚙️ Configuration {#configuration}

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

| Model | ROC-AUC | Recall | F1-Score | Training Time | Best Parameters |
|-------|---------|--------|----------|---------------|-----------------|
| **XGBoost** | **0.9768** | **0.8628** | **0.8927** | ~32 seconds | `learning_rate: 0.05`, `max_depth: 3`, `n_estimators: 200` |
| Logistic Regression | 0.9746 | 0.7774 | 0.8500 | ~2 seconds | `C: 1.0`, `penalty: l2` |
| Random Forest | 0.9738 | 0.8476 | 0.8701 | ~31 seconds | `max_depth: 10`, `min_samples_split: 2`, `n_estimators: 200` |
| Gradient Boosting | 0.9742 | 0.8049 | 0.8656 | ~52 seconds | `learning_rate: 0.05`, `max_depth: 3`, `n_estimators: 200` |

### Best Model Selection

- **Selected Model**: **XGBoost**
- **Selection Criteria**: Highest ROC-AUC score on validation set (0.9779)
- **Final Test Performance**: ROC-AUC 0.9768, Recall 0.8628, F1-Score 0.8927

### Feature Importance

**Top features contributing to pulsar classification:**

1. **ip_kurtosis**: 0.4368 (Integrated Profile Kurtosis)
2. **ip_skewness**: 0.3520 (Integrated Profile Skewness)
3. **dm_std**: 0.0678 (DM-SNR Curve Standard Deviation)
4. **ip_std**: 0.0412 (Integrated Profile Standard Deviation)
5. **ip_mean**: 0.0298 (Integrated Profile Mean)

### Confusion Matrix Analysis (XGBoost - Test Set)

```markdown
[[3229   23]   # True Negatives | False Positives
 [  45  283]]  # False Negatives | True Positives
```

**Performance Analysis:**

- **True Positives**: 283 pulsars correctly identified
- **True Negatives**: 3229 non-pulsars correctly identified  
- **False Positives**: 23 non-pulsars misclassified as pulsars
- **False Negatives**: 45 pulsars missed

**Key Metrics:**

- **Precision**: 92.5%
- **Recall**: 86.3%
- **F1-Score**: 89.3%

## ⚡ Performance Validation

**Dataset Characteristics:**

- **Small Size**: 17,898 samples × 8 features = 143,184 data points total
- **Low Dimensionality**: Only 8 features reduces computational complexity
- **Structured Data**: Clean, numerical data without missing values

**Computational Efficiency:**

- **Parallel Processing**: GridSearchCV used 8-fold CV with full CPU utilization
- **Optimized Libraries**: scikit-learn and XGBoost are highly optimized C++ implementations
- **Simple Models**: No deep learning or complex architectures

### Quality Assurance Steps Taken

1. **Stratified Splitting**: Maintained class distribution (90.84%/9.16%)
2. **Cross-Validation**: 8-fold CV ensures robust hyperparameter tuning
3. **Train-Validation-Test Split**: Proper evaluation protocol
4. **Multiple Algorithms**: Consistent performance across different model types
5. **Feature Importance**: Results align with domain knowledge (IP statistics most important)

## 🎯 Key Features Implementation

### Data Handling

- **Class**: `HTRU2DataHandler`
- **Methods**: Download, load, preprocess, split, export
- **Error Handling**: Missing file detection, data validation

### Model Training

- **Class**: `ModelTrainer`
- **Algorithms**: 4 different classifiers with hyperparameter tuning
- **Evaluation**: Comprehensive metrics and cross-validation

### Configuration Management

- **Pydantic Settings**: Environment variable support
- **TOML Config**: Model hyperparameters and training settings
- **Reproducibility**: Fixed random seeds and version tracking

## 🔍 Model Interpretation

### Business Impact

- **Recall Focus**: The model achieves 86.3% recall, meaning it correctly identifies 86.3% of actual pulsars
- **Precision Consideration**: With 92.5% precision, only 7.5% of predicted pulsars are false positives
- **Scientific Value**: The model significantly reduces manual verification workload while maintaining high detection rates

### Technical Considerations

- **Class Imbalance**: Successfully handled 9:1 class ratio through stratified sampling
- **Feature Scaling**: Standardization improved performance of distance-based algorithms
- **Model Calibration**: XGBoost provided well-calibrated probability estimates

### Performance Insights

1. **XGBoost Dominance**: Outperformed other models in both ROC-AUC and F1-score
2. **Feature Importance**: Integrated profile statistics (kurtosis, skewness) are most predictive
3. **Training Efficiency**: All models trained in under 1 minute total
4. **Cross-Validation**: 8-fold CV provided robust hyperparameter tuning

## 🐳 Model Deployment

### Quick Deployment

#### 1a. Build Docker Image

```bash
docker build -t pulsar-classification-api:latest .
```

#### 1b. Pull Docker Image from GitHub project

In case this method is used, modify the name of the image you are using for the following steps accordingly.

```bash
docker pull ghcr.io/mchadolias/<project-name>:<tag>
```

#### 2. Run Container

```bash
docker run -it -p 9696:9696 pulsar-classification-api:latest
```

#### 3. Verify Health

```bash
curl http://localhost:9696/health
```

**Response:** `{"status":"healthy","model_loaded":true}`

## 🌐 API Usage

![API Documentation](./outputs/screenshot/docker_docs_localhost.png)

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single sample prediction |
| `POST` | `/predict_batch` | Batch samples prediction |
| `GET` | `/` | API information |
| `GET` | `/health` | Service health check |
| `GET` | `/features` | Expected feature names |

![API Endpoints](./outputs/screenshot/docker_docs_localhost.png)

### Feature Specifications

![Feature Names](./outputs/screenshot/docker_get_features_localhost.png)

Get expected features:

```bash
curl -X 'GET' 'http://localhost:9696/features' -H 'accept: application/json'
```

**Response:**

```json
{
  "feature_names": [
    "ip_mean",
    "ip_std", 
    "ip_kurtosis",
    "ip_skewness",
    "dm_mean",
    "dm_std",
    "dm_kurtosis",
    "dm_skewness"
  ],
  "descriptions": {
    "ip_mean": "Mean of the integrated profile",
    "ip_std": "Standard deviation of the integrated profile",
    "ip_kurtosis": "Excess kurtosis of the integrated profile",
    "ip_skewness": "Skewness of the integrated profile",
    "dm_mean": "Mean of the DM-SNR curve",
    "dm_std": "Standard deviation of the DM-SNR curve",
    "dm_kurtosis": "Excess kurtosis of the DM-SNR curve",
    "dm_skewness": "Skewness of the DM-SNR curve"
  }
}
```

### Making Predictions

#### Single Prediction

![Single Prediction](./outputs/screenshot/docker_predict_localhost.png)

**Request:**

```bash
curl -X 'POST' 'http://localhost:9696/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "features": [
      99.3671875,
      41.57220208,
      1.547196967,
      4.154106043,
      27.55518395,
      61.71901588,
      2.20880796,
      3.662680136
    ]
  }'
```

**Response:**

![Single Result](./outputs/screenshot/client_single_classification.png)

```json
{
  "probability": 0.96047443151474,
  "is_pulsar": true
}
```

#### Batch Prediction

**Request:**

```bash
curl -X 'POST' 'http://localhost:9696/predict_batch' \
  -H 'Content-Type: application/json' \
  -d '{
    "samples": [
      [99.3671875, 41.57220208, 1.547196967, 4.154106043, 27.55518395, 61.71901588, 2.20880796, 3.662680136],
      [140.0, 45.0, 1.8, 3.9, 25.0, 60.0, 2.1, 3.5]
    ]
  }'
```

**Response:**

![Batch Result](./outputs/screenshot/client_batch_classification.png)

```json
{
  "predictions": [
    {
      "probability": 0.96047443151474,
      "is_pulsar": true
    },
    {
      "probability": 0.9847214818000793,
      "is_pulsar": true
    }
  ]
}
```

## 🐍 Python Client Usage

#### Single Prediction

```python
python client.py
```

#### Batch Prediction  

```python
python client.py --batch
```

### Interpretation Guide

#### Probability Thresholds

- **≥ 0.5**: Classified as pulsar (`"is_pulsar": true`)
- **< 0.5**: Classified as non-pulsar (`"is_pulsar": false`)

#### Confidence Levels

- **0.9-1.0**: High confidence pulsar
- **0.7-0.9**: Moderate confidence pulsar  
- **0.5-0.7**: Low confidence pulsar
- **0.3-0.5**: Possible non-pulsar
- **0.0-0.3**: High confidence non-pulsar

## 🔧 Technical Details

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

1. [HTRU2 Dataset Repository (UC Irvine)](https://archive.ics.uci.edu/dataset/372/htru2)
2. [Lyon, R. J., Stappers, B. W., et al. - Pulsar Classification](https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.1104L)
3. [Kaggle Pulsar Dataset](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

*Last Updated: 2025-11-19*  
*Last Pipeline Execution: 2025-11-18 01:33:06*  
*Best Model: XGBoost (ROC-AUC: 0.9768)*
