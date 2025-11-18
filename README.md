# 🌌 Pulsar Star Classification Project

## 📋 Project Overview

This project focuses on classifying pulsar stars using the HTRU2 dataset from Kaggle. Pulsars are rare and valuable astronomical objects, and accurate classification is crucial for astronomical research. The project implements a complete machine learning pipeline from data acquisition to model deployment.

### 🎯 Business Problem
Pulsars are rare neutron stars that produce valuable scientific data. Manual classification is time-consuming and prone to error. This project aims to automate pulsar classification using machine learning to assist astronomers in identifying genuine pulsar signals from noise.

### 📊 Dataset
- **Source**: [HTRU2 Dataset from Kaggle](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2)
- **Samples**: 17,898 total instances
- **Features**: 8 numerical features derived from pulsar candidate profiles
- **Target**: Binary classification (0 = non-pulsar, 1 = pulsar)
- **Class Distribution**: Highly imbalanced (~90% non-pulsars, ~10% pulsars)

## 🏗️ Project Structure

```
pulsar_classification/
├── scripts/
│   ├── main.py                 # Main pipeline execution script
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
├── config/
│   └── training_config.toml   # Model configuration
└── model_config.toml          # Hyperparameter configuration
```

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

## ⚙️ Configuration

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

## 📈 Model Performance

<!-- This section will be filled with actual performance metrics after model training -->

### Performance Summary
*Results will be updated after pipeline execution*

| Model | ROC-AUC | Recall | F1-Score | Precision | Training Time |
|-------|---------|--------|----------|-----------|---------------|
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| Gradient Boosting | - | - | - | - | - |
| XGBoost | - | - | - | - | - |

### Best Model Selection
- **Selected Model**: *To be determined*
- **Selection Criteria**: ROC-AUC score on validation set
- **Final Test Performance**: *To be updated*

### Feature Importance
*Top features contributing to pulsar classification:*
1. *Feature importance analysis pending model training*

### Confusion Matrix Analysis
*Detailed analysis of model performance across classes*

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

## 📊 Dataset Features

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

## 🔍 Model Interpretation

### Business Impact
- **Recall Focus**: Minimizing false negatives (missed pulsars) is critical
- **Precision Consideration**: Balancing scientific discovery with manual verification workload

### Technical Considerations
- **Class Imbalance**: Techniques to handle 9:1 class ratio
- **Feature Scaling**: Standardization for distance-based algorithms
- **Model Calibration**: Probability calibration for reliable confidence scores


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 References

1. [HTRU2 Dataset Repository (UC Irvine)](https://archive.ics.uci.edu/dataset/372/htru2)
2. [Lyon, R. J., Stappers, B. W., et al. - Pulsar Classification](https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.1104L)
3. [Kaggle Pulsar Dataset](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2)

## 🆘 Troubleshooting

### Common Issues

1. **Kaggle API Errors**
   - Verify `kaggle.json` credentials
   - Check internet connection
   - Ensure dataset is publicly accessible

2. **Dependency Conflicts**
   - Use UV for isolated environments
   - Check Python version compatibility
   - Review dependency versions in `pyproject.toml`


---

*Last Updated: 18-10-2025*