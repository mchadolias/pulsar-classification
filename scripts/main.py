# scripts/main.py
import os
import pandas as pd
import json
import pickle
import numpy as np
import logging
from datetime import datetime
from config import DataConfig
from data_handler import HTRU2DataHandler
from training import ModelTrainer


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def setup_logger():
    """Set up logging to both console and file."""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_filename = f"logs/pulsar_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_filename


def log_section_header(logger, title):
    """Log a formatted section header."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f" {title}")
    logger.info("=" * 60)


def main():
    """Main script to run the complete ML pipeline using existing TOML config."""

    # Setup logger
    logger, log_filename = setup_logger()

    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)

    logger.info("🚀 Starting HTRU2 Pulsar Classification Pipeline...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Log file: {log_filename}")

    try:
        # Load data configuration
        data_config = DataConfig()
        logger.info("✅ Data configuration loaded")

        # Initialize data handler
        data_handler = HTRU2DataHandler(data_config, logger=logger)

        # Step 1: Download data (if needed)
        log_section_header(logger, "STEP 1: DOWNLOADING DATA")
        data_handler.download_kaggle()

        # Step 2: Load data
        log_section_header(logger, "STEP 2: LOADING DATA")
        df = data_handler.load()
        logger.info(f"📊 Dataset shape: {df.shape}")

        # Step 3: Preprocess data
        log_section_header(logger, "STEP 3: PREPROCESSING DATA")
        df_processed = data_handler.preprocess()
        logger.info(f"✅ Processed dataset shape: {df_processed.shape}")

        # Check data balance
        check_data_balance(df_processed, logger=logger)

        # Step 4: Split data
        log_section_header(logger, "STEP 4: SPLITTING DATA")
        splits = data_handler.split_train_val_test()

        # Export data splits
        logger.info("💾 Exporting data splits...")
        data_handler.export_splits()

        # Prepare features and target
        log_section_header(logger, "STEP 5: PREPARING FEATURES AND TARGET")
        target_col = "signal"
        feature_cols = [col for col in df_processed.columns if col != target_col]

        X_train = splits["train"][feature_cols]
        y_train = splits["train"][target_col]
        X_val = splits["val"][feature_cols]
        y_val = splits["val"][target_col]
        X_test = splits["test"][feature_cols]
        y_test = splits["test"][target_col]

        logger.info(f"Training set: {X_train.shape}, {y_train.shape}")
        logger.info(f"Validation set: {X_val.shape}, {y_val.shape}")
        logger.info(f"Test set: {X_test.shape}, {y_test.shape}")

        # Step 6: Initialize model trainer
        log_section_header(logger, "STEP 6: INITIALIZING MODEL TRAINER")

        # Use your existing TOML config file
        config_path = "model_config.toml"

        if not os.path.exists(config_path):
            config_path = "../model_config.toml"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        trainer = ModelTrainer(config_path, logger=logger)
        logger.info(f"✅ Model trainer initialized with config: {config_path}")

        # Add numerical features to trainer's data config
        trainer.data_cfg["numerical_features"] = feature_cols
        logger.info(f"📊 Using numerical features: {feature_cols}")

        # Step 7: Train models
        log_section_header(logger, "STEP 7: TRAINING MODELS")

        models_to_train = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]

        best_metrics = {}
        best_model_name = None
        best_score = 0
        all_results = {}

        for model_name in models_to_train:
            log_section_header(logger, f"TRAINING {model_name.upper()}")

            try:
                # Train the model
                trainer.train(X_train, y_train, model_name)

                # Evaluate on validation set
                metrics = trainer.evaluate(X_val, y_val)

                logger.info(f"📊 {model_name} Validation Metrics:")
                for metric_name, value in metrics.items():
                    if metric_name != "confusion_matrix":
                        logger.info(f"  {metric_name}: {value:.4f}")
                    else:
                        logger.info(f"  {metric_name}:")
                        logger.info(f"    {value}")

                # Store results
                all_results[model_name] = metrics

                # Track best model based on ROC AUC
                roc_auc = metrics["roc_auc"]
                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model_name = model_name
                    best_metrics = metrics

            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {e}")
                continue

        # Save model comparison results
        save_model_comparison(all_results, best_model_name, best_score, logger)

        # Display comparison of all models
        log_section_header(logger, "MODEL COMPARISON SUMMARY")
        for model_name, metrics in all_results.items():
            roc_auc = metrics["roc_auc"]
            recall = metrics["recall"]
            f1 = metrics["f1_score"]
            logger.info(
                f"{model_name:20} | ROC AUC: {roc_auc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
            )

        logger.info(f"🏆 Best model: {best_model_name} (ROC AUC: {best_score:.4f})")

        # Step 8: Final evaluation with best model
        logger.info(
            f"🔄 Retraining best model ({best_model_name}) on combined training + validation data..."
        )

        # Combine train and validation sets
        X_full_train = pd.concat([X_train, X_val], axis=0)
        y_full_train = pd.concat([y_train, y_val], axis=0)

        # Retrain best model on full training data
        trainer.train(X_full_train, y_full_train, best_model_name)

        # Evaluate on test set
        log_section_header(logger, "FINAL EVALUATION ON TEST SET")
        test_metrics = trainer.evaluate(X_test, y_test)

        logger.info("📊 Test Set Metrics:")
        for metric_name, value in test_metrics.items():
            if metric_name != "confusion_matrix":
                logger.info(f"  {metric_name}: {value:.4f}")
            else:
                logger.info(f"  {metric_name}:")
                logger.info(f"    {value}")

        # Save test predictions
        save_test_predictions(trainer, X_test, y_test, best_model_name, logger)

        # Step 9: Feature importance (if available)
        feature_importances = {}
        try:
            logger.info("🔍 Feature Importances:")
            importances = trainer.get_feature_importances(feature_cols)
            for feature, importance in sorted(
                importances.items(), key=lambda x: x[1], reverse=True
            ):
                logger.info(f"  {feature}: {importance:.4f}")
            feature_importances = importances
        except ValueError as e:
            logger.info(f"  Note: {e}")
            feature_importances = {"error": str(e)}

        # Step 10: Save the model and all outputs
        log_section_header(logger, "STEP 10: SAVING MODEL AND OUTPUTS")

        # Save the trained model
        model_save_path = f"outputs/models/best_{best_model_name}_model.pkl"
        trainer.save_model(model_save_path)

        # Save comprehensive results
        save_final_results(
            best_model_name=best_model_name,
            test_metrics=test_metrics,
            feature_importances=feature_importances,
            feature_cols=feature_cols,
            model_config_path=config_path,
            logger=logger,
        )

        # Save training history
        save_training_history(all_results, best_model_name, data_handler, logger)

        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"📁 Best model saved as: {model_save_path}")
        logger.info(f"🎯 Best model: {best_model_name}")
        logger.info(f"📊 Test ROC AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"📂 All outputs saved in: outputs/")
        logger.info(f"📝 Log file: {log_filename}")

    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        logger.error("Stack trace:", exc_info=True)
        raise


def save_model_comparison(all_results, best_model_name, best_score, logger):
    """Save model comparison results to JSON and CSV."""
    # Convert metrics to serializable format
    serializable_results = {}
    for model_name, metrics in all_results.items():
        serializable_results[model_name] = {
            "roc_auc": float(metrics["roc_auc"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
            "confusion_matrix": (
                metrics["confusion_matrix"].tolist()
                if hasattr(metrics["confusion_matrix"], "tolist")
                else metrics["confusion_matrix"]
            ),
        }

    # Save to JSON
    comparison_data = {
        "best_model": best_model_name,
        "best_score": float(best_score),
        "model_comparison": serializable_results,
        "timestamp": datetime.now().isoformat(),
    }

    with open("outputs/metrics/model_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2, cls=NumpyEncoder)

    # Save to CSV
    comparison_df = pd.DataFrame(
        [
            {
                "model": model,
                "roc_auc": float(metrics["roc_auc"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1_score"]),
                "is_best": model == best_model_name,
            }
            for model, metrics in all_results.items()
        ]
    )
    comparison_df.to_csv("outputs/metrics/model_comparison.csv", index=False)

    logger.info("✅ Model comparison saved to outputs/metrics/")


def save_test_predictions(trainer, X_test, y_test, best_model_name, logger):
    """Save test set predictions."""
    # Get predictions
    y_pred = trainer.predict(X_test)
    y_pred_proba = trainer.predict_proba(X_test)[:, 1]

    # Create predictions DataFrame
    predictions_df = X_test.copy()
    predictions_df["true_label"] = y_test.values
    predictions_df["predicted_label"] = y_pred
    predictions_df["predicted_probability"] = y_pred_proba

    # Save predictions
    predictions_df.to_csv(
        f"outputs/predictions/{best_model_name}_test_predictions.csv", index=False
    )

    # Save prediction probabilities only
    proba_df = pd.DataFrame({"true_label": y_test.values, "predicted_probability": y_pred_proba})
    proba_df.to_csv(
        f"outputs/predictions/{best_model_name}_prediction_probabilities.csv", index=False
    )

    logger.info("✅ Test predictions saved to outputs/predictions/")


def save_final_results(
    best_model_name, test_metrics, feature_importances, feature_cols, model_config_path, logger
):
    """Save final results and model information."""
    # Convert test metrics to serializable format
    serializable_test_metrics = {
        "roc_auc": float(test_metrics["roc_auc"]),
        "recall": float(test_metrics["recall"]),
        "f1_score": float(test_metrics["f1_score"]),
        "confusion_matrix": (
            test_metrics["confusion_matrix"].tolist()
            if hasattr(test_metrics["confusion_matrix"], "tolist")
            else test_metrics["confusion_matrix"]
        ),
    }

    # Convert feature importances to serializable format
    serializable_feature_importances = {}
    if feature_importances and "error" not in feature_importances:
        for feature, importance in feature_importances.items():
            serializable_feature_importances[feature] = float(importance)
    else:
        serializable_feature_importances = feature_importances

    results = {
        "best_model": best_model_name,
        "test_metrics": serializable_test_metrics,
        "feature_importances": serializable_feature_importances,
        "features_used": feature_cols,
        "model_config": model_config_path,
        "timestamp": datetime.now().isoformat(),
        "training_completed": True,
    }

    with open("outputs/metrics/final_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Save feature importances separately
    if serializable_feature_importances and "error" not in serializable_feature_importances:
        feature_importance_df = pd.DataFrame(
            [
                {"feature": feature, "importance": importance}
                for feature, importance in serializable_feature_importances.items()
            ]
        ).sort_values("importance", ascending=False)

        feature_importance_df.to_csv("outputs/metrics/feature_importances.csv", index=False)

    logger.info("✅ Final results saved to outputs/metrics/")


def save_training_history(all_results, best_model_name, data_handler, logger):
    """Save training history and dataset information."""
    # Convert model performance to serializable format
    serializable_performance = {}
    for model, metrics in all_results.items():
        serializable_performance[model] = {
            "roc_auc": float(metrics["roc_auc"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
        }

    # Convert dataset info to serializable format
    if data_handler.df is not None:
        class_balance = {
            int(k): int(v) for k, v in data_handler.df["signal"].value_counts().to_dict().items()
        }
        dataset_info = {
            "total_samples": int(len(data_handler.df)),
            "features_count": int(len(data_handler.df.columns) - 1),
            "target_variable": "signal",
            "class_balance": class_balance,
        }
    else:
        dataset_info = {
            "total_samples": 0,
            "features_count": 0,
            "target_variable": "signal",
            "class_balance": {},
        }

    training_history = {
        "best_model": best_model_name,
        "all_models_trained": list(all_results.keys()),
        "dataset_info": dataset_info,
        "training_date": datetime.now().isoformat(),
        "model_performance": serializable_performance,
    }

    with open("outputs/metrics/training_history.json", "w") as f:
        json.dump(training_history, f, indent=2, cls=NumpyEncoder)

    logger.info("✅ Training history saved")


def check_data_balance(df, target_col="signal", logger=None):
    """Check the balance of the target variable."""
    # If no logger provided, use the default logger
    if logger is None:
        logger = logging.getLogger()

    logger.info("📊 Data Balance Check:")

    try:
        value_counts = df[target_col].value_counts()
        percentages = df[target_col].value_counts(normalize=True) * 100

        for value, count in value_counts.items():
            percentage = percentages[value]
            logger.info(f"  Class {value}: {count} samples ({percentage:.2f}%)")

        # Save data balance info - convert numpy types to Python native types
        balance_info = {
            "class_distribution": {int(k): int(v) for k, v in value_counts.to_dict().items()},
            "class_percentages": {int(k): float(v) for k, v in percentages.to_dict().items()},
            "total_samples": int(len(df)),
            "is_imbalanced": bool(abs(percentages.iloc[0] - percentages.iloc[1]) > 20),
        }

        os.makedirs("outputs/metrics", exist_ok=True)
        with open("outputs/metrics/data_balance.json", "w") as f:
            json.dump(balance_info, f, indent=2, cls=NumpyEncoder)

    except KeyError as e:
        logger.error(
            f"❌ Target column '{target_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )
        raise
    except Exception as e:
        logger.error(f"❌ Error checking data balance: {e}")
        raise


if __name__ == "__main__":
    main()
