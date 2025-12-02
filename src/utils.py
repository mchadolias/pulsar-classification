from pathlib import Path
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional


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


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (Path,)):
            return str(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): self.default(v) for k, v in obj.items()}
        return super(NumpyEncoder, self).default(obj)


def save_model_comparison(all_results, best_model_name, best_score, logger):
    """Save model comparison results with enhanced metrics structure."""
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "best_model": best_model_name,
        "best_score": float(best_score) if best_score is not None else 0.0,
        "all_models": {},
    }

    for model_name, metrics in all_results.items():
        # Get optimal metrics (they're nested under optimal_threshold_X.XXX)
        optimal_key = None
        for key in metrics.keys():
            if key.startswith("optimal_threshold_"):
                optimal_key = key
                break

        if optimal_key and optimal_key in metrics:
            opt_metrics = metrics[optimal_key]
            comparison["all_models"][model_name] = {
                "optimal_threshold": float(metrics.get("optimal_threshold", 0.5)),
                "f1": float(opt_metrics.get("f1", 0)),
                "recall": float(opt_metrics.get("recall", 0)),
                "precision": float(opt_metrics.get("precision", 0)),
                "roc_auc": float(opt_metrics.get("roc_auc", 0)),
                "f2": float(opt_metrics.get("f2", 0)),
            }
        else:
            # Fallback to flat structure if available
            comparison["all_models"][model_name] = {
                "optimal_threshold": float(metrics.get("optimal_threshold", 0.5)),
                "f1": float(metrics.get("f1", metrics.get("f1_score", 0))),
                "recall": float(metrics.get("recall", 0)),
                "precision": float(metrics.get("precision", 0)),
                "roc_auc": float(metrics.get("roc_auc", 0)),
                "f2": float(metrics.get("f2", 0)),
            }

    with open(metrics_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, cls=NumpyEncoder)

    logger.info(f"[SUCCESS] Model comparison saved to {metrics_dir}/model_comparison.json")


def save_test_predictions(trainer, X_test, y_test, best_model_name, logger):
    """Save test set predictions."""
    # Create predictions directory
    predictions_dir = Path("outputs/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get predictions
        y_pred = trainer.predict(X_test)
        y_pred_proba = trainer.predict_proba(X_test)[:, 1]

        # Create predictions DataFrame
        predictions_df = X_test.copy() if hasattr(X_test, "copy") else pd.DataFrame(X_test)
        predictions_df["true_label"] = y_test.values if hasattr(y_test, "values") else y_test
        predictions_df["predicted_label"] = y_pred
        predictions_df["predicted_probability"] = y_pred_proba

        # Save predictions
        predictions_file = predictions_dir / f"{best_model_name}_test_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)

        # Save prediction probabilities only
        proba_df = pd.DataFrame(
            {
                "true_label": y_test.values if hasattr(y_test, "values") else y_test,
                "predicted_probability": y_pred_proba,
            }
        )
        proba_file = predictions_dir / f"{best_model_name}_prediction_probabilities.csv"
        proba_df.to_csv(proba_file, index=False)

        logger.info(f"[SUCCESS] Test predictions saved to {predictions_dir}/")

    except Exception as e:
        logger.error(f"Failed to save test predictions: {e}")


def save_final_results(
    best_model_name: str,
    test_metrics: Dict[str, Any],
    feature_importances: Dict[str, float],
    feature_cols: list,
    model_config_path: str,
    logger: logging.Logger,
) -> None:
    """Save comprehensive final results."""
    results_dir = Path("outputs/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics from nested structure
    optimal_metrics = {}
    default_metrics = {}

    # Find optimal threshold metrics
    for key in test_metrics.keys():
        if key.startswith("optimal_threshold_"):
            optimal_metrics = test_metrics[key]
            break
        elif key == "default_threshold_0.5":
            default_metrics = test_metrics[key]

    # If no optimal metrics found, use flat structure
    if not optimal_metrics:
        optimal_metrics = {
            "roc_auc": float(test_metrics.get("roc_auc", 0)),
            "f1": float(test_metrics.get("f1", test_metrics.get("f1_score", 0))),
            "recall": float(test_metrics.get("recall", 0)),
            "precision": float(test_metrics.get("precision", 0)),
            "f2": float(test_metrics.get("f2", 0)),
        }

    final_results = {
        "best_model": best_model_name,
        "optimal_threshold": float(test_metrics.get("optimal_threshold", 0.5)),
        "test_metrics": {
            "optimal": optimal_metrics,
            "default": default_metrics,
        },
        "feature_importances": {k: float(v) for k, v in feature_importances.items()},
        "features_used": feature_cols,
        "config_file": str(model_config_path),
        "timestamp": datetime.now().isoformat(),
    }

    # Log summary
    logger.info("Final Test Metrics Summary:")
    logger.info(f"  Optimal threshold: {final_results['optimal_threshold']:.3f}")
    if optimal_metrics:
        logger.info(f"  F1 Score: {optimal_metrics.get('f1', 0):.4f}")
        logger.info(f"  Recall: {optimal_metrics.get('recall', 0):.4f}")
        logger.info(f"  Precision: {optimal_metrics.get('precision', 0):.4f}")
        logger.info(f"  ROC AUC: {optimal_metrics.get('roc_auc', 0):.4f}")

    results_file = results_dir / "final_results.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Final results saved to {results_file}")


def save_training_history(
    all_results: Dict[str, Dict], best_model_name: str, data_handler, logger: logging.Logger
) -> None:
    """Save training history and metadata."""
    history_dir = Path("outputs/metrics")
    history_dir.mkdir(parents=True, exist_ok=True)

    processed_results = {}
    for model_name, metrics in all_results.items():
        processed_metrics = {"optimal_threshold": float(metrics.get("optimal_threshold", 0.5))}

        # Extract optimal metrics
        for key in metrics.keys():
            if key.startswith("optimal_threshold_"):
                opt_metrics = metrics[key]
                processed_metrics["optimal"] = {
                    "roc_auc": float(opt_metrics.get("roc_auc", 0)),
                    "f1": float(opt_metrics.get("f1", 0)),
                    "recall": float(opt_metrics.get("recall", 0)),
                    "precision": float(opt_metrics.get("precision", 0)),
                    "f2": float(opt_metrics.get("f2", 0)),
                }
                break

        # Fallback to flat metrics
        if "optimal" not in processed_metrics:
            processed_metrics["optimal"] = {
                "roc_auc": float(metrics.get("roc_auc", 0)),
                "f1": float(metrics.get("f1", metrics.get("f1_score", 0))),
                "recall": float(metrics.get("recall", 0)),
                "precision": float(metrics.get("precision", 0)),
                "f2": float(metrics.get("f2", 0)),
            }

        processed_results[model_name] = processed_metrics

    history = {
        "training_run": {
            "best_model": best_model_name,
            "models_trained": list(all_results.keys()),
            "dataset_info": {
                "source": "HTRU2 Kaggle Dataset",
                "total_samples": (
                    len(data_handler.df) if hasattr(data_handler, "df") else "unknown"
                ),
            },
        },
        "model_results": processed_results,
        "timestamp": datetime.now().isoformat(),
    }

    history_file = history_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Training history saved to {history_file}")
    logger.info("[SUCCESS] Training history saved")


def check_data_balance(df, target_col="signal", logger=None):
    """Check the balance of the target variable."""
    # If no logger provided, use the default logger
    if logger is None:
        logger = logging.getLogger()

    logger.info("Data Balance Check:")

    try:
        value_counts = df[target_col].value_counts()
        percentages = df[target_col].value_counts(normalize=True) * 100

        for value, count in value_counts.items():
            percentage = percentages[value]
            logger.info(f"  Class {value}: {count} samples ({percentage:.2f}%)")

        # Save data balance info
        # For binary and multi-class problems, consider the data imbalanced if the gap
        # between the most and least frequent classes exceeds 20 percentage points.
        if len(percentages) >= 2:
            imbalance_gap = float(percentages.max() - percentages.min())
            is_imbalanced = bool(imbalance_gap > 20.0)
        else:
            # Single-class data is by definition extremely imbalanced
            imbalance_gap = 100.0
            is_imbalanced = True

        balance_info = {
            "class_distribution": {int(k): int(v) for k, v in value_counts.to_dict().items()},
            "class_percentages": {int(k): float(v) for k, v in percentages.to_dict().items()},
            "total_samples": int(len(df)),
            "imbalance_gap_percent": float(imbalance_gap),
            "is_imbalanced": is_imbalanced,
        }

        os.makedirs("outputs/metrics", exist_ok=True)
        with open("outputs/metrics/data_balance.json", "w") as f:
            json.dump(balance_info, f, indent=2, cls=NumpyEncoder)

        return balance_info

    except KeyError:
        logger.error(
            f"[ERROR] Target column '{target_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )
        raise
    except Exception as e:
        logger.error(f"[ERROR] Error checking data balance: {e}")
        raise
