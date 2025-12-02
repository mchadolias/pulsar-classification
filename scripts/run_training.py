"""
Main training pipeline for HTRU2 Pulsar Classification.
Runs the complete ML pipeline with enhanced imbalance handling.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import DataConfig
from src.data_handler import HTRU2DataHandler
from src.training import ModelTrainer
from src.utils import (
    setup_logger,
    log_section_header,
    check_data_balance,
    save_model_comparison,
    save_test_predictions,
    save_final_results,
    save_training_history,
)


# ---------------------------------------------------------
# 1. Argument parser
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the HTRU2 Pulsar Classification ML Training Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a TOML model configuration file. "
            "If not specified, the script will look for test_model.toml or configs/test_model.toml"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------
# 2. Main
# ---------------------------------------------------------


def main():
    """Main script to run the complete ML pipeline using existing TOML config."""
    args = parse_args()

    # Setup logger
    logger, log_filename = setup_logger()

    # Create output directories
    output_dirs = [
        "outputs",
        "outputs/models",
        "outputs/metrics",
        "outputs/predictions",
        "outputs/plots",
    ]
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)

    logger.info("Starting HTRU2 Pulsar Classification Pipeline...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Log file: {log_filename}")

    # ---------------------------------------------------------
    # CONFIG SELECTION LOGIC (ARGPARSE + FALLBACK)
    # ---------------------------------------------------------
    if args.config is not None:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.info(f"Using config file from CLI: {config_path}")
    else:
        # Fallback search logic:
        default_paths = [Path("test_model.toml"), Path("configs/test_model.toml")]
        config_path = None
        for p in default_paths:
            if p.exists():
                config_path = p.resolve()
                logger.info(f"Using default config file: {config_path}")
                break
        if config_path is None:
            raise FileNotFoundError(
                "No config file provided and no default test_model.toml found."
            )

    try:
        # Load data configuration
        data_config = DataConfig()
        logger.info("Data configuration loaded")

        # Initialize data handler
        data_handler = HTRU2DataHandler(data_config, logger=logger)

        # Step 1: Download data (if needed)
        log_section_header(logger, "STEP 1: DOWNLOADING DATA")
        data_handler.download_kaggle()

        # Step 2: Load data
        log_section_header(logger, "STEP 2: LOADING DATA")
        df = data_handler.load()
        logger.info(f"Dataset shape: {df.shape}")

        # Step 3: Preprocess data
        log_section_header(logger, "STEP 3: PREPROCESSING DATA")
        df_processed = data_handler.preprocess()
        logger.info(f"Processed dataset shape: {df_processed.shape}")

        # Check data balance
        check_data_balance(df_processed, logger=logger)

        # Step 4: Split data
        log_section_header(logger, "STEP 4: SPLITTING DATA")
        splits = data_handler.split_train_val_test()

        # Export data splits
        logger.info("Exporting data splits...")
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

        # config_path already set by argparse + fallback logic at the top
        trainer = ModelTrainer(str(config_path), logger=logger)
        logger.info(f"Model trainer initialized with config: {config_path}")

        # Add numerical features to trainer's data config
        trainer.data_cfg["numerical_features"] = feature_cols
        logger.info(f"Using numerical features: {feature_cols}")

        # Step 7: Train models
        log_section_header(logger, "STEP 7: TRAINING MODELS")

        models_to_train = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]

        best_metrics = {}
        best_model_name = None
        best_f1_score = 0  # Using F1 for imbalance
        all_results = {}

        for model_name in models_to_train:
            log_section_header(logger, f"TRAINING {model_name.upper()}")

            try:
                # Train the model
                trainer.train(X_train, y_train, model_name)

                # Evaluate on validation set with threshold optimization
                metrics = trainer.evaluate(X_val, y_val, save_plots=True)

                logger.info(f"{model_name} Validation Metrics:")

                # Log optimal threshold metrics (per-model)
                optimal_key = None
                for key in metrics.keys():
                    if key.startswith("optimal_threshold_"):
                        optimal_key = key
                        break

                if optimal_key is not None:
                    opt_metrics = metrics[optimal_key]
                    logger.info(
                        f"  Optimal threshold: {metrics.get('optimal_threshold', trainer.optimal_threshold):.3f}"
                    )
                    logger.info(f"  F1 Score: {opt_metrics.get('f1', 0):.4f}")
                    logger.info(f"  Recall: {opt_metrics.get('recall', 0):.4f}")
                    logger.info(f"  Precision: {opt_metrics.get('precision', 0):.4f}")
                else:
                    logger.warning(
                        f"No optimal_threshold_* key found for {model_name}; falling back to flat metrics."
                    )

                # Store results
                all_results[model_name] = metrics

                # Track best model based on F1 score (better for imbalance)
                if optimal_key is not None:
                    current_f1 = metrics[optimal_key].get("f1", 0)
                else:
                    current_f1 = metrics.get("f1", metrics.get("f1_score", 0))

                if current_f1 > best_f1_score:
                    best_f1_score = current_f1
                    best_model_name = model_name
                    best_metrics = metrics

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        # Save model comparison results
        save_model_comparison(all_results, best_model_name, best_f1_score, logger)

        # Display comparison of all models
        log_section_header(logger, "MODEL COMPARISON SUMMARY")
        for model_name, metrics in all_results.items():
            optimal_key = None
            for key in metrics.keys():
                if key.startswith("optimal_threshold_"):
                    optimal_key = key
                    break

            if optimal_key is not None:
                opt_metrics = metrics[optimal_key]
            else:
                opt_metrics = metrics

            f1 = opt_metrics.get("f1", opt_metrics.get("f1_score", 0))
            recall = opt_metrics.get("recall", 0)
            roc_auc = opt_metrics.get("roc_auc", 0)
            logger.info(
                f"{model_name:20} | F1: {f1:.4f} | Recall: {recall:.4f} | ROC AUC: {roc_auc:.4f}"
            )

        logger.info(f"Best model: {best_model_name} (F1 Score: {best_f1_score:.4f})")

        # Step 8: Final evaluation with best model
        logger.info(
            f"Retraining best model ({best_model_name}) on combined training + validation data..."
        )

        # Combine train and validation sets
        X_full_train = pd.concat([X_train, X_val], axis=0)
        y_full_train = pd.concat([y_train, y_val], axis=0)

        # Retrain best model on full training data
        trainer.train(X_full_train, y_full_train, best_model_name)

        # Evaluate on test set
        log_section_header(logger, "FINAL EVALUATION ON TEST SET")
        test_metrics = trainer.evaluate(X_test, y_test, save_plots=True)

        logger.info("Test Set Metrics (Optimal Threshold):")
        optimal_key = None
        for key in test_metrics.keys():
            if key.startswith("optimal_threshold_"):
                optimal_key = key
                break

        if optimal_key is not None:
            opt_test_metrics = test_metrics[optimal_key]
            for metric_name, value in opt_test_metrics.items():
                if metric_name not in [
                    "confusion_matrix",
                    "classification_report",
                    "confusion_counts",
                ]:
                    logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.warning(
                "No optimal_threshold_* key found in test_metrics; skipping detailed per-metric log."
            )

        # Save test predictions
        save_test_predictions(trainer, X_test, y_test, best_model_name, logger)

        # Step 9: Feature importance (if available)
        feature_importances = {}
        try:
            logger.info("Feature Importances:")
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

        logger.info("Pipeline completed successfully!")
        logger.info(f"Best model saved as: {model_save_path}")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Optimal threshold: {trainer.optimal_threshold:.4f}")

        # Log a brief test summary if an optimal-threshold block is available
        optimal_key = None
        for key in test_metrics.keys():
            if key.startswith("optimal_threshold_"):
                optimal_key = key
                break

        if optimal_key is not None:
            opt_metrics = test_metrics[optimal_key]
            logger.info(f"Test F1 Score: {opt_metrics.get('f1', 0):.4f}")
            logger.info(f"Test Recall: {opt_metrics.get('recall', 0):.4f}")

        logger.info(f"All outputs saved in: outputs/")
        logger.info(f"Log file: {log_filename}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.error("Stack trace:", exc_info=True)
        raise


if __name__ == "__main__":
    main()
