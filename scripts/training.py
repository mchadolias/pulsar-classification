# scripts/training.py
import pickle
import numpy as np
import toml
import logging

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


class ModelTrainer:
    """Class for training and evaluating ML models using TOML configuration."""

    def __init__(self, config_path: str, logger=None):
        # Set up logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.logger.info("🤖 Initializing ModelTrainer...")

        try:
            # Load full TOML config
            config = toml.load(config_path)
            self.logger.info(f"✅ Config loaded from: {config_path}")

            # Extract model parameters (direct sections)
            self.model_params = {
                "logistic_regression": config.get("logistic_regression", {}),
                "random_forest": config.get("random_forest", {}),
                "gradient_boosting": config.get("gradient_boosting", {}),
                "xgboost": config.get("xgboost", {}),
            }

            # Extract grid parameters (grid.* sections)
            self.param_grids = {
                "logistic_regression": config.get("grid", {}).get("logistic_regression", {}),
                "random_forest": config.get("grid", {}).get("random_forest", {}),
                "gradient_boosting": config.get("grid", {}).get("gradient_boosting", {}),
                "xgboost": config.get("grid", {}).get("xgboost", {}),
            }

            self.training_cfg = config.get("training", {})
            self.data_cfg = config.get("data", {})
            self.cv_cfg = config.get("cv", {})

            # Log configuration summary
            self.logger.info("📋 Configuration Summary:")
            self.logger.info(f"  - Models configured: {list(self.model_params.keys())}")
            self.logger.info(
                f"  - Training scoring: {self.training_cfg.get('scoring', 'roc_auc')}"
            )
            self.logger.info(f"  - CV folds: {self.cv_cfg.get('folds', 5)}")

            # Instantiate model dictionary
            self.models = {
                "logistic_regression": LogisticRegression(
                    **self.model_params.get("logistic_regression", {})
                ),
                "random_forest": RandomForestClassifier(
                    **self.model_params.get("random_forest", {})
                ),
                "gradient_boosting": GradientBoostingClassifier(
                    **self.model_params.get("gradient_boosting", {})
                ),
                "xgboost": XGBClassifier(
                    eval_metric="logloss",
                    **self.model_params.get("xgboost", {}),
                ),
            }

            self.best_model = None
            self.best_model_name = None
            self.logger.info("✅ ModelTrainer initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ModelTrainer: {e}")
            raise

    # Preprocessing pipeline (numerical only for now)
    def make_pipeline(self, model_name: str):
        numerical_features = self.data_cfg.get("numerical_features", [])

        self.logger.debug(
            f"Creating pipeline for {model_name} with {len(numerical_features)} numerical features"
        )

        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), numerical_features)]
        )

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", self.models[model_name])]
        )

        return pipeline

    # Training method
    def train(self, X_train, y_train, model_name: str):
        self.logger.info(f"🎯 Starting training for {model_name}...")
        self.logger.info(f"  Training data shape: {X_train.shape}")
        self.logger.info(f"  Target distribution: {dict(y_train.value_counts())}")

        if model_name not in self.models:
            error_msg = f"Model '{model_name}' is not supported. Available models: {list(self.models.keys())}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Build pipeline for preprocessing + model
        pipeline = self.make_pipeline(model_name)

        # Hyperparameter grid if exists
        param_grid = self.param_grids.get(model_name, None)

        # Convert grid keys to refer to "classifier__param"
        if param_grid is not None:
            param_grid = {f"classifier__{k}": v for k, v in param_grid.items()}
            self.logger.info(f"  Hyperparameter grid: {len(param_grid)} parameters")

        if param_grid:
            self.logger.info(f"🔍 Running GridSearchCV for {model_name}...")
            self.logger.info(f"  CV folds: {self.cv_cfg.get('folds', 5)}")
            self.logger.info(f"  Scoring: {self.training_cfg.get('scoring', 'roc_auc')}")

            try:
                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    scoring=self.training_cfg.get("scoring", "roc_auc"),
                    cv=self.cv_cfg.get("folds", 5),
                    n_jobs=-1,
                    verbose=self.training_cfg.get("verbose", 1),
                )
                grid.fit(X_train, y_train)

                self.best_model = grid.best_estimator_
                self.best_model_name = model_name

                self.logger.info(f"✅ GridSearchCV completed for {model_name}")
                self.logger.info(f"✔ Best score: {grid.best_score_:.4f}")
                self.logger.info(f"✔ Best params: {grid.best_params_}")

            except Exception as e:
                self.logger.error(f"❌ GridSearchCV failed for {model_name}: {e}")
                raise

        else:
            self.logger.info(
                f"⚠ No hyperparameter grid provided. Training {model_name} with default parameters..."
            )
            try:
                pipeline.fit(X_train, y_train)
                self.best_model = pipeline
                self.best_model_name = model_name
                self.logger.info(f"✅ {model_name} trained successfully with default parameters")
            except Exception as e:
                self.logger.error(f"❌ Training failed for {model_name}: {e}")
                raise

    # Evaluation method
    def evaluate(self, X_test, y_test):
        if self.best_model is None:
            error_msg = "No model has been trained yet. Call train() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"📊 Evaluating model {self.best_model_name}...")
        self.logger.info(f"  Test data shape: {X_test.shape}")
        self.logger.info(f"  Test target distribution: {dict(y_test.value_counts())}")

        try:
            y_pred = self.best_model.predict(X_test)
            y_proba = self.best_model.predict_proba(X_test)[:, 1]

            metrics = {
                "roc_auc": roc_auc_score(y_test, y_proba),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
            }

            self.logger.info(f"✅ Evaluation completed for {self.best_model_name}")
            self.logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            self.logger.info(f"  Recall: {metrics['recall']:.4f}")
            self.logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
            self.logger.info(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}")
            raise

    # Feature importances (if supported)
    def get_feature_importances(self, feature_names):
        if self.best_model is None:
            error_msg = "No model has been trained yet. Call train() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("🔍 Extracting feature importances...")

        try:
            model = self.best_model.named_steps["classifier"]

            if hasattr(model, "feature_importances_"):
                importances = dict(zip(feature_names, model.feature_importances_))
                self.logger.info("✅ Feature importances extracted successfully")

                # Log top 5 features
                top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                self.logger.info("📈 Top 5 features:")
                for feature, importance in top_features:
                    self.logger.info(f"  {feature}: {importance:.4f}")

                return importances
            else:
                warning_msg = f"Model {type(model).__name__} does not support feature importances"
                self.logger.warning(warning_msg)
                raise ValueError(warning_msg)

        except Exception as e:
            self.logger.error(f"❌ Failed to extract feature importances: {e}")
            raise

    # Prediction wrapper
    def predict(self, X):
        if self.best_model is None:
            error_msg = "No model has been trained yet. Call train() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.debug(f"Making predictions on {X.shape[0]} samples")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            error_msg = "No model has been trained yet. Call train() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.debug(f"Making probability predictions on {X.shape[0]} samples")
        return self.best_model.predict_proba(X)

    # Persistence
    def save_model(self, file_path=None):
        if self.best_model is None:
            error_msg = "No model to save. Train a model first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if file_path is None:
            file_path = self.training_cfg.get("save_path", "saved_model.pkl")

        try:
            with open(file_path, "wb") as f:
                pickle.dump(self.best_model, f)

            self.logger.info(f"💾 Model saved to {file_path}")
            self.logger.info(f"  Model type: {self.best_model_name}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save model to {file_path}: {e}")
            raise

    def load_model(self, file_path):
        try:
            with open(file_path, "rb") as f:
                self.best_model = pickle.load(f)

            self.logger.info(f"📦 Model loaded from {file_path}")
            self.logger.info(f"  Model type: {type(self.best_model).__name__}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model from {file_path}: {e}")
            raise
