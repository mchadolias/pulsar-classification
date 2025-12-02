"""
Final updated training.py — restores backward compatibility and fixes evaluate signature.
Features:
- Automatic numeric feature detection
- F1 used for GridSearchCV, F2 for threshold optimisation
- evaluate(save_plots=...) restored for compatibility with existing scripts
- Robust metric calculations and safe handling for single-class cases
- Plot generation when save_plots=True
"""

import pickle
import toml
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    fbeta_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier


class ModelTrainer:
    def __init__(self, config_path: str, logger=None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        config = toml.load(config_path)
        config = self._convert_nulls(config)

        self.model_params = {
            "logistic_regression": config.get("logistic_regression", {}),
            "random_forest": config.get("random_forest", {}),
            "gradient_boosting": config.get("gradient_boosting", {}),
            "xgboost": config.get("xgboost", {}),
        }

        grid_config = config.get("grid", {})
        self.param_grids = {
            "logistic_regression": grid_config.get("logistic_regression", {}),
            "random_forest": grid_config.get("random_forest", {}),
            "gradient_boosting": grid_config.get("gradient_boosting", {}),
            "xgboost": grid_config.get("xgboost", {}),
        }

        self.training_cfg = config.get("training", {})
        self.data_cfg = config.get("data", {})
        self.cv_cfg = config.get("cv", {})
        self.threshold_cfg = config.get("threshold", {})
        self.metrics_cfg = config.get("metrics", {})
        self.costs_cfg = config.get("costs", {})

        # Use sklearn-compatible metric for GridSearchCV
        self.scoring_metric = "f1"
        self.optimal_threshold = 0.5
        self.class_weights = None
        self._sample_X_train = None
        self.calibration_data = {}

        self.models = self._initialize_models_with_weights()
        self.best_model = None
        self.best_model_name = None

    # -----------------------------
    # Helpers
    # -----------------------------
    def _convert_nulls(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_nulls(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_nulls(v) for v in obj]
        return None if obj in ("null", "None", None) else obj

    def set_training_sample(self, X_train):
        """Store a sample (DataFrame) for automatic feature detection."""
        self._sample_X_train = X_train

    def _clean_params(self, d):
        return {k: v for k, v in d.items() if v is not None}

    # -----------------------------
    # Model initialization
    # -----------------------------
    def _initialize_models_with_weights(self):
        use_weights = self.data_cfg.get("class_weights", False)
        strategy = self.data_cfg.get("weight_strategy", "balanced")
        custom = self.data_cfg.get("custom_weights", [1.0, 10.0])

        models = {}

        lr_params = self._clean_params(self.model_params["logistic_regression"].copy())
        if use_weights:
            if strategy == "balanced":
                lr_params["class_weight"] = "balanced"
            elif strategy == "custom":
                lr_params["class_weight"] = {0: custom[0], 1: custom[1]}
        models["logistic_regression"] = LogisticRegression(**lr_params)

        rf_params = self._clean_params(self.model_params["random_forest"].copy())
        if use_weights:
            if strategy == "balanced":
                rf_params["class_weight"] = "balanced"
            elif strategy == "balanced_subsample":
                rf_params["class_weight"] = "balanced_subsample"
            elif strategy == "custom":
                rf_params["class_weight"] = {0: custom[0], 1: custom[1]}
        models["random_forest"] = RandomForestClassifier(**rf_params)

        gb_params = self._clean_params(self.model_params["gradient_boosting"].copy())
        models["gradient_boosting"] = GradientBoostingClassifier(**gb_params)

        xgb_params = self._clean_params(self.model_params["xgboost"].copy())
        models["xgboost"] = XGBClassifier(eval_metric="logloss", **xgb_params)

        return models

    def _compute_scale_pos_weight(self, y):
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        if pos == 0:
            return 1.0
        return max(1.0, neg / pos)

    # -----------------------------
    # Pipeline & features
    # -----------------------------
    def make_pipeline(self, model_name: str):
        if self.data_cfg.get("numerical_features"):
            numerical = self.data_cfg.get("numerical_features")
        else:
            if self._sample_X_train is None:
                raise ValueError("Call trainer.set_training_sample(X_train) before training")
            numerical = self._sample_X_train.select_dtypes(include=[np.number]).columns.tolist()
            self.data_cfg["numerical_features"] = numerical
            self.logger.info(f"Auto-detected numerical features: {numerical}")

        pre = ColumnTransformer([("num", StandardScaler(), numerical)])
        pipeline = Pipeline([("preprocessor", pre), ("classifier", self.models[model_name])])
        return pipeline

    # -----------------------------
    # Training
    # -----------------------------
    def train(self, X_train, y_train, model_name: str):
        self.logger.info(f"Starting training for {model_name}...")

        if model_name is None or model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        # ensure feature inference sample is set
        if self._sample_X_train is None:
            self.set_training_sample(X_train)

        # set XGBoost imbalance parameter
        scale = self._compute_scale_pos_weight(y_train)
        if model_name == "xgboost":
            try:
                self.models["xgboost"].set_params(scale_pos_weight=scale)
            except Exception:
                self.logger.warning("Could not set scale_pos_weight on xgboost model")

        pipeline = self.make_pipeline(model_name)
        param_grid = self._clean_param_grid(self.param_grids.get(model_name, {}))

        if param_grid:
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid={f"classifier__{k}": v for k, v in param_grid.items()},
                scoring=self.scoring_metric,
                cv=self.cv_cfg.get("folds", 3),
                n_jobs=self.training_cfg.get("n_jobs", -1),
                verbose=self.training_cfg.get("verbose", 1),
            )
            grid.fit(X_train, y_train)
            self.best_model = grid.best_estimator_
            self.logger.info(f"GridSearchCV best score: {grid.best_score_}")
        else:
            pipeline.fit(X_train, y_train)
            self.best_model = pipeline

        self.best_model_name = model_name
        self.logger.info(f"Training complete for {model_name}")

    def _clean_param_grid(self, grid):
        cleaned = {}
        for k, v in grid.items():
            if isinstance(v, list):
                v = [x for x in v if x is not None]
                if v:
                    cleaned[k] = v
            elif v is not None:
                cleaned[k] = v
        return cleaned

    # -----------------------------
    # Evaluation (backward-compatible)
    # -----------------------------
    def evaluate(self, X_test, y_test, save_plots: bool = False):
        """
        Evaluate model and return a metrics dict. Kept backward-compatible keys that
        the rest of your pipeline expects.
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train() first.")

        # predict probabilities safely
        try:
            y_proba = self.best_model.predict_proba(X_test)[:, 1]
        except Exception:
            # Some models may not implement predict_proba; fallback to predict
            y_pred_only = self.best_model.predict(X_test)
            y_proba = np.array(y_pred_only, dtype=float)

        # optimise threshold using configured F-beta metric (default: F2)
        if self.threshold_cfg.get("optimization", True):
            self.optimal_threshold = self._optimize_threshold(y_test, y_proba)
        else:
            self.optimal_threshold = 0.5

        y_pred_default = (y_proba >= 0.5).astype(int)
        y_pred_opt = (y_proba >= self.optimal_threshold).astype(int)

        default_metrics = self._calculate_metrics(y_test, y_pred_default, y_proba)
        optimal_metrics = self._calculate_metrics(y_test, y_pred_opt, y_proba)

        metrics = {
            "default_threshold_0.5": default_metrics,
            f"optimal_threshold_{self.optimal_threshold:.3f}": optimal_metrics,
            "optimal_threshold": float(self.optimal_threshold),
            "model_name": self.best_model_name,
        }

        # add backward-compatible flat keys (most-used)
        opt = optimal_metrics
        metrics["roc_auc"] = float(opt.get("roc_auc", 0.0))
        metrics["f1"] = float(opt.get("f1", 0.0))
        metrics["f1_score"] = float(opt.get("f1", 0.0))
        metrics["recall"] = float(opt.get("recall", 0.0))
        metrics["precision"] = float(opt.get("precision", 0.0))
        metrics["f2"] = float(opt.get("f2", 0.0))

        # store calibration data
        try:
            self._calculate_calibration(y_test, y_proba)
            metrics["calibration"] = self.calibration_data
        except Exception as e:
            self.logger.debug(f"Calibration failed: {e}")

        # generate plots if asked
        if save_plots:
            try:
                self._generate_evaluation_plots(y_test, y_proba, y_pred_opt)
            except Exception as e:
                self.logger.warning(f"Plot generation failed: {e}")

        return metrics

    def _optimize_threshold(self, y_true, y_proba):
        method = self.threshold_cfg.get("method", "precision_recall")
        num_thresholds = int(self.threshold_cfg.get("num_thresholds", 50))

        if method == "precision_recall":
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            beta = float(self.threshold_cfg.get("beta", 1.0))
            # compute F-beta for each (precision, recall) pair
            f_scores = (
                (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-12)
            )
            # pr_thresholds length = len(precision)-1, align indices
            best_idx = int(np.nanargmax(f_scores[:-1])) if len(f_scores) > 1 else 0
            best_threshold = (
                pr_thresholds[min(best_idx, len(pr_thresholds) - 1)]
                if len(pr_thresholds) > 0
                else 0.5
            )
            return float(best_threshold)

        # fallback simple grid search on F-beta
        thresholds = np.linspace(0.01, 0.99, num_thresholds)
        best_th = 0.5
        best_score = -1
        beta = float(self.threshold_cfg.get("beta", 1.0))
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if score > best_score:
                best_score = score
                best_th = float(t)
        return best_th

    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        # safe roc
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                roc = float(roc_auc_score(y_true, y_proba))
            except Exception:
                roc = 0.0
        else:
            roc = 0.0

        # safe confusion matrix with explicit labels
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # fallback
            tn = int(cm[0, 0]) if cm.size > 0 else 0
            fp = fn = tp = 0

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

        # classification report
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except Exception:
            report = {}

        fp_cost = float(self.costs_cfg.get("false_positive", 1))
        fn_cost = float(self.costs_cfg.get("false_negative", 10))
        total_cost = fp * fp_cost + fn * fn_cost

        return {
            "roc_auc": roc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f2": f2,
            "confusion_matrix": cm.tolist(),
            "confusion_counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "total_cost": float(total_cost),
            "classification_report": report,
        }

    def _calculate_calibration(self, y_true, y_proba, n_bins: int = 10):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        self.calibration_data = {
            "prob_true": [float(x) for x in prob_true.tolist()],
            "prob_pred": [float(x) for x in prob_pred.tolist()],
        }

    def _generate_evaluation_plots(self, y_true, y_proba, y_pred):
        output_dir = Path("outputs/plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        # ROC
        if len(np.unique(y_true)) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = roc_auc_score(y_true, y_proba)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.3f})")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC - {self.best_model_name}")
                plt.legend()
                plt.savefig(
                    output_dir / f"roc_{self.best_model_name}.png", dpi=150, bbox_inches="tight"
                )
                plt.close()
            except Exception as e:
                self.logger.debug(f"ROC plot failed: {e}")

        # Precision-Recall
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall - {self.best_model_name}")
            plt.savefig(
                output_dir / f"pr_{self.best_model_name}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
        except Exception as e:
            self.logger.debug(f"PR plot failed: {e}")

        # Threshold analysis
        try:
            thresholds = np.linspace(0.01, 0.99, 100)
            f1_scores = [
                f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds
            ]
            f2_scores = [
                fbeta_score(y_true, (y_proba >= t).astype(int), beta=2, zero_division=0)
                for t in thresholds
            ]

            plt.figure(figsize=(6, 5))
            plt.plot(thresholds, f1_scores, label="F1")
            plt.plot(thresholds, f2_scores, label="F2")
            plt.axvline(
                self.optimal_threshold,
                color="r",
                linestyle="--",
                label=f"Optimal {self.optimal_threshold:.3f}",
            )
            plt.axvline(0.5, color="g", linestyle="--", label="Default 0.5")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title(f"Threshold analysis - {self.best_model_name}")
            plt.legend()
            plt.savefig(
                output_dir / f"threshold_{self.best_model_name}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
        except Exception as e:
            self.logger.debug(f"Threshold plot failed: {e}")

    # -----------------------------
    # Utilities
    # -----------------------------
    def predict_with_threshold(self, X, threshold=None):
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train() first.")
        if threshold is None:
            threshold = self.optimal_threshold

        # Try to obtain calibrated probabilities; fall back to hard predictions if needed
        try:
            proba = self.best_model.predict_proba(X)[:, 1]
        except Exception:
            y_pred_only = self.best_model.predict(X)
            proba = np.array(y_pred_only, dtype=float)

        preds = (proba >= threshold).astype(int)
        return {"predictions": preds, "probabilities": proba, "threshold": threshold}

    def get_feature_importances(self, feature_names=None):
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train() first.")
        try:
            model = self.best_model.named_steps["classifier"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                if feature_names is None or len(feature_names) != len(importances):
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                return dict(zip(feature_names, importances))
            else:
                self.logger.warning("Model does not expose feature_importances_")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to extract feature importances: {e}")
            return {}

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train() first.")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train() first.")
        return self.best_model.predict_proba(X)

    def save_model(self, file_path=None):
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        if file_path is None:
            file_path = self.training_cfg.get("save_path", "models/best_model.pkl")
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.best_model, f)

    def load_model(self, file_path):
        with open(file_path, "rb") as f:
            self.best_model = pickle.load(f)
        self.logger.info(f"Model loaded from {file_path}")
