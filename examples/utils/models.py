from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Import additional model types
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class BaseModel:
    """
    Base class for all models providing common functionality.
    """

    def __init__(
        self,
        feature_mode: str = "baseline",
        random_state=42,
        test_size=0.2,
        threshold_strategy="f_beta",
        beta=2.0,
        min_precision=None,
        min_recall=None,
        custom_threshold=None,
        target_sensitivity=None,
        compute_ci: bool = True,
        ci_n_bootstraps: int = 1000,
        ci_alpha: float = 0.05,
        save_path: Path = None,
    ):
        self.feature_mode = feature_mode
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = "base"
        self.metrics = {}
        self.feature_importance = None
        self.threshold_strategy = threshold_strategy
        self.threshold_params = {
            "beta": beta,
            "min_precision": min_precision,
            "min_recall": min_recall,
            "custom_threshold": custom_threshold,
            "target_sensitivity": target_sensitivity,
        }
        self.save_path = save_path
        self.compute_ci = compute_ci
        self.ci_n_bootstraps = max(0, int(ci_n_bootstraps))
        self.ci_alpha = float(ci_alpha)
        # Hold onto the most recent train/test split and predictions for downstream explainer tooling.
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self._shap_model = None
        self.test_ids = None  # Store test set IDs for subset evaluation

    def _prepare_data(self, data, target_col, id_col="id", group_split=True):
        """
        Prepare data for training and testing, with option for mission/case-level splitting.

        Args:
            data: DataFrame containing features and target
            target_col: Name of the target column
            id_col: Name of the mission/case ID column
            group_split: If True, splitting will respect mission/case boundaries
        """
        if not group_split:
            # Original row-level splitting (not respecting mission/case boundaries)
            X = data.select(pl.exclude([id_col, target_col])).to_numpy()
            y = data.select(pl.col(target_col)).to_numpy().ravel()

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
        else:
            # Group-level splitting (respecting mission/case boundaries)
            # Build a single, sorted table of unique case IDs and labels to keep alignment deterministic
            case_labels = (
                data.select([id_col, target_col]).unique(subset=id_col, keep="first").sort(id_col)
            )
            unique_cases = case_labels.select(id_col).to_numpy().ravel()
            case_y = case_labels.select(target_col).to_numpy().ravel()

            # Split the case IDs
            train_ids, test_ids, _, _ = train_test_split(
                unique_cases,
                case_y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=case_y,
            )

            # Store test IDs for later subset evaluation
            self.test_ids = set(test_ids)

            # Filter data based on the split IDs
            train_data = data.filter(pl.col(id_col).is_in(train_ids))
            test_data = data.filter(pl.col(id_col).is_in(test_ids))

            # Extract features and targets
            X_train = train_data.select(pl.exclude([id_col, target_col])).to_numpy()
            y_train = train_data.select(pl.col(target_col)).to_numpy().ravel()
            X_test = test_data.select(pl.exclude([id_col, target_col])).to_numpy()
            y_test = test_data.select(pl.col(target_col)).to_numpy().ravel()

            print(f"Split data into {len(train_ids)} train cases and {len(test_ids)} test cases")
            print(f"Training data: {X_train.shape[0]} rows, Test data: {X_test.shape[0]} rows")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # X_train_balanced, y_train_balanced = SMOTE(sampling_strategy=0.3, random_state=42).fit_resample(X_train_scaled, y_train)
        # print(f"Before SMOTE: {sum(y_train)} positives, {len(y_train)-sum(y_train)} negatives")
        # print(f"After SMOTE:  {sum(y_train_balanced)} positives, {len(y_train_balanced)-sum(y_train_balanced)} negatives")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, data, target_col, id_col="id", group_split=True):
        """
        Train the model and evaluate performance.

        Args:
            data: DataFrame containing features and target
            target_col: Name of the target column
            id_col: Name of the mission/case ID column
            group_split: If True, splitting will respect mission/case boundaries
        """
        if target_col is None:
            raise ValueError("Please provide the name of the target column.")

        X_train_scaled, X_test_scaled, y_train, y_test = self._prepare_data(
            data, target_col, id_col=id_col, group_split=group_split
        )

        # Persist the split so external consumers can access it (e.g. SHAP plots).
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

        # To be implemented by derived classes
        self._train_model(X_train_scaled, y_train)

        # Default SHAP explainer target falls back to the trained estimator.
        if self._shap_model is None:
            self._shap_model = self.model

        # Make predictions
        y_pred, y_pred_proba = self._predict(X_test_scaled)
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        # Calculate metrics
        metrics, best_y_pred = self._calculate_metrics(y_test, y_pred_proba)
        self.metrics = metrics

        print(
            f"\nClassification Report for {self.model_type} (threshold={metrics['threshold']:.3f}):"
        )
        print(classification_report(y_test, best_y_pred))

        return self.model, self.metrics

    def evaluate_on_subset(self, data, target_col, id_col="id", subset_name="subset"):
        """
        Evaluate an already-trained model on a subset of data.

        IMPORTANT: Only evaluates on cases that were in the original test set
        to avoid data leakage from training samples.

        Parameters:
            data (pl.DataFrame): DataFrame containing features and target for the subset.
            target_col (str): Name of the target column.
            id_col (str): Name of the ID column.
            subset_name (str): Name of the subset for plot titles and filenames.

        Returns:
            dict: Metrics dictionary for the subset (test cases only).
        """
        if self.model is None:
            raise RuntimeError("Model hasn't been trained yet. Call train() first.")

        if self.test_ids is None:
            raise RuntimeError("No test IDs stored. Model must be trained with group_split=True.")

        # Filter to only include cases from the original test set
        test_subset = data.filter(pl.col(id_col).is_in(list(self.test_ids)))

        if test_subset.height == 0:
            raise ValueError("No test set cases found in the provided subset.")

        n_subset_cases = test_subset.n_unique(id_col)
        print(f"Evaluating on {n_subset_cases} test cases ({test_subset.height} rows)")

        # Extract features and target
        X = test_subset.select(pl.exclude([id_col, target_col])).to_numpy()
        y = test_subset.select(pl.col(target_col)).to_numpy().ravel()

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        # Scale features using the fitted scaler
        X_scaled = self.scaler.transform(X)

        # Make predictions
        y_pred, y_pred_proba = self._predict(X_scaled)

        # Calculate metrics
        metrics, _ = self._calculate_metrics(y, y_pred_proba, subset=subset_name)

        return metrics

    def _train_model(self, X_train, y_train):
        """To be implemented by derived classes."""
        raise NotImplementedError("This method should be implemented by derived classes.")

    def _predict(self, X_test):
        """Make predictions with the trained model."""
        if self.model is None:
            raise RuntimeError("Model hasn't been trained yet.")

        y_pred = self.model.predict(X_test)
        # Most sklearn-like models will have predict_proba
        try:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        except AttributeError:
            # Fall back for models without predict_proba
            y_pred_proba = y_pred

        return y_pred, y_pred_proba

    def _calculate_metrics(self, y_test, y_pred_proba, subset="full"):
        """Calculate performance metrics with an optimized probability threshold."""
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

        self._plot_pr(recall, precision, subset=subset)
        self._plot_roc(y_test, y_pred_proba, subset=subset)

        best_threshold = self._determine_threshold(precision, recall, thresholds)
        best_y_pred = (y_pred_proba >= best_threshold).astype(int)

        metrics = self._assemble_metrics(y_test, best_y_pred, y_pred_proba, best_threshold)

        if self.compute_ci and self.ci_n_bootstraps > 0:
            ci_bounds = self._bootstrap_confidence_intervals(y_test, y_pred_proba)
            metrics.update(ci_bounds)

        return metrics, best_y_pred

    def _assemble_metrics(self, y_true, y_pred, y_pred_proba, threshold):
        """Collect point estimates for the configured suite of metrics."""
        beta_val = self.threshold_params.get("beta", 1.0)

        metrics = {
            "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            f"f{beta_val}_score": fbeta_score(y_true, y_pred, beta=beta_val, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "auc_pr": average_precision_score(y_true, y_pred_proba),
            "brier": brier_score_loss(y_true, y_pred_proba),
            "threshold": float(threshold),
            "threshold_strategy": self.threshold_strategy,
        }

        target_sensitivity = self.threshold_params.get("target_sensitivity")
        if target_sensitivity is not None:
            spec_at_sens, achieved_sens, sens_threshold = self._specificity_at_sensitivity(
                y_true,
                y_pred_proba,
                float(target_sensitivity),
            )
            metrics.update(
                {
                    "target_sensitivity": float(target_sensitivity),
                    "specificity_at_target_sensitivity": float(spec_at_sens),
                    "achieved_sensitivity_at_target_threshold": float(achieved_sens),
                    "threshold_at_target_sensitivity": float(sens_threshold),
                }
            )

        return metrics

    def _specificity_at_sensitivity(self, y_true, y_pred_proba, target_sensitivity):
        """Return the specificity, achieved sensitivity, and threshold at a target sensitivity."""
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        validity_mask = np.isfinite(thresholds)
        if not np.any(validity_mask):
            return 0.0, 0.0, float("nan")

        fpr = fpr[validity_mask]
        tpr = tpr[validity_mask]
        thresholds = thresholds[validity_mask]
        specificity = 1.0 - fpr

        viable = tpr >= target_sensitivity
        if np.any(viable):
            candidate_idx = np.flatnonzero(viable)
            best_idx = candidate_idx[int(np.nanargmax(specificity[candidate_idx]))]
        else:
            distance = np.abs(tpr - target_sensitivity)
            best_idx = int(np.nanargmin(distance))

        return specificity[best_idx], tpr[best_idx], thresholds[best_idx]

    def _bootstrap_confidence_intervals(self, y_true, y_pred_proba):
        """Estimate confidence intervals for evaluation metrics via bootstrapping."""
        rng = np.random.default_rng(self.random_state)
        n_samples = y_true.shape[0]
        lower_pct = 100 * (self.ci_alpha / 2)
        upper_pct = 100 * (1 - self.ci_alpha / 2)

        collected = {}
        successful_draws = 0

        for _ in range(self.ci_n_bootstraps):
            indices = rng.integers(0, n_samples, n_samples)
            sample_y = y_true[indices]
            sample_proba = y_pred_proba[indices]

            if np.unique(sample_y).size < 2:
                # Degenerate resample; skip to avoid undefined metrics.
                continue

            precision, recall, thresholds = precision_recall_curve(sample_y, sample_proba)
            sample_threshold = self._determine_threshold(precision, recall, thresholds)
            sample_pred = (sample_proba >= sample_threshold).astype(int)

            try:
                sample_metrics = self._assemble_metrics(
                    sample_y, sample_pred, sample_proba, sample_threshold
                )
            except ValueError:
                # Metrics like ROC AUC can raise if a class is absent after filtering; skip.
                continue

            for key, value in sample_metrics.items():
                if key == "threshold_strategy":
                    continue
                collected.setdefault(key, []).append(float(value))

            successful_draws += 1

        if successful_draws == 0 or not collected:
            return {"ci_replications": 0}

        ci_bounds = {"ci_replications": successful_draws, "ci_alpha": self.ci_alpha}
        for metric_name, values in collected.items():
            if not values:
                continue
            ci_bounds[f"{metric_name}_ci_lower"] = float(np.percentile(values, lower_pct))
            ci_bounds[f"{metric_name}_ci_upper"] = float(np.percentile(values, upper_pct))
            ci_bounds[f"{metric_name}_ci_mean"] = float(np.mean(values))

        return ci_bounds

    def _plot_pr(self, recall, precision, subset="full"):
        # plot and show PR curve
        plt.figure()
        plt.plot(recall, precision, marker=".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        subset_label = f" ({subset})" if subset != "full" else ""
        plt.title(f"Precision-Recall Curve for {self.model_type}{subset_label}")
        plt.savefig(
            self.save_path
            / f"pr_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.model_type}_{self.feature_mode}_{subset}.svg"
        )
        plt.close()

    def _plot_roc(self, y_test, y_pred_proba, subset="full"):
        # plot ROC
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        subset_label = f" ({subset})" if subset != "full" else ""
        plt.title(f"ROC Curve for {self.model_type}{subset_label}")
        plt.legend(loc="lower right")
        plt.savefig(
            self.save_path
            / f"roc_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.model_type}_{self.feature_mode}_{subset}.svg"
        )
        plt.close()

    def _determine_threshold(self, precision, recall, thresholds):
        """Select a probability threshold according to the configured strategy."""
        # If no thresholds are returned (degenerate case), fall back to provided override or 0.5.
        if thresholds.size == 0:
            override = self.threshold_params.get("custom_threshold")
            return float(override) if override is not None else 0.5

        precisions = precision[:-1]
        recalls = recall[:-1]
        eps = 1e-10
        strategy = getattr(self, "threshold_strategy", "f1")

        if strategy == "custom_threshold":
            custom_threshold = self.threshold_params.get("custom_threshold")
            if custom_threshold is None:
                raise ValueError(
                    "custom_threshold must be provided when strategy is 'custom_threshold'."
                )
            return float(custom_threshold)

        if strategy == "f_beta":
            beta = self.threshold_params.get("beta", 1.0)
            beta_sq = float(beta) ** 2
            scores = (1 + beta_sq) * precisions * recalls / (beta_sq * precisions + recalls + eps)
        elif strategy == "recall_at_precision":
            min_precision = self.threshold_params.get("min_precision")
            if min_precision is None:
                raise ValueError("min_precision must be set for recall_at_precision strategy.")
            mask = precisions >= min_precision
            if not np.any(mask):
                scores = precisions
            else:
                candidate_indices = np.flatnonzero(mask)
                chosen = candidate_indices[int(np.nanargmax(recalls[mask]))]
                return float(thresholds[chosen])
        elif strategy == "precision_at_recall":
            min_recall = self.threshold_params.get("min_recall")
            if min_recall is None:
                raise ValueError("min_recall must be set for precision_at_recall strategy.")
            mask = recalls >= min_recall
            if not np.any(mask):
                scores = recalls
            else:
                candidate_indices = np.flatnonzero(mask)
                chosen = candidate_indices[int(np.nanargmax(precisions[mask]))]
                return float(thresholds[chosen])
        else:
            # Default to F1 optimisation when strategy is 'f1' or unknown.
            scores = 2 * precisions * recalls / (precisions + recalls + eps)

        if np.all(np.isnan(scores)):
            return 0.5

        best_idx = int(np.nanargmax(scores))
        return float(thresholds[best_idx])

    def get_feature_importance(self, feature_names):
        """Get feature importance if available."""
        if self.feature_importance is None:
            return None

        importance_df = pl.DataFrame(
            {"feature": feature_names, "importance": self.feature_importance}
        ).sort("importance", descending=True)

        return importance_df

    def compute_shap(self, X, feature_names, save_dir: Path, prefix: str, file_format="svg"):
        """To be implemented by derived classes."""
        return
        # raise NotImplementedError("This method should be implemented by derived classes.")


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression implementation.
    """

    def __init__(
        self,
        feature_mode,
        random_state=42,
        test_size=0.3,
        save_path: Path = None,
        target_sensitivity=None,
    ):
        super().__init__(
            feature_mode,
            random_state,
            test_size,
            save_path=save_path,
            target_sensitivity=target_sensitivity,
        )
        self.model_type = "LogisticRegression"

    def _train_model(self, X_train, y_train):
        """Train a logistic regression model."""
        self.model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.model.fit(X_train, y_train)
        self.feature_importance = np.abs(self.model.coef_[0])


class LightGBMModel(BaseModel):
    """
    LightGBM implementation.
    """

    def __init__(
        self,
        feature_mode,
        random_state=42,
        test_size=0.3,
        params=None,
        save_path: Path = None,
        target_sensitivity=None,
    ):
        super().__init__(
            feature_mode,
            random_state,
            test_size,
            save_path=save_path,
            target_sensitivity=target_sensitivity,
        )
        self.model_type = "LightGBM"
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": random_state,
            "is_unbalance": True,  # Handle class imbalance
        }

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install it using 'pip install lightgbm'")

    def _train_model(self, X_train, y_train):
        """Train a LightGBM model."""
        # Create dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)

        # Train model
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=100,
            # LightGBM uses 'verbose' in params, not 'verbose_eval' as a separate parameter
        )

        # Store feature importance
        self.feature_importance = self.model.feature_importance(importance_type="gain")

    def _predict(self, X_test):
        """Override prediction for LightGBM to ensure binary classification output."""
        if self.model is None:
            raise RuntimeError("Model hasn't been trained yet.")

        # Get raw probability predictions
        y_pred_proba = self.model.predict(X_test)

        # Convert probabilities to binary predictions (0 or 1)
        y_pred = (y_pred_proba > 0.5).astype(int)

        return y_pred, y_pred_proba


class XGBoostModel(BaseModel):
    """
    XGBoost implementation with optional probability calibration.
    """

    def __init__(
        self,
        feature_mode="baseline",
        random_state=42,
        test_size=0.3,
        params=None,
        calibrate=False,
        calibration_method="isotonic",
        calibration_cv=5,
        save_path: Path = None,
        target_sensitivity=None,
    ):
        super().__init__(
            feature_mode,
            random_state,
            test_size,
            save_path=save_path,
            target_sensitivity=target_sensitivity,
        )
        self.model_type = "XGBoost"
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 300,
            "scale_pos_weight": None,  # set automatically in _train_model from class ratio
            "random_state": random_state,
            "tree_method": "hist",
            "n_jobs": 1,  # single-thread for strict determinism
            "predictor": "cpu_predictor",
            "verbosity": 0,
        }

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install it using 'pip install xgboost'")

        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.calibration_cv = StratifiedKFold(
            n_splits=calibration_cv, shuffle=True, random_state=random_state
        )

    def _train_model(self, X_train, y_train):
        """Train an XGBoost model and optionally calibrate its probabilities."""
        pos = float(y_train.sum())
        neg = float(y_train.shape[0] - pos)
        if self.params.get("scale_pos_weight") is None:
            self.params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0

        base_estimator = xgb.XGBClassifier(**self.params)

        if self.calibrate:
            calibrator = CalibratedClassifierCV(
                estimator=base_estimator,
                method=self.calibration_method,
                cv=self.calibration_cv,
            )
            calibrator.fit(X_train, y_train)
            self.model = calibrator

            # Train a standalone estimator for feature importance reporting.
            importance_estimator = xgb.XGBClassifier(**self.params)
            importance_estimator.fit(X_train, y_train)
            self.feature_importance = getattr(importance_estimator, "feature_importances_", None)
            self._shap_model = importance_estimator
        else:
            base_estimator.fit(X_train, y_train)
            self.model = base_estimator
            self.feature_importance = getattr(base_estimator, "feature_importances_", None)
            self._shap_model = base_estimator

        if self.feature_importance is not None:
            self.feature_importance = np.asarray(self.feature_importance)

    def _predict(self, X_test):
        """Use the fitted estimator (or calibrator) for predictions."""
        if self.model is None:
            raise RuntimeError("Model hasn't been trained yet.")

        y_pred = self.model.predict(X_test)
        try:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_pred_proba = y_pred

        return y_pred, y_pred_proba

    def compute_shap(
        self,
        X=None,
        feature_names=None,
        save_dir=None,
        prefix=None,
        file_format="svg",
    ):
        """Compute SHAP values and optionally persist plots.

        Parameters
        ----------
        X : np.ndarray | None, optional
            Samples to explain. Falls back to the stored hold-out set when omitted.
        feature_names : Sequence[str] | None, optional
            Names used when rendering the summary plot. Required when ``save_dir`` is provided
            and ``X`` is a plain numpy array without feature metadata.
        save_dir : str | Path | None, optional
            Directory where SHAP visualisations should be written. When omitted, the plots are
            not generated; only the explanation object is returned.
        prefix : str | None, optional
            Identifier injected into the output filenames; defaults to the model type.
        file_format : str, optional
            Matplotlib-supported format (e.g. "svg", "png") for persisted figures.

        Returns
        -------
        shap.Explanation | None
            The computed SHAP explanation, or ``None`` if SHAP is unavailable.
        """
        try:
            import shap
        except ImportError:
            print("SHAP is not installed. Install it using 'pip install shap'")
            return None

        model_for_shap = self._shap_model or self.model
        if model_for_shap is None:
            raise RuntimeError("Model hasn't been trained yet.")

        if X is None:
            X = self.X_test
        if X is None:
            raise ValueError(
                "No data provided for SHAP computation and no stored test set is available."
            )

        shap_target = (
            model_for_shap.get_booster()
            if hasattr(model_for_shap, "get_booster")
            else model_for_shap
        )
        explainer = shap.TreeExplainer(shap_target)
        explanation = explainer(X)

        if save_dir is not None:
            output_dir = Path(save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            tag = prefix or self.model_type
            tag = tag.replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            plt.figure()
            shap.summary_plot(
                explanation.values,
                X,
                feature_names=feature_names,
                show=False,
                max_display=30,
            )
            summary_path = output_dir / f"shap_summary_{timestamp}_{tag}.{file_format}"
            plt.savefig(summary_path, format=file_format)
            plt.close()

            mean_abs = np.abs(explanation.values).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:30]

            df_top20 = pl.DataFrame(
                {
                    "feature": np.array(feature_names)[top_idx],
                    "mean_abs_shap": mean_abs[top_idx],
                }
            )

            df_top20_path = output_dir / f"shap_top20_{timestamp}_{tag}.csv"
            df_top20.write_csv(df_top20_path)

        return explanation


class RandomForestModel(BaseModel):
    """
    Random Forest implementation.
    """

    def __init__(
        self,
        feature_mode,
        random_state=42,
        test_size=0.3,
        params=None,
        save_path: Path = None,
        target_sensitivity=None,
    ):
        super().__init__(
            feature_mode,
            random_state,
            test_size,
            save_path=save_path,
            target_sensitivity=target_sensitivity,
        )
        self.model_type = "RandomForest"
        self.params = params or {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": random_state,
            "class_weight": "balanced",
        }

    def _train_model(self, X_train, y_train):
        """Train a Random Forest model."""
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)

        # Store feature importance
        self.feature_importance = self.model.feature_importances_

    def compute_shap(self, X, feature_names, save_dir, prefix, file_format="svg"):
        """Compute SHAP values and optionally persist plots.

        Parameters
        ----------
        X : np.ndarray
            Samples to explain.
        feature_names : Sequence[str]
            Names used when rendering the summary plot.
        save_dir : str | Path
            Directory where SHAP visualisations should be written.
        prefix : str
            Identifier injected into the output filenames.
        file_format : str, optional
            Matplotlib-supported format (e.g. "svg", "png") for persisted figures.

        Returns
        -------
        shap.Explanation | None
            The computed SHAP explanation, or ``None`` if SHAP is unavailable.
        """
        try:
            import shap
        except ImportError:
            print("SHAP is not installed. Install it using 'pip install shap'")
            return None

        if self.model is None:
            raise RuntimeError("Model hasn't been trained yet.")

        explainer = shap.TreeExplainer(self.model)
        explanation = explainer(X)

        if save_dir is not None:
            output_dir = Path(save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            tag = prefix or self.model_type
            tag = tag.replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            plt.figure()
            shap.summary_plot(
                explanation.values,
                X,
                feature_names=feature_names,
                show=False,
                max_display=30,
            )
            summary_path = output_dir / f"shap_summary_{timestamp}_{tag}.{file_format}"
            plt.savefig(summary_path, format=file_format)
            plt.close()

            mean_abs = np.abs(explanation.values).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:30]

            df_top20 = pl.DataFrame(
                {
                    "feature": np.array(feature_names)[top_idx],
                    "mean_abs_shap": mean_abs[top_idx],
                }
            )

            df_top20_path = output_dir / f"shap_top20_{timestamp}_{tag}.csv"
            df_top20.write_csv(df_top20_path)

        return explanation


def evaluate_and_compare_models(
    data, target_col, models_dict, id_col="id", group_split=True, missingness_df=None
):
    """
    Evaluate multiple models and compare their performance.

    Parameters:
        data (pl.DataFrame): DataFrame containing features and target
        target_col (str): Name of the target column
        models_dict (dict): Dictionary of model objects to evaluate
        id_col (str): Name of the mission/case ID column
        group_split (bool): If True, splitting will respect mission/case boundaries
        missingness_df (pl.DataFrame): Optional DataFrame with [id_col, "missingness_rate"] for stratified evaluation

    Returns:
        pl.DataFrame: DataFrame with comparison metrics (including stratified if missingness_df provided)
    """
    results = []

    # Print information about the splitting approach
    if group_split:
        n_unique_cases = data.select(id_col).n_unique()
        print(f"Using mission/case-level splitting with {n_unique_cases} unique cases")
    else:
        print("Using standard row-level splitting (not respecting mission/case boundaries)")

    for model_name, model in models_dict.items():
        print(f"\n--- Training {model_name} model ---")
        try:
            _, metrics = model.train(data, target_col, id_col=id_col, group_split=group_split)
            metrics["model"] = model_name
            metrics["subset"] = "full"
            metrics["subset_size"] = data.n_unique(id_col)
            # Add prevalence for full dataset
            n_positive_full = data.filter(pl.col(target_col) == 1).n_unique(id_col)
            metrics["subset_positive_cases"] = n_positive_full
            metrics["subset_prevalence"] = n_positive_full / data.n_unique(id_col)
            results.append(metrics)

            # Stratified evaluation if missingness data provided
            if missingness_df is not None and model.test_ids is not None:
                median_threshold = missingness_df["missingness_rate"].median()

                # Split by missingness
                low_miss_ids = set(
                    missingness_df.filter(pl.col("missingness_rate") <= median_threshold)
                    .select(id_col)
                    .to_series()
                    .to_list()
                )
                high_miss_ids = set(
                    missingness_df.filter(pl.col("missingness_rate") > median_threshold)
                    .select(id_col)
                    .to_series()
                    .to_list()
                )

                # Filter to test set only
                test_ids = model.test_ids
                low_miss_test_ids = low_miss_ids & test_ids
                high_miss_test_ids = high_miss_ids & test_ids

                # Evaluate on low missingness test subset
                if len(low_miss_test_ids) > 50:
                    low_miss_data = data.filter(pl.col(id_col).is_in(list(low_miss_test_ids)))
                    n_positive = low_miss_data.filter(pl.col(target_col) == 1).n_unique(id_col)
                    if n_positive > 10:
                        print(
                            f"  Evaluating on LOW missingness subset (n_test={len(low_miss_test_ids)})"
                        )
                        low_metrics = model.evaluate_on_subset(
                            low_miss_data, target_col, id_col=id_col, subset_name="low_missingness"
                        )
                        low_metrics["model"] = model_name
                        low_metrics["subset"] = "low_missingness"
                        low_metrics["subset_size"] = len(low_miss_test_ids)
                        low_metrics["subset_positive_cases"] = n_positive
                        low_metrics["subset_prevalence"] = n_positive / len(low_miss_test_ids)
                        results.append(low_metrics)

                # Evaluate on high missingness test subset
                if len(high_miss_test_ids) > 50:
                    high_miss_data = data.filter(pl.col(id_col).is_in(list(high_miss_test_ids)))
                    n_positive = high_miss_data.filter(pl.col(target_col) == 1).n_unique(id_col)
                    if n_positive > 10:
                        print(
                            f"  Evaluating on HIGH missingness subset (n_test={len(high_miss_test_ids)})"
                        )
                        high_metrics = model.evaluate_on_subset(
                            high_miss_data,
                            target_col,
                            id_col=id_col,
                            subset_name="high_missingness",
                        )
                        high_metrics["model"] = model_name
                        high_metrics["subset"] = "high_missingness"
                        high_metrics["subset_size"] = len(high_miss_test_ids)
                        high_metrics["subset_positive_cases"] = n_positive
                        high_metrics["subset_prevalence"] = n_positive / len(high_miss_test_ids)
                        results.append(high_metrics)

        except Exception as e:
            print(f"Error training {model_name}: {e}")

    # Create comparison DataFrame
    if results:
        metrics_df = pl.from_dicts(results)

        print("\n--- Model Comparison ---")
        print(f"Target variable: {target_col}\n")
        print(metrics_df)

        return metrics_df

    return None
