"""Train the candidate models on clean.diabetes and log everything to MLflow.

Metric decisions (see README):
- Selection metric: PR-AUC on the validation split. Only 11% of encounters are
  readmitted within 30 days; PR-AUC focuses on that class and does not depend
  on a threshold. Accuracy is useless here (always "no" scores 0.89).
- Decision threshold: the one that maximizes F2 on validation, among those
  that flag at most MAX_FLAGGED_RATE (30%) of patients. A missed readmission
  (false negative) costs more than a false alarm, and F2 weights recall twice
  as much as precision; but unconstrained, max-F2 flagged ~54% of patients with
  this model. The cap reflects follow-up capacity: "we can follow up 3 in 10
  patients; which ones?"
- The test split is not touched here: compare.py uses it to judge the winner
  against the current champion.
"""

import logging
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from diabetes_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from trainer.common import EXPERIMENT_NAME, db_engine, load_split, step_input, write_result

log = logging.getLogger("trainer.train")

RANDOM_STATE = 42
MAX_FLAGGED_RATE = float(os.environ.get("MAX_FLAGGED_RATE", "0.30"))

# model type -> (factory, hyperparameter sets tried on every run)
CANDIDATES = {
    "logistic_regression": (
        lambda **p: LogisticRegression(class_weight="balanced", max_iter=2000, **p),
        [{"C": 0.1}, {"C": 1.0}],
    ),
    "hist_gradient_boosting": (
        lambda **p: HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE, **p),
        [{"learning_rate": 0.05, "max_leaf_nodes": 15}, {"learning_rate": 0.05, "max_leaf_nodes": 31},
         {"learning_rate": 0.1, "max_leaf_nodes": 15}, {"learning_rate": 0.1, "max_leaf_nodes": 31}],
    ),
}

# MLflow 3 saves sklearn models with skops, which refuses tree internals
# unless trusted. We train these models ourselves, so we trust exactly the
# type each family needs.
SKOPS_TRUSTED_TYPES = {
    "logistic_regression": [],
    "hist_gradient_boosting": ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
}


def make_pipeline(classifier) -> Pipeline:
    preprocessor = ColumnTransformer([
        # Categories seen fewer than 50 times are merged into one "infrequent"
        # column; unseen categories at inference go there too instead of failing.
        ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                      sparse_output=False), CATEGORICAL_FEATURES),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def capped_f2_threshold(y_true, scores, max_flagged_rate: float = MAX_FLAGGED_RATE) -> float:
    """Threshold with the best F2 among those flagging at most max_flagged_rate of rows."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    f2 = 5 * precision * recall / np.clip(4 * precision + recall, 1e-12, None)
    sorted_scores = np.sort(scores)
    flagged = 1 - np.searchsorted(sorted_scores, thresholds, side="left") / len(scores)  # share with score >= t
    f2[flagged > max_flagged_rate] = -1  # not allowed
    return float(thresholds[int(np.argmax(f2))])


def metrics_at(y_true, scores, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {"pr_auc": average_precision_score(y_true, scores), "roc_auc": roc_auc_score(y_true, scores),
            "precision": precision, "recall": recall, "f2": f2,
            "flagged_rate": float(predicted.mean())}


def log_plots(y_true, scores, threshold: float) -> None:
    with tempfile.TemporaryDirectory() as folder:
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax, name="validation")
        ax.axhline(y_true.mean(), linestyle="--", color="grey", label="random")
        ax.legend()
        fig.savefig(Path(folder) / "pr_curve_validation.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_true, (scores >= threshold).astype(int), ax=ax,
                                                display_labels=["no", "<30 days"], colorbar=False)
        ax.set_title(f"validation, threshold {threshold:.3f}")
        fig.savefig(Path(folder) / "confusion_matrix_validation.png", bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifacts(folder, artifact_path="plots")


def main() -> None:
    context = step_input()
    engine = db_engine()
    X_train, y_train = load_split(engine, "train")
    X_val, y_val = load_split(engine, "validation")
    log.info("train %d rows (%.1f%% positive), validation %d rows", len(y_train), 100 * y_train.mean(), len(y_val))

    mlflow.set_experiment(EXPERIMENT_NAME)
    results = []
    with mlflow.start_run(run_name=f"pipeline_{context.get('dag_run_id', 'manual')}") as parent:
        mlflow.set_tags({"dag_run_id": context.get("dag_run_id", "manual"),
                         "batches_loaded": str(context.get("batches_loaded", "")),
                         "selection_metric": "validation_pr_auc",
                         "threshold_rule": f"max F2 on validation, flagging <= {MAX_FLAGGED_RATE:.0%}"})
        mlflow.log_params({"train_rows": len(y_train), "validation_rows": len(y_val),
                           "train_positive_rate": round(float(y_train.mean()), 4),
                           "max_flagged_rate": MAX_FLAGGED_RATE})

        for model_type, (factory, param_sets) in CANDIDATES.items():
            for params in param_sets:
                name = model_type + "_" + "_".join(f"{k}={v}" for k, v in params.items())
                with mlflow.start_run(run_name=name, nested=True) as run:
                    model = make_pipeline(factory(**params)).fit(X_train, y_train)
                    train_scores = model.predict_proba(X_train)[:, 1]
                    val_scores = model.predict_proba(X_val)[:, 1]
                    threshold = capped_f2_threshold(y_val, val_scores)

                    mlflow.set_tag("model_type", model_type)
                    mlflow.log_params({**params, "decision_threshold": round(threshold, 6)})
                    mlflow.log_metrics({f"train_{k}": v for k, v in metrics_at(y_train, train_scores, threshold).items()})
                    val_metrics = metrics_at(y_val, val_scores, threshold)
                    mlflow.log_metrics({f"validation_{k}": v for k, v in val_metrics.items()})
                    log_plots(y_val, val_scores, threshold)
                    info = mlflow.sklearn.log_model(
                        model, name="model", input_example=X_train.head(5),
                        skops_trusted_types=SKOPS_TRUSTED_TYPES[model_type],
                    )
                results.append({"run_id": run.info.run_id, "run_name": name, "model_uri": info.model_uri,
                                "model_type": model_type, "threshold": threshold,
                                "validation_pr_auc": val_metrics["pr_auc"],
                                "validation_recall": val_metrics["recall"],
                                "validation_precision": val_metrics["precision"],
                                "validation_f2": val_metrics["f2"]})
                log.info("%-55s val PR-AUC %.3f  recall %.2f  precision %.2f  (threshold %.3f)",
                         name, val_metrics["pr_auc"], val_metrics["recall"], val_metrics["precision"], threshold)

        best = max(results, key=lambda r: r["validation_pr_auc"])
        mlflow.set_tag("best_run", best["run_name"])
        mlflow.log_metric("best_validation_pr_auc", best["validation_pr_auc"])

    write_result({**best, "parent_run_id": parent.info.run_id, "candidates": len(results)})


if __name__ == "__main__":
    main()
