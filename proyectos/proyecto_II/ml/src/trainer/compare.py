"""Best model of this run (challenger) vs the current @champion.

Both are scored on the SAME rows: today's test split, which neither model
was trained or tuned on. The challenger wins only with a strictly higher
PR-AUC; with no champion yet, it wins by default.
"""

import logging

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import average_precision_score

from trainer.common import CHAMPION_ALIAS, MODEL_NAME, db_engine, load_split, step_input, write_result

log = logging.getLogger("trainer.compare")


def test_pr_auc(model_uri: str, X_test, y_test) -> float:
    model = mlflow.sklearn.load_model(model_uri)
    return float(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))


def main() -> None:
    best = step_input()  # train's result
    X_test, y_test = load_split(db_engine(), "test")
    challenger = test_pr_auc(best["model_uri"], X_test, y_test)

    try:
        champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        champion = None

    if champion is None:
        result = {"promote": True, "reason": "no champion yet",
                  "challenger_test_pr_auc": challenger, "champion_version": None, "champion_test_pr_auc": None}
    else:
        incumbent = test_pr_auc(f"models:/{MODEL_NAME}/{champion.version}", X_test, y_test)
        result = {"promote": challenger > incumbent,
                  "reason": f"challenger {challenger:.4f} vs champion v{champion.version} {incumbent:.4f} (test PR-AUC)",
                  "challenger_test_pr_auc": challenger, "champion_version": champion.version,
                  "champion_test_pr_auc": incumbent}

    log.info("test rows %d — %s -> %s", len(y_test), result["reason"], "PROMOTE" if result["promote"] else "keep champion")
    write_result({**result, "test_rows": len(y_test)})


if __name__ == "__main__":
    main()
