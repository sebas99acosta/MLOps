"""Register this run's best model as a new version and, if it won the
comparison, point @champion at it.

Every run registers a version (full history); only the alias decides what the
API serves. The decision threshold travels with the version as a tag, so the
API applies the threshold the model was tuned with, with no code change.
"""

import logging

import mlflow
from mlflow import MlflowClient

from trainer.common import CHAMPION_ALIAS, MODEL_NAME, step_input, write_result

log = logging.getLogger("trainer.promote")


def main() -> None:
    context = step_input()  # {"best": train's result, "decision": compare's result, "dag_run_id": ...}
    best, decision = context["best"], context["decision"]
    client = MlflowClient()

    version = mlflow.register_model(best["model_uri"], MODEL_NAME).version
    tags = {
        "run_name": best["run_name"],
        "model_type": best["model_type"],
        "decision_threshold": f"{best['threshold']:.6f}",
        "validation_pr_auc": f"{best['validation_pr_auc']:.4f}",
        "validation_recall": f"{best['validation_recall']:.4f}",
        "validation_precision": f"{best['validation_precision']:.4f}",
        "test_pr_auc": f"{decision['challenger_test_pr_auc']:.4f}",
        "dag_run_id": str(context.get("dag_run_id", "manual")),
        "promoted": str(decision["promote"]).lower(),
        "comparison": decision["reason"],
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, version, key, value)

    if decision["promote"]:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    log.info("%s v%s registered; %s", MODEL_NAME, version,
             f"@{CHAMPION_ALIAS} -> v{version}" if decision["promote"] else f"@{CHAMPION_ALIAS} unchanged")
    write_result({"model_name": MODEL_NAME, "version": version, "promoted": decision["promote"]})


if __name__ == "__main__":
    main()
