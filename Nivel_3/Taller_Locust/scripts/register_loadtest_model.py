"""Point the @loadtest alias at the 50-tree random forest.

The workshop limits each API replica to 1 GB. Taller MLflow's @champion
(random forest, 200 trees, no depth limit, 372 MB file) is OOM-killed while
loading under that limit; the 50-tree version (macro-F1 0.843 vs 0.851) peaks
at ~553 MB. This registers that run as a version of `covertype` (once) and
gives it the @loadtest alias. @champion is left untouched.

Run it inside the API image, which already has MLflow installed:
    docker compose --profile setup run --rm register-model
"""

import os

from mlflow import MlflowClient

MODEL_NAME = os.environ.get("MODEL_NAME", "covertype")
ALIAS = os.environ.get("MODEL_ALIAS", "loadtest")
EXPERIMENT_NAME = "covertype_tuning"
RUN_NAME = "random_forest_n_estimators=50_max_depth=None"

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' not found: run Taller MLflow's notebook first.")

# Latest run with that name (each notebook execution creates one).
runs = client.search_runs(
    [experiment.experiment_id],
    filter_string=f"attributes.run_name = '{RUN_NAME}' AND attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if not runs:
    raise SystemExit(f"No finished run named '{RUN_NAME}'.")
run = runs[0]

# Idempotent: reuse the version if this run was already registered.
existing = client.search_model_versions(f"name = '{MODEL_NAME}' AND run_id = '{run.info.run_id}'")
if existing:
    version = existing[0].version
    print(f"Run already registered as {MODEL_NAME} v{version}.")
else:
    model_id = run.outputs.model_outputs[0].model_id
    version = client.create_model_version(
        MODEL_NAME, source=f"models:/{model_id}", run_id=run.info.run_id,
        tags={"macro_f1": f"{run.data.metrics['macro_f1']:.4f}", "purpose": "load test (fits 1 GB)"},
    ).version
    print(f"Registered {RUN_NAME} as {MODEL_NAME} v{version}.")

client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
print(f"@{ALIAS} -> {MODEL_NAME} v{version} (macro-F1 {run.data.metrics['macro_f1']:.4f})")
