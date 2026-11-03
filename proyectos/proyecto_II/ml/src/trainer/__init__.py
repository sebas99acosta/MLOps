"""Heavy DAG steps, each run as its own pod by Airflow's KubernetesPodOperator:

    python -m trainer.preprocess   raw (valid rows) -> clean
    python -m trainer.train        train candidates, log to MLflow, pick the best
    python -m trainer.compare      best of this run vs @champion on the test split
    python -m trainer.promote      register the version, move @champion if it won

Inputs from earlier steps arrive as JSON in the STEP_INPUT environment
variable; each step's result is written as JSON to /airflow/xcom/return.json,
which Airflow turns into an XCom.
"""
