"""Load test for the diabetes readmission API, from inside the cluster.

Each simulated user sends a random real encounter out of payloads.json (200
eligible rows drawn from Diabetes.csv with a fixed seed, all valid for the
API), waits 1–2.5 s, and repeats. A request counts as a success only if the
API answered 200 with a prediction of 0 or 1 and a probability in [0, 1].

This file must stay identical for the 1-replica and 3-replica tests.
"""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

PAYLOADS = json.loads((Path(__file__).parent / "payloads.json").read_text())


class ClinicianUser(HttpUser):
    # Pause between two requests of the same user. With ~1.75 s on average,
    # each user sends about 0.57 requests per second: users ≠ RPS.
    wait_time = between(1, 2.5)

    @task
    def predict(self):
        with self.client.post(
            "/predict", json=random.choice(PAYLOADS), name="/predict", catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                body = response.json()
                prediction, probability = body["prediction"], body["probability"]
            except (ValueError, KeyError):
                response.failure("Response is not JSON with prediction/probability")
                return
            if prediction not in (0, 1) or not 0 <= probability <= 1:
                response.failure(f"Invalid prediction: {body}")
                return
            response.success()
