"""Load test for the covertype inference API (behind Nginx).

Each simulated user sends a random real row out of payloads.json (100 rows
drawn from covertype.csv with a fixed seed), waits 1–2.5 s, and repeats.
A request counts as a success only if the API answered 200 with valid JSON
and a cover_type that is an integer class from 0 to 6.

This file must stay identical for the 1-replica and 3-replica tests.
"""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

PAYLOADS = json.loads((Path(__file__).parent / "payloads.json").read_text())
VALID_CLASSES = range(7)


class CovertypeUser(HttpUser):
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
                cover_type = response.json()["cover_type"]
            except (ValueError, KeyError):
                response.failure("Response is not JSON with a 'cover_type' field")
                return
            if not isinstance(cover_type, int) or cover_type not in VALID_CLASSES:
                response.failure(f"cover_type out of range: {cover_type!r}")
                return
            response.success()
