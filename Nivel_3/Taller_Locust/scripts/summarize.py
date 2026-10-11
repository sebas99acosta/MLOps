"""Summarize Locust CSV results against the pass/fail rule.

    python scripts/summarize.py            # every run in resultados/
    python scripts/summarize.py rep1_u     # only runs whose name starts with rep1_u

Rule: a level "passes" with < 1% failures and p95 < 500 ms. Each row is the
Aggregated line of <name>_stats.csv, which (thanks to --reset-stats) covers
only the window after all users were running.

If resultados/docker_<label>.csv exists (scripts/record_stats.sh on the VM),
the docker stats samples that fall inside each level's stable window are
added: average CPU per API replica, the highest memory of any replica, and
Nginx's CPU.
"""

import csv
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "resultados"
MAX_FAILURE_PCT = 1.0
MAX_P95_MS = 500
MEMORY_UNITS_MIB = {"B": 1 / 2**20, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}

prefix = sys.argv[1] if len(sys.argv) > 1 else ""


def run_key(path: Path):
    label, users = re.match(r"(.+)_u(\d+)_stats\.csv$", path.name).groups()
    return label, int(users)


def number(value: str):
    """Locust writes N/A when there is nothing to measure (e.g. no requests)."""
    return None if value in ("", "N/A") else float(value)


def show(value, digits=0):
    return "-" if value is None else f"{value:.{digits}f}"


def mib(usage: str) -> float:
    """'505.1MiB / 1GiB' -> 505.1"""
    amount, unit = re.match(r"([\d.]+)\s*([A-Za-z]+)", usage.split("/")[0].strip()).groups()
    return float(amount) * MEMORY_UNITS_MIB[unit]


def stable_window(name: str, users: int):
    """First and last timestamp at which all users were running."""
    history = RESULTS / f"{name}_stats_history.csv"
    if not history.exists():
        return None
    with history.open(newline="") as f:
        times = [int(r["Timestamp"]) for r in csv.DictReader(f)
                 if r["Name"] == "Aggregated" and r["User Count"] == str(users)]
    return (min(times), max(times)) if times else None


def docker_samples(label: str):
    path = RESULTS / f"docker_{label}.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [{"t": int(r["timestamp"]), "name": r["container"],
                 "cpu": float(r["cpu_pct"].rstrip("%")), "mem": mib(r["mem_usage"])}
                for r in csv.DictReader(f)]


def resources(samples, window):
    """Average CPU % per API replica, max memory (MiB) of any replica, Nginx CPU %."""
    if not samples or window is None:
        return None, None, None
    inside = [s for s in samples if window[0] <= s["t"] <= window[1]]
    api = [s for s in inside if "-api-" in s["name"]]
    proxy = [s for s in inside if "-proxy-" in s["name"]]
    if not api:
        return None, None, None
    return (sum(s["cpu"] for s in api) / len(api), max(s["mem"] for s in api),
            sum(s["cpu"] for s in proxy) / len(proxy) if proxy else None)


samples_by_label = {}
rows = []
for path in sorted(RESULTS.glob(f"{prefix}*_stats.csv"), key=run_key):
    label, users = run_key(path)
    name = f"{label}_u{users}"
    with path.open(newline="") as f:
        aggregated = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
    requests = int(aggregated["Request Count"])
    failures = int(aggregated["Failure Count"])
    failure_pct = 100 * failures / requests if requests else 100.0
    p95 = number(aggregated["95%"])
    log = RESULTS / f"{name}.log"
    generator_saturated = log.exists() and "CPU usage above" in log.read_text(errors="ignore")
    passes = requests > 0 and failure_pct < MAX_FAILURE_PCT and p95 is not None and p95 < MAX_P95_MS

    if label not in samples_by_label:
        samples_by_label[label] = docker_samples(label)
    api_cpu, api_mem, proxy_cpu = resources(samples_by_label[label], stable_window(name, users))

    rows.append([
        label, users, show(number(aggregated["Requests/s"]), 1), requests, f"{failure_pct:.2f}",
        show(number(aggregated["50%"])), show(p95), show(number(aggregated["99%"])),
        show(api_cpu), show(api_mem), show(proxy_cpu),
        "yes" if generator_saturated else "no", "PASS" if passes else "FAIL",
    ])

header = ["config", "users", "RPS", "requests", "failures %", "p50 ms", "p95 ms", "p99 ms",
          "API CPU %/replica", "API max MiB", "Nginx CPU %", "Locust CPU >90%", "passes"]
print("| " + " | ".join(header) + " |")
print("|" + "|".join("---" for _ in header) + "|")
for row in rows:
    print("| " + " | ".join(str(value) for value in row) + " |")
