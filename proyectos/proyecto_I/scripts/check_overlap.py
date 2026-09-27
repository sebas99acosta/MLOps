"""Checks whether the Data API's batches really serve different data.

Fetches twice within the same batch window, then waits for a rotation and
fetches again, and prints how many rows each pair shares.

  - Two random 10% samples of the SAME pool share ~10% of their rows.
  - Samples from DIFFERENT pools share 0 rows.

So if "same batch" and "different batch" overlaps come out similar, the
batch number is only a label and every request samples the same pool.

Usage:
  python check_overlap.py                                  # local API, 30s rotation
  python check_overlap.py http://10.43.97.110:8080 310     # professor's API (spends 2 real windows!)
"""

import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8020"
WAIT = int(sys.argv[2]) if len(sys.argv) > 2 else 32  # must exceed MIN_UPDATE_TIME
GROUP = 5


def fetch():
    with urllib.request.urlopen(f"{URL}/data?group_number={GROUP}", timeout=60) as r:
        body = json.load(r)
    return body["batch_number"], set(map(tuple, body["data"]))


b1, s1 = fetch()
b1b, s1b = fetch()
print(f"fetch 1: batch {b1}, {len(s1)} rows")
print(f"fetch 2: batch {b1b}, {len(s1b)} rows  (same window)")

print(f"waiting {WAIT}s for the batch to rotate...")
time.sleep(WAIT)

b2, s2 = fetch()
print(f"fetch 3: batch {b2}, {len(s2)} rows  (after rotation)")

print()
print(f"shared rows, same batch      (fetch 1 vs 2): {len(s1 & s1b)}")
print(f"shared rows, different batch (fetch 1 vs 3): {len(s1 & s2)}")
