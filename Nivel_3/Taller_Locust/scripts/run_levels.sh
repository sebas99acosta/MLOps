#!/usr/bin/env bash
# Runs Locust headless once per user level, from the PC, against Nginx on the VM.
#
#   scripts/run_levels.sh <label> <users> [<users> ...]
#   scripts/run_levels.sh rep1 1 25 50 100 150 200
#
# Every level uses the same locustfile, spawn rate and duration, so 1 and 3
# replicas are compared under identical conditions. Results go to
# resultados/<label>_u<users>_*.csv plus a .log per level; then
# scripts/summarize.py prints the table.
#
# Overridable: TARGET_HOST (default the VM), SPAWN_RATE (10), RUN_TIME (3m),
# PAUSE (20 s between levels, so queues drain before the next one).
set -u

label=${1:?usage: run_levels.sh <label> <users> [<users> ...]}
shift
[ $# -gt 0 ] || { echo "give at least one user level"; exit 1; }

TARGET_HOST=${TARGET_HOST:-http://10.43.97.90:8000}
SPAWN_RATE=${SPAWN_RATE:-10}
RUN_TIME=${RUN_TIME:-3m}
PAUSE=${PAUSE:-20}

cd "$(dirname "$0")/.."
# Git Bash on Windows rewrites paths like /resultados; this stops it.
export MSYS_NO_PATHCONV=1

docker compose -f locust/docker-compose.yaml build --quiet

for users in "$@"; do
    name="${label}_u${users}"
    echo "=== ${name}: ${users} users, spawn ${SPAWN_RATE}/s, ${RUN_TIME} -> ${TARGET_HOST}"
    # --reset-stats: statistics restart once all users are running, so the
    # ramp-up is excluded. A non-zero exit only means there were failures.
    docker compose -f locust/docker-compose.yaml run --rm --no-deps -T locust \
        --headless --host "$TARGET_HOST" \
        --users "$users" --spawn-rate "$SPAWN_RATE" --run-time "$RUN_TIME" \
        --reset-stats --only-summary \
        --csv "/resultados/${name}" \
        > "resultados/${name}.log" 2>&1
    python scripts/summarize.py "$name"
    if grep -q "CPU usage above" "resultados/${name}.log"; then
        echo "!!! Locust's CPU was saturated: this level measures the generator, not the API."
    fi
    [ "$users" = "${*: -1}" ] || sleep "$PAUSE"
done

echo
python scripts/summarize.py "${label}_u"
