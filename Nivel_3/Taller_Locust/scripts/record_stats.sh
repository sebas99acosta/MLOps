#!/usr/bin/env bash
# Records `docker stats` of every container on the VM while the load test runs.
#
#   sudo scripts/record_stats.sh rep1        # Ctrl+C to stop
#
# Writes resultados/docker_<label>.csv: unix timestamp, container, CPU %, memory.
# scripts/summarize.py (on the PC) matches these samples to each Locust level
# by time, so keep the VM and PC clocks in sync (compare `date` on both).
# With --cpus 1, a replica's CPU % tops out at 100.
set -u

label=${1:?usage: record_stats.sh <label>}
cd "$(dirname "$0")/.."
out="resultados/docker_${label}.csv"
[ -f "$out" ] || echo "timestamp,container,cpu_pct,mem_usage" > "$out"

# Run with sudo, but leave the file owned by the normal user.
trap '[ -n "${SUDO_UID:-}" ] && chown "$SUDO_UID:$SUDO_GID" "$out"; echo; echo "saved $out"' EXIT

echo "recording to $out (Ctrl+C to stop)"
while true; do
    now=$(date +%s)
    # --no-stream takes ~2 s to sample, so with the sleep: one sample every ~5 s.
    docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' \
        | sed "s/^/${now},/" >> "$out"
    sleep 3
done
