#!/usr/bin/env bash
# Run all four benchmarks and write every CSV row to results.csv.
# Usage: ./run_all.sh            (CPU + GPU)
#        ./run_all.sh --cpu-only (CPU timings only, no GPU needed)
set -euo pipefail
cd "$(dirname "$0")"

out=results.csv
echo "example,variant,size,time_ms,throughput,unit,error,check" > "$out"
for prog in vector_add matmul image_blur reduction; do
    echo "== $prog"
    # Show everything on screen; keep only the data rows for the CSV file.
    ./"$prog" "$@" | tee /dev/stderr | grep -v '^#' | grep -v '^example,' >> "$out"
done
echo "Wrote $out"
