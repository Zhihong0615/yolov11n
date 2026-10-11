#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 MODEL.xmodel IMAGE_LIST OUTPUT_DIR [REPEATS]" >&2
  exit 2
fi
script_dir="$(cd "$(dirname "$0")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
runner="$root_dir/build/kv260_yolo26n_runner"
model="$1"
image_list="$2"
output_dir="$3"
repeats=5
if [ "$#" -eq 4 ]; then repeats="$4"; fi
test -x "$runner" || { echo "Build first: $script_dir/build_on_board.sh" >&2; exit 2; }
test -s "$model" || { echo "Model not found: $model" >&2; exit 2; }
test -s "$image_list" || { echo "Image list not found: $image_list" >&2; exit 2; }
case "$repeats" in ''|*[!0-9]*) echo 'REPEATS must be a positive integer' >&2; exit 2;; esac
if [ "$repeats" -lt 1 ] || [ "$repeats" -gt 20 ]; then
  echo 'REPEATS must be between 1 and 20' >&2
  exit 2
fi
mkdir -p "$output_dir"

for run in $(seq 1 "$repeats"); do
  run_dir="$output_dir/run_$(printf '%02d' "$run")"
  mkdir -p "$run_dir"
  echo "=== Repeat $run/$repeats: 6000 measured frames ==="
  "$runner" "$model" "$image_list" "$run_dir/result.bin" 200 6000 2 \
    | tee "$run_dir/console.json"
done
echo "Completed $repeats repeats; evidence: $output_dir"
