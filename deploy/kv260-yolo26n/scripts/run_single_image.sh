#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 MODEL.xmodel IMAGE [OUTPUT_DIR]" >&2
  exit 2
fi
script_dir="$(cd "$(dirname "$0")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
model="$1"
image="$2"
if [ "$#" -eq 3 ]; then
  output_dir="$3"
else
  output_dir="$root_dir/output/single_$(date +%Y%m%d_%H%M%S)"
fi
runner="$root_dir/build/kv260_yolo26n_runner"
test -x "$runner" || { echo "Build first: $script_dir/build_on_board.sh" >&2; exit 2; }
test -s "$model" || { echo "Model not found: $model" >&2; exit 2; }
test -s "$image" || { echo "Image not found: $image" >&2; exit 2; }
mkdir -p "$output_dir"
printf '%s\n' "$image" > "$output_dir/images.txt"

"$runner" "$model" "$output_dir/images.txt" "$output_dir/result.bin" 3 1 2 \
  | tee "$output_dir/console.json"
python3 "$script_dir/render_result.py" \
  "$output_dir/images.txt" "$output_dir/result.bin" \
  --frame-index 0 --output "$output_dir/preview.png"
echo "Preview: $output_dir/preview.png"
echo "Results: $output_dir"
