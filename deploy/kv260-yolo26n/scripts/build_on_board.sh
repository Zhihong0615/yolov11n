#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
build_dir="$root_dir/build"
mkdir -p "$build_dir"

command -v g++ >/dev/null || { echo 'Missing aarch64 g++' >&2; exit 2; }
command -v pkg-config >/dev/null || { echo 'Missing pkg-config' >&2; exit 2; }
pkg-config --exists opencv4 || { echo 'OpenCV 4 development files are unavailable through pkg-config' >&2; exit 2; }
test -f /usr/include/vart/runner.hpp || {
  echo 'VART headers were not found at /usr/include/vart/runner.hpp' >&2
  exit 2
}

read -r -a opencv_cflags <<< "$(pkg-config --cflags opencv4)"
read -r -a opencv_libs <<< "$(pkg-config --libs opencv4)"
g++ -O3 -std=c++17 -pthread \
  "${opencv_cflags[@]}" \
  "$root_dir/src/bench_cpp_dual_runner_pipeline.cpp" \
  "$root_dir/src/fast_letterbox.cpp" \
  "$root_dir/src/fast_decode_six.cpp" \
  "${opencv_libs[@]}" \
  -lvart-runner -lvart-util -lxir \
  -o "$build_dir/kv260_yolo26n_runner"

echo "Built: $build_dir/kv260_yolo26n_runner"
