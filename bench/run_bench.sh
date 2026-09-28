#!/usr/bin/env bash
# Uso: bench/run_bench.sh <etiqueta> [flags extra...]
# Ejecuta el benchmark estándar (30 s, sin cámara virtual ni preview) con un
# vídeo de conducción en bucle y guarda el resumen en bench/results/<etiqueta>.txt
set -e
cd "$(dirname "$0")/.."
label="$1"; shift
mkdir -p bench/results
VIDEO="${DRIVING_VIDEO:-liveportrait_src/assets/examples/driving/d6.mp4}"
.venv/bin/python run_live_cam.py --dry-run 30 --no-virtualcam --no-preview \
    --driving-video "$VIDEO" "$@" < /dev/null 2>&1 \
    | tee bench/results/"$label".log \
    | sed -n '/PERFORMANCE BENCHMARK SUMMARY/,$p' > bench/results/"$label".txt
cat bench/results/"$label".txt
