#!/usr/bin/env bash
# Genera los motores TensorRT 8.6 usados por `run_live_cam.py --backend trt`
# dentro de la imagen Docker de FasterLivePortrait (TensorRT 8.6.1.6 + plugin GridSample3D).
#
# Requisitos: docker + nvidia-container-toolkit, y los ONNX de FasterLivePortrait:
#   .venv/bin/hf download warmshao/FasterLivePortrait --include "liveportrait_onnx/*" \
#       --local-dir liveportrait_src/pretrained_weights/faster_liveportrait
#
# Los motores quedan en liveportrait_src/pretrained_weights/faster_liveportrait/trt_engines/
# Son específicos de la GPU y de la versión de TensorRT: regenéralos si cambias de GPU.
set -euo pipefail

cd "$(dirname "$0")/.."
WEIGHTS="$PWD/liveportrait_src/pretrained_weights/faster_liveportrait"
ONNX_DIR="$WEIGHTS/liveportrait_onnx"
ENGINE_DIR="$WEIGHTS/trt_engines"
IMAGE="${FLP_IMAGE:-shaoguo/faster_liveportrait:v3}"

if [ ! -f "$ONNX_DIR/warping_spade-fix.onnx" ]; then
    echo "[!] Faltan los ONNX en $ONNX_DIR (ver cabecera de este script)." >&2
    exit 1
fi
mkdir -p "$ENGINE_DIR"

# Conversión con scripts/onnx2trt.py (derivado del de FasterLivePortrait, con InstanceNorm
# nativo y sin tácticas cuDNN). El plugin GridSample3D se copia junto a los motores.
docker run --rm --gpus=all \
    -v "$PWD/scripts":/scripts:ro \
    -v "$ONNX_DIR":/onnx:ro \
    -v "$ENGINE_DIR":/engines \
    "$IMAGE" bash -lc '
set -e
export PATH=/root/miniconda3/bin:$PATH
export LD_LIBRARY_PATH=/opt/TensorRT-8.6.1.6/lib:${LD_LIBRARY_PATH:-}
build() { python /scripts/onnx2trt.py --plugin /onnx/libgrid_sample_3d_plugin.so -o "/onnx/$1.onnx" -e "/engines/$1.trt" ${2:+-p $2}; }
build warping_spade-fix
build motion_extractor fp32
build stitching
build stitching_eye
build stitching_lip
build landmark
cp /onnx/libgrid_sample_3d_plugin.so /engines/
chown -R '"$(id -u):$(id -g)"' /engines
'
echo "[+] Motores TensorRT generados en $ENGINE_DIR"
ls -la "$ENGINE_DIR"
