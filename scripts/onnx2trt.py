#!/usr/bin/env python3
"""
Conversor ONNX -> TensorRT 8.6 para --backend trt (se ejecuta DENTRO de la imagen Docker
de FasterLivePortrait, ver scripts/build_trt_engines.sh). Basado en scripts/onnx2trt.py de
FasterLivePortrait (MIT), con dos cambios para que el runtime del host no dependa de cuDNN 8:

  * OnnxParserFlag.NATIVE_INSTANCENORM: InstanceNorm como capa nativa de TensorRT en vez del
    plugin InstanceNormalization_TRT (que llama a cuDNN 8, incluido libcudnn_ops_train).
  * Fuentes de tácticas sin CUDNN.
"""
import argparse
import ctypes
import os
import sys

import tensorrt as trt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx", required=True)
    ap.add_argument("-e", "--engine", required=True)
    ap.add_argument("-p", "--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--plugin", default=None, help="Plugin GridSample3D (.so)")
    ap.add_argument("--workspace-gb", type=float, default=8.0)
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    if args.plugin:
        ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    config.set_tactic_sources(
        (1 << int(trt.TacticSource.CUBLAS)) | (1 << int(trt.TacticSource.CUBLAS_LT))
        | (1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS))
        | (1 << int(trt.TacticSource.JIT_CONVOLUTIONS))
    )
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"[*] Construyendo {args.precision}: {os.path.basename(args.onnx)} -> {args.engine}")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        sys.exit(f"[!] Falló la construcción de {args.engine}")
    with open(args.engine, "wb") as f:
        f.write(plan)
    print(f"[+] {args.engine} ({os.path.getsize(args.engine) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
