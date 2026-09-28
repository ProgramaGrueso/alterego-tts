#!/usr/bin/env python3
"""Compara las salidas del backend TensorRT contra PyTorch (mismas entradas) y mide su latencia."""
import os, sys, time
import torch
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_live_cam as r

def bench(fn, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000

pt = r.LivePortraitCamPipeline(r.find_default_source_image(), skin_noise=0)
w = pt.wrapper
res = {}
with torch.inference_mode():
    I = pt.I_s
    x_d = pt.x_s.clone(); x_d[..., 1] += 0.02
    for name in ("torch", "trt"):
        if name == "trt":
            import trt_backend
            trt_backend.install_trt_backend(w, pt.cropper, r.DEFAULT_TRT_DIR)
        res[name] = (w.get_kp_info(I), w.warp_decode(pt.f_s, pt.x_s, x_d)["out"].clone(), w.stitching(pt.x_s, x_d))
        print(f"[{name}] motion_extractor {bench(lambda: w.get_kp_info(I)):.2f} ms | "
              f"warp_decode {bench(lambda: w.warp_decode(pt.f_s, pt.x_s, x_d)):.2f} ms | "
              f"stitching {bench(lambda: w.stitching(pt.x_s, x_d)):.2f} ms")
(kp0, o0, s0), (kp1, o1, s1) = res["torch"], res["trt"]
for k in ["pitch", "yaw", "roll", "exp", "t", "scale", "kp"]:
    print(f"  {k:6s} max|diff| {float((kp1[k] - kp0[k]).abs().max()):.5f}")
print(f"  imagen warp_decode: mean|diff| {float((o1 - o0).abs().mean()) * 255:.2f}/255, max {float((o1 - o0).abs().max()) * 255:.1f}/255")
print(f"  stitching max|diff| {float((s1.float() - s0.float()).abs().max()):.5f}")

# Landmarks: LandmarkRunner (ONNX Runtime) vs motor TRT, con el mismo prior sobre frames reales
import cv2, numpy as np
from trt_backend import LandmarkSessionTRT
runner = pt.cropper.human_landmark_runner
cap = cv2.VideoCapture(os.path.join(r.CURRENT_DIR, "liveportrait_src/assets/examples/driving/d6.mp4"))
frames = [cv2.cvtColor(cv2.resize(cap.read()[1], (640, 480)), cv2.COLOR_BGR2RGB) for _ in range(30)]
pt.mp_detector = None
priors = [pt._detect_landmarks(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), f) for f in frames]
trt_session = runner.session
import onnxruntime
runner.session = onnxruntime.InferenceSession(pt.crop_cfg.landmark_ckpt_path, providers=["CUDAExecutionProvider"])
ref = [runner.run(f, p) for f, p in zip(frames, priors)]
t0 = time.perf_counter(); [runner.run(f, p) for f, p in zip(frames, priors)]; t_ort = (time.perf_counter() - t0) / 30 * 1000
runner.session = trt_session
out = [runner.run(f, p) for f, p in zip(frames, priors)]
t0 = time.perf_counter(); [runner.run(f, p) for f, p in zip(frames, priors)]; t_trt = (time.perf_counter() - t0) / 30 * 1000
d = np.mean([np.linalg.norm(a - b, axis=1).mean() for a, b in zip(ref, out)])
print(f"[landmark] ORT-CUDA {t_ort:.2f} ms | TRT {t_trt:.2f} ms | distancia media {d:.3f} px (640x480)")
