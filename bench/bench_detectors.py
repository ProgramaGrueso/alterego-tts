#!/usr/bin/env python3
"""
Compara la latencia de (re)adquisición de rostro: InsightFace vs MediaPipe FaceLandmarker.
Mide detector + LandmarkRunner (lo que ocurre al perder/recuperar el rostro) sobre frames
de un vídeo, y la distancia media entre los landmarks finales de ambos caminos.
"""
import os, sys, time
import numpy as np, cv2
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_live_cam as r

video = os.path.join(r.CURRENT_DIR, "liveportrait_src/assets/examples/driving/d6.mp4")
cap = cv2.VideoCapture(video)
frames = [cap.read()[1] for _ in range(150)]
frames = [cv2.resize(f, (640, 480)) for f in frames]  # resolución típica de webcam

p = r.LivePortraitCamPipeline(r.find_default_source_image(), detector="mediapipe")
mp_det = p.mp_detector

def run(fn, n_warm=10):
    res, t = [], []
    for i, f in enumerate(frames):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter(); out = fn(f, rgb); dt = (time.perf_counter() - t0) * 1000
        if i >= n_warm:
            t.append(dt)
        res.append(out)
    return res, np.array(t)

p.mp_detector = None
lm_if, t_if = run(p._detect_landmarks)
p.mp_detector = mp_det
lm_mp, t_mp = run(p._detect_landmarks)
t_mp_only = run(lambda f, rgb: mp_det.detect(rgb))[1]

both = [(a, b) for a, b in zip(lm_if, lm_mp) if a is not None and b is not None]
dist = np.mean([np.linalg.norm(a - b, axis=1).mean() for a, b in both])
print(f"InsightFace + LandmarkRunner: media {t_if.mean():.2f} ms | p95 {np.percentile(t_if, 95):.2f} ms | detecciones {sum(x is not None for x in lm_if)}/{len(frames)}")
print(f"MediaPipe   + LandmarkRunner: media {t_mp.mean():.2f} ms | p95 {np.percentile(t_mp, 95):.2f} ms | detecciones {sum(x is not None for x in lm_mp)}/{len(frames)}")
print(f"MediaPipe solo (FaceLandmarker CPU): media {t_mp_only.mean():.2f} ms")
print(f"Distancia media entre landmarks finales: {dist:.2f} px (frames 640x480)")
