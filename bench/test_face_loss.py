#!/usr/bin/env python3
"""
Test funcional de pérdida/recuperación de rostro: vídeo -> frames negros -> vídeo.
Informa el salto máximo entre frames consecutivos de salida (media abs. de diferencia
por píxel) en cada transición. Un corte duro produce un pico muy superior al resto.
"""
import os, sys, time
import numpy as np, cv2
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_live_cam as r

video = os.path.join(r.CURRENT_DIR, "liveportrait_src/assets/examples/driving/d6.mp4")
cap = cv2.VideoCapture(video)
frames = [cap.read()[1] for _ in range(90)]
black = np.zeros_like(frames[0])
p = r.LivePortraitCamPipeline(r.find_default_source_image())

seq = [("face", f) for f in frames[:45]] + [("lost", black)] * 30 + [("face", f) for f in frames[45:90]]
outs, labels = [], []
for label, f in seq:
    o = p.process_frame(f)
    if o is None:  # comportamiento original: salta a la imagen fuente
        o = cv2.cvtColor(p.source_rgb, cv2.COLOR_RGB2BGR)
    outs.append(cv2.resize(o, (256, 256)).astype(np.float32))
    labels.append(label)
    time.sleep(1 / 30)  # ritmo de webcam real para que el dt/fade sea realista

d = [np.abs(outs[i] - outs[i - 1]).mean() for i in range(1, len(outs))]
steady = np.median(d[5:44])
loss = max(d[44:50]); recover = max(d[74:80])
print(f"diff mediana en tracking: {steady:.2f} | pico al perder: {loss:.2f} | pico al recuperar: {recover:.2f}")
