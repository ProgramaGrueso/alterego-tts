#!/usr/bin/env python3
"""
run_live_cam.py - Real-Time LivePortrait Face Animation to Virtual Camera
Accelerated by NVIDIA RTX 4070 Ti Super (Ada Lovelace / sm_89) via PyTorch & pyvirtualcam.
"""

import os
import sys

# Ensure reliable GUI window rendering on Wayland / KDE via XWayland
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
import signal
import argparse
import threading
import queue
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Ensure liveportrait_src is discoverable in PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "liveportrait_src")
DEFAULT_TRT_DIR = os.path.join(SRC_DIR, "pretrained_weights/faster_liveportrait/trt_engines")
DEFAULT_MEDIAPIPE_MODEL = os.path.join(SRC_DIR, "pretrained_weights/mediapipe/face_landmarker.task")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix
from src.utils.crop import paste_back, prepare_paste_back, crop_image, _transform_img
from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio


class OneEuroFilter:
    """
    1€ (One-Euro) Filter: Industry gold standard for real-time tracking stabilization.
    Adaptive low-pass filter that eliminates micro-jitter when still and eliminates lag when moving.
    """
    def __init__(self, min_cutoff=0.6, beta=0.03, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x, dt=1.0/30.0):
        if self.x_prev is None:
            self.x_prev = x.clone() if isinstance(x, torch.Tensor) else x.copy()
            self.dx_prev = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
            return self.x_prev

        # 1. Estimate filtered derivative (velocity)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # 2. Adaptive cutoff frequency based on velocity magnitude
        speed = torch.abs(dx_hat) if isinstance(dx_hat, torch.Tensor) else np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        tau = 1.0 / (2.0 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)

        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def soft_deadband(x: torch.Tensor, threshold: float, floor_gain: float) -> torch.Tensor:
    """
    Deadband continuo: la ganancia sube de `floor_gain` (en 0) a 1 (en |x| >= threshold)
    siguiendo un smoothstep, sin el escalón que producía torch.where en el umbral.
    """
    u = torch.clamp(torch.abs(x) / threshold, 0.0, 1.0)
    return x * (floor_gain + (1.0 - floor_gain) * (u * u * (3.0 - 2.0 * u)))


class MediaPipeFaceDetector:
    """
    Detector alternativo a InsightFace (--detector mediapipe): FaceLandmarker de MediaPipe
    (478 puntos, CPU). Los puntos solo se usan como prior para el recorte del LandmarkRunner,
    igual que los 106 puntos de InsightFace, así que el seguimiento posterior no cambia.
    """
    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo MediaPipe no encontrado: {model_path} (ver README)")
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, frame_rgb: np.ndarray):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        res = self.landmarker.detect(img)
        if not res.face_landmarks:
            return None
        h, w = frame_rgb.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)


class LivePortraitCamPipeline:
    def __init__(self, source_image_path: str, device_id: int = 0, flag_pasteback: bool = False, flag_compile: bool = False, driving_multiplier: float = 0.50, skin_noise: float = 2.5, flag_fp32_generator: bool = False, flag_lip_retargeting: bool = True, lip_multiplier: float = 1.00, flag_seamless: bool = False, output_size=(640, 480), backend: str = "torch", detector: str = "insightface", trt_engine_dir: str = None, gaze_multiplier: float = 0.0, flag_wink: bool = False):
        self.device_id = device_id
        self.flag_pasteback = flag_pasteback
        self.flag_seamless = flag_seamless
        self.output_size = output_size
        self.driving_multiplier = driving_multiplier
        self.skin_noise = float(skin_noise)
        self.flag_fp32_generator = flag_fp32_generator
        self.flag_lip_retargeting = flag_lip_retargeting
        self.gaze_multiplier = float(gaze_multiplier)
        self.flag_wink = flag_wink
        self.lip_multiplier = float(lip_multiplier)

        # Facial keypoint indices in LivePortrait latent expression space
        self.LIP_INDICES = [6, 12, 14, 17, 19, 20]
        self.FACE_INDICES = [i for i in range(21) if i not in self.LIP_INDICES]
        self.device = f"cuda:{device_id}"

        # Duración del fade a pose neutra al perder el rostro (y del retorno al recuperarlo)
        self.FADE_S = 0.30
        # Límites del dt real entre frames que se pasa a los filtros One-Euro
        self.DT_MIN, self.DT_MAX = 1.0 / 120.0, 1.0 / 10.0

        # Anti-jitter One-Euro filters tailored for each facial/pose component
        self.smoother_lmk = OneEuroFilter(min_cutoff=0.8, beta=0.02)
        self.smoother_angles = OneEuroFilter(min_cutoff=0.3, beta=0.02)
        # Separate filters: smooth for facial resting expressions, fast/reactive for lip speech articulation
        self.smoother_exp_face = OneEuroFilter(min_cutoff=0.4, beta=0.025)
        self.smoother_exp_lip = OneEuroFilter(min_cutoff=1.0, beta=0.18)
        self.smoother_t = OneEuroFilter(min_cutoff=0.45, beta=0.025)
        self.smoother_scale = OneEuroFilter(min_cutoff=0.45, beta=0.02)
        self.smoother_eye = OneEuroFilter(min_cutoff=2.5, beta=0.10)
        self.smoother_lip_seal = OneEuroFilter(min_cutoff=1.0, beta=0.05)
        self.smoother_gaze = OneEuroFilter(min_cutoff=1.5, beta=0.5)

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real-time LivePortrait animation.")

        torch.backends.cudnn.benchmark = True
        print(f"[*] Initializing LivePortrait on GPU: {torch.cuda.get_device_name(device_id)}...")

        # 1. Configs
        self.inf_cfg = InferenceConfig()
        self.inf_cfg.device_id = device_id
        self.inf_cfg.flag_use_half_precision = True
        self.inf_cfg.flag_relative_motion = True
        self.inf_cfg.flag_stitching = True
        self.inf_cfg.flag_eye_retargeting = True
        self.inf_cfg.flag_lip_retargeting = self.flag_lip_retargeting
        self.inf_cfg.flag_do_torch_compile = flag_compile

        self.crop_cfg = CropConfig()
        self.crop_cfg.device_id = device_id
        self.crop_cfg.insightface_root = os.path.join(SRC_DIR, "pretrained_weights/insightface")
        self.crop_cfg.landmark_ckpt_path = os.path.join(SRC_DIR, "pretrained_weights/liveportrait/landmark.onnx")

        # 2. Instantiate wrapper and cropper
        self.wrapper = LivePortraitWrapper(inference_cfg=self.inf_cfg)
        self.cropper = Cropper(crop_cfg=self.crop_cfg)

        # Convert core generators to half precision for Ada Lovelace tensor cores
        self.wrapper.warping_module.half()
        if not self.flag_fp32_generator:
            self.wrapper.spade_generator.half()
        else:
            print("[*] Diagnostic: spade_generator retained in FP32 precision.")
        self.wrapper.motion_extractor.half()
        if self.wrapper.stitching_retargeting_module:
            for k in self.wrapper.stitching_retargeting_module:
                self.wrapper.stitching_retargeting_module[k].half()

        # 3. Pre-process static source avatar (One-Time Execution)
        if not os.path.isabs(source_image_path):
            if os.path.exists(source_image_path):
                source_image_path = os.path.abspath(source_image_path)
            elif os.path.exists(os.path.join(CURRENT_DIR, source_image_path)):
                source_image_path = os.path.join(CURRENT_DIR, source_image_path)

        print(f"[*] Pre-processing source avatar from: {source_image_path}...")
        if not os.path.exists(source_image_path):
            raise FileNotFoundError(f"Source avatar image not found at: {source_image_path}")

        img_bgr = cv2.imread(source_image_path)
        if img_bgr is None:
            raise ValueError(f"Failed to load image from: {source_image_path}")

        self.source_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crop_info = self.cropper.crop_source_image(self.source_rgb, self.crop_cfg)
        if crop_info is None:
            raise ValueError("No face detected in the source avatar image. Please use a clear portrait.")

        self.img_crop_256x256 = crop_info["img_crop_256x256"]
        self.source_lmk = crop_info["lmk_crop"]
        self.M_c2o = crop_info["M_c2o"]

        # Precompute 3D features and canonical keypoints on GPU
        with torch.inference_mode():
            self.I_s = self.wrapper.prepare_source(self.img_crop_256x256).half()
            self.x_s_info = self.wrapper.get_kp_info(self.I_s)
            self.x_c_s = self.x_s_info["kp"].half()
            self.R_s = get_rotation_matrix(self.x_s_info["pitch"], self.x_s_info["yaw"], self.x_s_info["roll"]).half()
            self.f_s = self.wrapper.extract_feature_3d(self.I_s).half()
            self.x_s = self.wrapper.transform_keypoint(self.x_s_info).half()

        # Precompute source eye ratio for blinking retargeting
        self.c_s_eyes = calc_eye_close_ratio(self.source_lmk[None])
        self.c_s_eyes_tensor = torch.from_numpy(self.c_s_eyes).half().to(self.device)
        self.c_s_eye_mean = float(self.c_s_eyes.mean())
        self.c_d_eye_0 = None
        self.c_d_lip_0 = None

        # Precompute source lip ratio for lip retargeting
        self.c_s_lip = calc_lip_close_ratio(self.source_lmk[None])
        self.c_s_lip_tensor = torch.from_numpy(self.c_s_lip).half().to(self.device)

        # Buffers preasignados para los ratios de ojos/labios: [fuente | conducción].
        # La columna de conducción se rellena por frame con fill_() (sin torch.tensor().to()).
        n_eye = self.c_s_eyes_tensor.shape[1]
        self.eye_ratio_buf = torch.empty((1, n_eye + 1), dtype=torch.float16, device=self.device)
        self.eye_ratio_buf[:, :n_eye] = self.c_s_eyes_tensor
        n_lip = self.c_s_lip_tensor.shape[1]
        self.lip_ratio_buf = torch.empty((1, n_lip + 1), dtype=torch.float16, device=self.device)
        self.lip_ratio_buf[:, :n_lip] = self.c_s_lip_tensor

        # Máscara de keypoints labiales para fusionar expresiones sin bucles Python
        num_kp = self.x_s.shape[1]
        self.lip_mask = torch.zeros((1, num_kp, 1), dtype=torch.bool, device=self.device)
        self.lip_mask[:, self.LIP_INDICES] = True

        # --wink: el retargeting de ojos mueve los dos ojos a la vez; su delta se separa por
        # lado según la x del keypoint fuente (x<0 = ojo izquierdo de la imagen = landmarks 0:24).
        self.eye_side_masks = ((self.x_s[..., 0:1] < 0).half(), (self.x_s[..., 0:1] >= 0).half())
        self.c_d_eyes_0 = None
        self.smoother_eyes = OneEuroFilter(min_cutoff=2.5, beta=0.10)

        # --gaze-multiplier: delta de mirada sobre los keypoints de los globos oculares
        self.gaze_delta = torch.zeros_like(self.x_s_info["exp"]).half()
        self.gaze_0 = None

        # Textura de grano de piel generada una sola vez (se crea al conocer la resolución
        # de salida); cada frame toma un recorte con offset aleatorio para que no quede fija.
        self.NOISE_PAD = 64
        self.noise_tex = None

        # Precompute pasteback mask and seamlessClone ROI if requested
        self.mask_ori_float = None
        self.paste_bbox = None
        self.paste_center = None
        self.mask_roi = None
        if self.flag_pasteback and self.flag_seamless:
            h, w = self.source_rgb.shape[:2]
            self.mask_ori_float = prepare_paste_back(
                self.inf_cfg.mask_crop, self.M_c2o, dsize=(w, h)
            )
            mask_u8 = (self.mask_ori_float[..., 0] * 255).astype(np.uint8)
            bx, by, bw, bh = cv2.boundingRect(mask_u8)
            self.paste_bbox = (bx, by, bw, bh)
            self.paste_center = (bx + bw // 2, by + bh // 2)
            self.mask_roi = mask_u8[by:by+bh, bx:bx+bw]
        elif self.flag_pasteback:
            self._prepare_gpu_pasteback()

        print("[+] Source avatar features pre-computed and cached in VRAM.")

        # Backend TensorRT opcional: sustituye los módulos del wrapper por motores TRT
        self.backend = backend
        if backend == "trt":
            from trt_backend import install_trt_backend
            install_trt_backend(self.wrapper, self.cropper, trt_engine_dir or DEFAULT_TRT_DIR)

        # Detector de rostro para (re)adquisición
        self.detector = detector
        self.mp_detector = MediaPipeFaceDetector(DEFAULT_MEDIAPIPE_MODEL) if detector == "mediapipe" else None

        # Reference driving motion cache (calibrated on initial detection)
        self.x_d_0_info = None
        self.R_d_0 = None
        self.last_lmk = None
        self.is_tracking = False

        # Estado temporal: dt real, pérdida/recuperación de rostro con fade
        self.t_prev = None
        self.last_kp_out = None      # últimos keypoints renderizados
        self.lost_since = None       # instante en que se perdió el rostro
        self.kp_lost_from = None     # keypoints al perderlo (origen del fade a neutral)
        self.neutral_frame = None    # frame neutral cacheado tras completar el fade
        self.recover_since = None    # instante de re-adquisición
        self.kp_recover_from = None  # keypoints mostrados al re-adquirir
        self.reacquired = False

    def _prepare_gpu_pasteback(self, feather_sigma: float = 8.0):
        """
        Pasteback en GPU: como M_c2o es fijo, se precalculan una vez la rejilla de muestreo
        (ROI del fondo -> coordenadas del recorte), la máscara con feather y el fondo.
        Se compone a la resolución de salida: el avatar puede medir 2048 px pero se emite
        a 640x480, así que mezclar a resolución completa es trabajo desperdiciado.
        """
        h, w = self.source_rgb.shape[:2]
        out_w, out_h = self.output_size
        s = min(1.0, max(out_w / w, out_h / h))
        work_w, work_h = max(1, round(w * s)), max(1, round(h * s))
        self.paste_bg_bgr = cv2.resize(cv2.cvtColor(self.source_rgb, cv2.COLOR_RGB2BGR),
                                       (work_w, work_h), interpolation=cv2.INTER_AREA)

        crop_size = self.inf_cfg.mask_crop.shape[0]
        M = np.diag([s, s, 1.0]) @ np.vstack([self.M_c2o[:2], [0.0, 0.0, 1.0]])
        mask_crop = self.inf_cfg.mask_crop[..., 0].astype(np.float32) / 255.0
        mask_crop = cv2.GaussianBlur(mask_crop, (0, 0), feather_sigma)
        mask_work = cv2.warpAffine(mask_crop, M[:2], (work_w, work_h), flags=cv2.INTER_LINEAR)
        bx, by, bw, bh = cv2.boundingRect((mask_work > 1e-3).astype(np.uint8))
        self.paste_bbox = (bx, by, bw, bh)

        # Coordenadas (centros de píxel) del ROI llevadas al espacio del recorte, normalizadas
        # para grid_sample con align_corners=False (misma convención que cv2.warpAffine)
        M_inv = np.linalg.inv(M)
        xs, ys = np.meshgrid(np.arange(bx, bx + bw, dtype=np.float64), np.arange(by, by + bh, dtype=np.float64))
        cx = M_inv[0, 0] * xs + M_inv[0, 1] * ys + M_inv[0, 2]
        cy = M_inv[1, 0] * xs + M_inv[1, 1] * ys + M_inv[1, 2]
        grid = np.stack([(2 * cx + 1) / crop_size - 1, (2 * cy + 1) / crop_size - 1], axis=-1)
        self.paste_grid = torch.from_numpy(grid[None].astype(np.float32)).to(self.device)
        self.paste_mask = torch.from_numpy(mask_work[by:by+bh, bx:bx+bw][None, None].copy()).to(self.device)
        bg_roi = self.paste_bg_bgr[by:by+bh, bx:bx+bw]
        self.paste_bg_roi = torch.from_numpy(bg_roi.copy()).permute(2, 0, 1)[None].float().to(self.device)

    def _reset_smoothers(self):
        for f in (self.smoother_lmk, self.smoother_angles, self.smoother_exp_face,
                  self.smoother_exp_lip, self.smoother_t, self.smoother_scale,
                  self.smoother_eye, self.smoother_lip_seal, self.smoother_gaze):
            f.reset()

    def calibrate_neutral_pose(self, x_d_info, lmk):
        """Calibrates neutral expression, eye openness, lip baseline, and head pose from driving webcam."""
        self.x_d_0_info = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in x_d_info.items()}
        r_eyes = calc_eye_close_ratio(lmk[None])
        self.c_d_eye_0 = max(float(r_eyes.mean()), 0.15)
        r_lip = calc_lip_close_ratio(lmk[None])
        self.c_d_lip_0 = max(float(r_lip[0, 0]), 0.02)
        self.c_d_eyes_0 = np.maximum(r_eyes[0], 0.15)
        self.gaze_0 = calc_gaze_offset(lmk)
        self._reset_smoothers()
        print(f"[+] Pose neutra calibrada (Sensibilidad cabeza: {self.driving_multiplier:.2f} | Labios: {self.lip_multiplier:.2f} | Ojos: {self.c_d_eye_0:.2f} | Boca reposo: {self.c_d_lip_0:.2f}).")

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Executes LivePortrait deformation on one driving frame.
        Returns animated output frame (BGR, uint8), or None if nothing has been animated yet.
        """
        now = time.perf_counter()
        dt = 1.0 / 30.0 if self.t_prev is None else min(max(now - self.t_prev, self.DT_MIN), self.DT_MAX)
        self.t_prev = now

        with torch.inference_mode():
            kp_tracked = self._drive_keypoints(frame_bgr, dt)
            if kp_tracked is None:
                self.is_tracking = False
                return self._render_face_lost(now)

            self.is_tracking = True
            # Re-adquisición (tras pérdida o recalibración): mezclar desde lo que se mostraba
            if self.reacquired and self.last_kp_out is not None:
                self.kp_recover_from = self.last_kp_out
                self.recover_since = now
            self.reacquired = False
            self.lost_since = None

            kp = kp_tracked
            if self.recover_since is not None:
                w = smoothstep((now - self.recover_since) / self.FADE_S)
                kp = torch.lerp(self.kp_recover_from, kp_tracked, w)
                if w >= 1.0:
                    self.recover_since = None
            return self._render(kp)

    def _render_face_lost(self, now: float):
        """Mantiene el último frame animado y hace un fade suave hacia la pose neutra."""
        if self.last_kp_out is None:
            return None  # Aún no se animó nada: el llamador muestra la imagen fuente
        if self.lost_since is None:
            self.lost_since = now
            self.kp_lost_from = self.last_kp_out
            self.recover_since = None
        w = smoothstep((now - self.lost_since) / self.FADE_S)
        if w >= 1.0 and self.neutral_frame is not None:
            self.last_kp_out = self.x_s
            return self.neutral_frame
        frame = self._render(torch.lerp(self.kp_lost_from, self.x_s, w))
        if w >= 1.0:
            self.neutral_frame = frame
        return frame

    def _detect_landmarks(self, frame_bgr, frame_rgb):
        if self.mp_detector is not None:
            prior = self.mp_detector.detect(frame_rgb)
            return None if prior is None else self.cropper.human_landmark_runner.run(frame_rgb, prior)
        src_face = self.cropper.face_analysis_wrapper.get(
            frame_bgr, flag_do_landmark_2d_106=True, direction="large-small"
        )
        if len(src_face) == 0:
            return None
        return self.cropper.human_landmark_runner.run(frame_rgb, src_face[0].landmark_2d_106)

    def _drive_keypoints(self, frame_bgr: np.ndarray, dt: float):
        """Seguimiento facial + cálculo de keypoints de conducción. None si no hay rostro."""
        # Safety check: if webcam is covered / pitch dark, report no tracking
        if frame_bgr.mean() < 12.0:
            self.last_lmk = None
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. Face tracking / alignment
        if self.last_lmk is None:
            # First frame or re-acquisition: run detector
            lmk = self._detect_landmarks(frame_bgr, frame_rgb)
            if lmk is None:
                return None
            self.reacquired = True
            if self.lost_since is not None:
                # Estado de filtros obsoleto tras la pérdida: el fade de retorno cubre la continuidad
                self._reset_smoothers()
        else:
            # Tracking mode: pass previous landmark as prior (fast ONNX execution ~2ms)
            lmk = self.cropper.human_landmark_runner.run(frame_rgb, self.last_lmk)
            if lmk is None or lmk.min() < -50 or lmk.max() > max(frame_bgr.shape[:2]) + 50:
                # Re-detect if tracking was lost or landmark collapsed
                lmk = self._detect_landmarks(frame_bgr, frame_rgb)
                if lmk is None:
                    self.last_lmk = None
                    return None
        self.last_lmk = lmk

        # Apply temporal smoothing to facial landmarks to eliminate crop jitter
        lmk = self.smoother_lmk.update(lmk, dt)

        # 2. Crop 256x256 driving face
        ret_crop = crop_image(
            frame_rgb,
            lmk,
            dsize=self.crop_cfg.dsize,
            scale=self.crop_cfg.scale,
            vx_ratio=self.crop_cfg.vx_ratio,
            vy_ratio=self.crop_cfg.vy_ratio,
            flag_do_rot=self.crop_cfg.flag_do_rot,
        )
        if ret_crop["img_crop"].mean() < 10.0:
            self.last_lmk = None
            return None

        img_driving_crop_256 = cv2.resize(ret_crop["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)

        # 3. Driving keypoints inference
        I_d = self.wrapper.prepare_source(img_driving_crop_256).half()
        x_d_i_info = self.wrapper.get_kp_info(I_d)

        # Calibrate on first valid face
        if self.x_d_0_info is None:
            self.calibrate_neutral_pose(x_d_i_info, lmk)

        # 4. Relative Head Rotation (pitch, yaw, roll in degrees)
        delta_pitch = x_d_i_info["pitch"] - self.x_d_0_info["pitch"]
        delta_yaw = x_d_i_info["yaw"] - self.x_d_0_info["yaw"]
        delta_roll = x_d_i_info["roll"] - self.x_d_0_info["roll"]
        angles_raw = torch.cat([delta_pitch, delta_yaw, delta_roll], dim=-1)

        # Deadband continuo: micro-movimientos < 0.30° se atenúan (postura calmada en reposo)
        angles_raw = soft_deadband(angles_raw, 0.30, 0.25)
        angles_smooth = self.smoother_angles.update(angles_raw, dt)

        # Apply motion multiplier to rotation
        angles_damped = angles_smooth * self.driving_multiplier
        pitch_new = self.x_s_info["pitch"] + angles_damped[:, 0:1]
        yaw_new = self.x_s_info["yaw"] + angles_damped[:, 1:2]
        roll_new = self.x_s_info["roll"] + angles_damped[:, 2:3]
        R_new = get_rotation_matrix(pitch_new, yaw_new, roll_new).half()

        # 5. Expression delta (Capa 1: Lipsync reactivo desacoplado para labios + estabilidad facial)
        delta_raw = x_d_i_info["exp"] - self.x_d_0_info["exp"]

        # Región facial (cejas, pómulos): deadband suave anti-jitter y filtro One-Euro suave
        delta_face_smooth = self.smoother_exp_face.update(soft_deadband(delta_raw, 0.0045, 0.25), dt)

        # Región labial (articulación vocal / fonemas): deadband anti-muecas en reposo + One-Euro adaptativo
        delta_lip_smooth = self.smoother_exp_lip.update(soft_deadband(delta_raw, 0.0055, 0.20), dt)

        # Fusión de componentes:
        # - Resto del rostro escala con driving_multiplier (control natural de expresiones secundarias)
        # - Labios escalan con lip_multiplier (articulación nítida e independiente de la cabeza)
        delta_combined = torch.where(self.lip_mask,
                                     delta_lip_smooth * self.lip_multiplier,
                                     delta_face_smooth * self.driving_multiplier)

        delta_new = self.x_s_info["exp"] + delta_combined
        if self.gaze_multiplier > 0.0:
            delta_new = delta_new + self._gaze_delta(lmk, dt)

        # 6. Translation and scale (smoothed and controlled)
        t_raw = x_d_i_info["t"] - self.x_d_0_info["t"]
        t_smooth = self.smoother_t.update(t_raw, dt)
        t_new = self.x_s_info["t"] + t_smooth * self.driving_multiplier
        t_new[..., 2].fill_(0)

        scale_raw = (x_d_i_info["scale"] / self.x_d_0_info["scale"] - 1.0) * self.driving_multiplier + 1.0
        scale_smooth = self.smoother_scale.update(scale_raw, dt)
        scale_new = self.x_s_info["scale"] * scale_smooth

        x_d_i_new = (scale_new * (self.x_c_s @ R_new + delta_new) + t_new).half()

        # 7. Eye Retargeting: Natural blink closure ONLY, strictly preventing bulging/crazy eyes
        if self.flag_wink:
            x_d_i_new = x_d_i_new + self._wink_delta(lmk, dt)
        else:
            x_d_i_new = self._blink_both(x_d_i_new, lmk, dt)

        return self._finish_keypoints(x_d_i_new, lmk, dt)

    def _blink_both(self, x_d_i_new, lmk, dt):
        """Parpadeo simétrico: ambos ojos se cierran según la media de los dos."""
        r_eyes_cur = calc_eye_close_ratio(lmk[None])
        cur_eye_val = float(r_eyes_cur.mean())
        if self.c_d_eye_0 is None:
            self.c_d_eye_0 = max(cur_eye_val, 0.15)
        elif cur_eye_val > self.c_d_eye_0:
            # Slowly adapt baseline upward if user opens eyes wider
            self.c_d_eye_0 = 0.98 * self.c_d_eye_0 + 0.02 * cur_eye_val

        rel_eye = cur_eye_val / self.c_d_eye_0
        if rel_eye < 0.82:
            # User is closing eyes / blinking
            blink_w = smoothstep((0.82 - rel_eye) / (0.82 - 0.45))
            target_val = (1.0 - blink_w) * self.c_s_eye_mean + blink_w * 0.03
            target_filtered = float(self.smoother_eye.update(np.array([target_val]), dt)[0])
            self.eye_ratio_buf[:, -1].fill_(target_filtered)
            delta_eye = self.wrapper.retarget_eye(self.x_s, self.eye_ratio_buf)
            x_d_i_new = x_d_i_new + delta_eye * (blink_w * 0.85)
        else:
            # Fully open: delta_eye is zero. Eyes remain 100% natural, never bulging!
            self.smoother_eye.reset()
        return x_d_i_new

    def _wink_delta(self, lmk, dt):
        """
        --wink: cada ojo con su propio ratio y su propia referencia, así un guiño cierra
        solo ese ojo. La red de retargeting solo acepta un valor de cierre, así que se evalúa
        una vez por ojo y de cada salida se conserva solo el lado correspondiente.
        """
        r_eyes = calc_eye_close_ratio(lmk[None])[0]
        # Referencia por ojo: sube lentamente si se abren más (igual que el modo simétrico)
        self.c_d_eyes_0 = np.where(r_eyes > self.c_d_eyes_0,
                                   0.98 * self.c_d_eyes_0 + 0.02 * r_eyes, self.c_d_eyes_0)
        rel = r_eyes / self.c_d_eyes_0
        blink_w = np.array([smoothstep((0.82 - v) / (0.82 - 0.45)) if v < 0.82 else 0.0 for v in rel])
        if not blink_w.any():
            self.smoother_eyes.reset()
            return 0.0
        targets = self.smoother_eyes.update((1.0 - blink_w) * self.c_s_eye_mean + blink_w * 0.03, dt)
        delta = 0.0
        for side in range(2):
            if blink_w[side] > 0.0:
                self.eye_ratio_buf[:, -1].fill_(float(targets[side]))
                d = self.wrapper.retarget_eye(self.x_s, self.eye_ratio_buf)
                delta = delta + d * self.eye_side_masks[side] * float(blink_w[side] * 0.85)
        return delta

    # Ganancia pupila->edición de mirada, medida renderizando el avatar con la edición de
    # globo ocular de LivePortrait: 15 unidades desplazan la pupila ~0.16 anchos de ojo en x,
    # así que tu desplazamiento relativo se copia ~1:1. En y el landmark de la pupila apenas
    # se mueve (se mueve el párpado), así que se usa la misma ganancia para no amplificar ruido.
    GAZE_GAIN_X = GAZE_GAIN_Y = 15.0 / 0.16
    GAZE_LIMIT = 25.0

    def _gaze_delta(self, lmk, dt):
        """--gaze-multiplier: posición de la pupila (landmarks 197/198) -> keypoints 11/15."""
        g = self.smoother_gaze.update(calc_gaze_offset(lmk) - self.gaze_0, dt)
        gx = float(np.clip(g[0] * self.GAZE_GAIN_X * self.gaze_multiplier, -self.GAZE_LIMIT, self.GAZE_LIMIT))
        # En vertical el párpado arrastra el landmark de la pupila: se atenúa al parpadear
        r_eyes = calc_eye_close_ratio(lmk[None])[0] / np.maximum(self.c_d_eyes_0, 1e-3)
        open_w = float(np.clip((r_eyes.min() - 0.6) / 0.3, 0.0, 1.0))
        gy = float(np.clip(-g[1] * self.GAZE_GAIN_Y * self.gaze_multiplier * open_w, -self.GAZE_LIMIT, self.GAZE_LIMIT))
        # Fórmula de edición de globo ocular de LivePortrait (gradio_pipeline), en espacio exp
        k11, k15 = (0.0007, 0.001) if gx > 0 else (0.001, 0.0007)
        self.gaze_delta[0, 11, 0] = gx * k11
        self.gaze_delta[0, 15, 0] = gx * k15
        self.gaze_delta[0, 11, 1] = gy * -0.001
        self.gaze_delta[0, 15, 1] = gy * -0.001
        return self.gaze_delta

    def _finish_keypoints(self, x_d_i_new, lmk, dt):
        # Capa 2: Lip Retargeting Asistido (Asistencia inteligente de sellado sin sobre-apertura)
        # Solo interviene para cerrar suavemente si la boca está en reposo; al hablar no suma apertura extra
        if self.inf_cfg.flag_lip_retargeting and self.c_d_lip_0 is not None:
            r_lip_raw = float(calc_lip_close_ratio(lmk[None])[0, 0])
            r_lip_cur = float(self.smoother_lip_seal.update(np.array([r_lip_raw]), dt)[0])
            if r_lip_cur < self.c_d_lip_0 * 1.15:
                self.lip_ratio_buf[:, -1].fill_(r_lip_cur)
                delta_lip = self.wrapper.retarget_lip(self.x_s, self.lip_ratio_buf)
                seal_factor = max(0.0, 1.0 - (r_lip_cur / (self.c_d_lip_0 * 1.15)))
                x_d_i_new = x_d_i_new + delta_lip * (0.08 * seal_factor)

        # 8. Stitching
        if self.inf_cfg.flag_stitching:
            x_d_i_new = self.wrapper.stitching(self.x_s, x_d_i_new).half()
        return x_d_i_new

    def _postprocess(self, out: torch.Tensor) -> torch.Tensor:
        """1x3xHxW float [0,1] RGB -> 1x3xHxW float [0,255] BGR en GPU, con grano de piel aplicado."""
        out = out.flip(1) * 255.0
        if self.skin_noise > 0.0:
            _, c, h, w = out.shape
            if self.noise_tex is None or self.noise_tex.shape[-2:] != (h + self.NOISE_PAD, w + self.NOISE_PAD):
                self.noise_tex = torch.randn((1, c, h + self.NOISE_PAD, w + self.NOISE_PAD),
                                             device=out.device, dtype=out.dtype) * self.skin_noise
            oy, ox = np.random.randint(0, self.NOISE_PAD + 1, size=2)
            out = out + self.noise_tex[:, :, oy:oy + h, ox:ox + w]
        return out

    @staticmethod
    def _to_numpy_u8(img: torch.Tensor) -> np.ndarray:
        """1x3xHxW float [0,255] en GPU -> HxWx3 uint8 en CPU."""
        return img.clamp_(0.0, 255.0).to(torch.uint8)[0].permute(1, 2, 0).contiguous().cpu().numpy()

    def _render(self, kp: torch.Tensor) -> np.ndarray:
        """Warping + SPADE decoding de unos keypoints de conducción y composición final (BGR)."""
        self.last_kp_out = kp
        out = self.wrapper.warp_decode(self.f_s, self.x_s, kp)
        out_bgr = self._postprocess(out["out"])

        if not self.flag_pasteback:
            return self._to_numpy_u8(out_bgr)

        if not self.flag_seamless:
            # 10. Pasteback GPU: alpha blend con máscara difuminada sobre el ROI precomputado
            face = F.grid_sample(out_bgr, self.paste_grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=False)
            blended = self.paste_bg_roi + self.paste_mask * (face - self.paste_bg_roi)
            bx, by, bw, bh = self.paste_bbox
            out_full = self.paste_bg_bgr.copy()
            out_full[by:by+bh, bx:bx+bw] = self._to_numpy_u8(blended)
            return out_full

        # 10b. Pasteback clásico con seamlessClone (--seamless), a resolución completa del avatar
        out_crop = cv2.cvtColor(self._to_numpy_u8(out_bgr), cv2.COLOR_BGR2RGB)
        try:
            dsize = (self.source_rgb.shape[1], self.source_rgb.shape[0])
            transformed_crop = _transform_img(out_crop, self.M_c2o, dsize=dsize)
            bx, by, bw, bh = self.paste_bbox
            if bw > 0 and bh > 0:
                src_roi = transformed_crop[by:by+bh, bx:bx+bw]
                out_full = cv2.seamlessClone(src_roi, self.source_rgb, self.mask_roi, self.paste_center, cv2.NORMAL_CLONE)
            else:
                out_full = paste_back(out_crop, self.M_c2o, self.source_rgb, self.mask_ori_float)
        except Exception:
            out_full = paste_back(out_crop, self.M_c2o, self.source_rgb, self.mask_ori_float)
        return cv2.cvtColor(out_full, cv2.COLOR_RGB2BGR)

def calc_gaze_offset(lmk: np.ndarray) -> np.ndarray:
    """
    Posición media de las pupilas (landmarks 197/198) respecto al centro de cada ojo
    (contornos 0:24 y 24:48), en anchos de ojo. Devuelve [dx, dy].
    """
    offs = []
    for contour, pupil in ((lmk[0:24], lmk[197]), (lmk[24:48], lmk[198])):
        width = max(float(np.ptp(contour[:, 0])), 1.0)
        offs.append((pupil - contour.mean(axis=0)) / width)
    return np.mean(offs, axis=0)


def format_frame_for_output(frame: np.ndarray, target_w: int = 640, target_h: int = 480, is_pasteback: bool = False) -> np.ndarray:
    """
    Ensures natural aspect ratio when streaming to virtual camera.
    Prevents horizontal stretching, fat/squished faces, and distortion.
    """
    h, w = frame.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h

    if is_pasteback and abs(current_ratio - target_ratio) < 0.05:
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    elif is_pasteback and current_ratio > target_ratio:
        # Source is wider than target: crop the sides
        crop_w = int(h * target_ratio)
        x_offset = (w - crop_w) // 2
        cropped = frame[:, x_offset:x_offset + crop_w]
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        # Face crop, or a portrait taller than 4:3 (e.g. 9:16): show the whole frame
        # centered over a blurred background instead of cropping it down to the face
        scale = min(target_w / w, target_h / h)
        rw, rh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
        # Blur a baja resolución (mucho más barato) y luego escalar de vuelta;
        # visualmente casi idéntico a difuminar el frame completo a 640x480.
        small = cv2.resize(frame, (target_w // 4, target_h // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (9, 9), 0)
        bg = cv2.resize(small, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        x_off = (target_w - rw) // 2
        y_off = (target_h - rh) // 2
        bg[y_off:y_off + rh, x_off:x_off + rw] = resized
        return bg


class StdinCommandListener:
    """
    Escucha stdin en un hilo aparte para poder recalibrar la pose neutral
    (o ajustar sensibilidad) escribiendo un comando + Enter en la misma
    terminal, sin necesidad de una ventana de preview.
    """
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd:
                self.q.put(cmd)

    def poll(self):
        """Devuelve el próximo comando pendiente, o None si no hay ninguno."""
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


def find_default_virtual_cam():
    for node in ["/dev/video2", "/dev/video10", "/dev/video3", "/dev/video4"]:
        if os.path.exists(node):
            return node
    return "/dev/video2"


def find_default_source_image():
    for candidate in ["avatar.png", "avatar2.png", "avatar.jpg"]:
        candidate_path = os.path.join(CURRENT_DIR, candidate)
        if os.path.exists(candidate_path):
            return candidate_path
    return os.path.join(CURRENT_DIR, "avatar.jpg")


def parse_args():
    default_img = find_default_source_image()
    parser = argparse.ArgumentParser(description="LivePortrait Real-Time Animation Pipeline")
    parser.add_argument("--source-image", "-s", type=str, default=default_img,
                        help=f"Path to the static portrait photo to animate as avatar (default: {default_img}).")
    parser.add_argument("--webcam-id", "-w", default="0",
                        help="Webcam ID index (e.g. 0) or device path (e.g. /dev/video0).")
    parser.add_argument("--output-device", "-o", type=str, default=find_default_virtual_cam(),
                        help=f"Virtual camera device node (default: {find_default_virtual_cam()}).")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target streaming frame rate (default: 30).")
    parser.add_argument("--pasteback", action=argparse.BooleanOptionalAction, default=True,
                        help="Paste animated face back into full portrait frame (default: True, use --no-pasteback for the face crop only).")
    parser.add_argument("--seamless", action="store_true",
                        help="With --pasteback: use cv2.seamlessClone at full avatar resolution instead of the GPU feathered alpha blend (much slower).")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for maximum inference speed (>40-60 FPS).")
    parser.add_argument("--driving-multiplier", "-m", type=float, default=0.50,
                        help="Head pose and general motion intensity multiplier (default: 0.50). Values between 0.40-0.60 provide stable, subtle motion.")
    parser.add_argument("--lip-multiplier", type=float, default=1.00,
                        help="Lip speech articulation multiplier (default: 1.00). Adjusts amplitude of mouth visemes/phonemes without head motion interference.")
    parser.add_argument("--lip-retargeting", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable secondary landmark lip-seal assistance (default: True, use --no-lip-retargeting to disable).")
    parser.add_argument("--gaze-multiplier", type=float, default=0.0,
                        help="(Experimental) Make the avatar follow your gaze from your pupil landmarks. 1.0 copies your eye movement 1:1 (default: 0 = off).")
    parser.add_argument("--wink", action=argparse.BooleanOptionalAction, default=False,
                        help="(Experimental) Close each eye independently so winks are copied (default: off, both eyes blink together).")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable interactive OpenCV preview window (preview is enabled by default).")
    parser.add_argument("--skin-noise", type=float, default=2.5,
                        help="Intensity of subtle Gaussian grain added post-decode to eliminate plastic look (default: 2.5, 0 = disabled).")
    parser.add_argument("--fp32-generator", action="store_true",
                        help="Keep spade_generator in FP32 precision to compare skin color banding against FP16.")
    parser.add_argument("--dry-run", type=float, default=0.0,
                        help="Run in benchmark mode for N seconds and exit with FPS statistics.")
    parser.add_argument("--no-virtualcam", action="store_true",
                        help="Disable virtual camera output (useful for testing without v4l2loopback).")
    default_backend = "trt" if os.path.exists(os.path.join(DEFAULT_TRT_DIR, "warping_spade-fix.trt")) else "torch"
    parser.add_argument("--backend", choices=["torch", "trt"], default=default_backend,
                        help=f"Inference backend: torch or trt (TensorRT engines, see README). Default: trt if the engines are built, else torch (now: {default_backend}).")
    parser.add_argument("--trt-engine-dir", type=str, default=DEFAULT_TRT_DIR,
                        help=f"Directory with the TensorRT engines + GridSample3D plugin (default: {DEFAULT_TRT_DIR}).")
    parser.add_argument("--detector", choices=["insightface", "mediapipe"], default="insightface",
                        help="Face detector used for (re)acquisition (default: insightface).")
    parser.add_argument("--driving-video", type=str, default=None,
                        help="Use a video file (looped at its native FPS) instead of the webcam. Reproducible benchmarks.")
    return parser.parse_args()


class LatestFrameSource:
    """
    Base para fuentes leídas en un hilo aparte que exponen siempre el frame MÁS RECIENTE.
    read(wait_new=True) bloquea (con timeout) hasta que llegue un frame que aún no se
    procesó, para no gastar GPU re-procesando duplicados ni añadir latencia de cola.
    """
    def _start(self):
        self.frame = None
        self.seq = 0
        self.consumed = 0
        self.cond = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        # Espera al primer frame real antes de continuar
        for _ in range(50):
            if self.frame is not None:
                break
            time.sleep(0.02)

    def _publish(self, frame):
        with self.cond:
            self.frame = frame
            self.seq += 1
            self.cond.notify_all()

    def read(self, wait_new=False, timeout=0.25):
        with self.cond:
            if wait_new:
                self.cond.wait_for(lambda: self.seq != self.consumed or not self.running, timeout)
            self.consumed = self.seq
            return self.frame is not None, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        self.thread.join(timeout=1.0)
        self.cap.release()


class ThreadedWebcam(LatestFrameSource):
    """
    Lee la webcam en un hilo aparte y siempre expone el frame MÁS RECIENTE.
    Evita que el loop de inferencia GPU tenga que esperar a cap.read() cada
    vez, que es la causa más común de "stutter" en pipelines real-time.
    """
    def __init__(self, cap):
        self.cap = cap
        self._start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._publish(frame)

    def isOpened(self):
        return self.cap.isOpened()


class LoopingVideoSource(LatestFrameSource):
    """
    Fuente de conducción reproducible para benchmarks: reproduce un vídeo en
    bucle a su FPS nativo en un hilo aparte y expone siempre el frame más
    reciente, igual que ThreadedWebcam con una webcam real.
    """
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el vídeo de conducción: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.period = 1.0 / (fps if fps and fps > 0 else 30.0)
        self._start()

    def _reader(self):
        t_next = time.perf_counter()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self._publish(frame)
            t_next += self.period
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()

    def isOpened(self):
        return self.running


class VirtualCamSender:
    """
    Envía a la cámara virtual desde un hilo propio, a ritmo constante, siempre el
    último frame listo (repitiéndolo si la inferencia aún no produjo uno nuevo).
    Así el loop de inferencia nunca se bloquea en send()/sleep_until_next_frame().
    """
    def __init__(self, cam):
        self.cam = cam
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def _loop(self):
        while self.running:
            with self.lock:
                frame = self.frame
            if frame is None:
                # Aún no hay frame: sleep_until_next_frame() falla si no se ha enviado ninguno
                time.sleep(0.005)
                continue
            self.cam.send(frame)
            self.cam.sleep_until_next_frame()

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cam.close()


def check_camera_locks(device_path):
    import subprocess
    try:
        out = subprocess.check_output(["fuser", device_path], stderr=subprocess.DEVNULL).decode().strip()
        pids = out.split()
        holders = []
        for pid in pids:
            try:
                name = subprocess.check_output(["ps", "-p", pid, "-o", "comm="], stderr=subprocess.DEVNULL).decode().strip()
                holders.append(f"{name} (PID {pid})")
            except Exception:
                holders.append(f"PID {pid}")
        return holders
    except Exception:
        return []


def open_webcam_robust(webcam_id, target_fps=30, retries=4, delay=0.4):
    """
    Robustly acquires webcam stream via path or index, trying native V4L2 then CAP_ANY.
    Includes automated retries and detailed lock diagnostics.
    """
    dev_path = f"/dev/video{webcam_id}" if str(webcam_id).isdigit() else str(webcam_id)
    idx = int(webcam_id) if str(webcam_id).isdigit() else None

    for attempt in range(retries):
        candidates = [(dev_path, cv2.CAP_V4L2), (dev_path, cv2.CAP_ANY)]
        if idx is not None:
            candidates.extend([(idx, cv2.CAP_V4L2), (idx, cv2.CAP_ANY)])

        for dev, backend in candidates:
            try:
                cap = cv2.VideoCapture(dev, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, target_fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return cap
                    cap.release()
                elif cap is not None:
                    cap.release()
            except Exception:
                pass

        time.sleep(delay)

    holders = check_camera_locks(dev_path)
    if holders:
        print(f"[!] Error: La cámara {dev_path} está ocupada por: {', '.join(holders)}")
        print("    Cierra esa aplicación para que LivePortrait pueda acceder a la webcam.")
    return None


def main():
    args = parse_args()

    print("=" * 65)
    print(" LIVEPORTRAIT REAL-TIME GPU INFERENCE PIPELINE")
    print("=" * 65)
    print(f" Avatar Source:    {args.source_image}")
    print(f" Physical Webcam:  {args.webcam_id}")
    print(f" Virtual Output:   {args.output_device}")
    print(f" Target FPS:       {args.fps}")
    print(f" Full Pasteback:   {args.pasteback}{' (seamlessClone)' if args.seamless else ''}")
    print(f" Torch Compile:    {args.compile}")
    print(f" Motion Mult:      {args.driving_multiplier}")
    print(f" Lip Mult:         {args.lip_multiplier}")
    print(f" Lip Retargeting:  {args.lip_retargeting}")
    print(f" Skin Grain Noise: {args.skin_noise}")
    print(f" FP32 Generator:   {args.fp32_generator}")
    print(f" Backend:          {args.backend}")
    print(f" Gaze follow:      {args.gaze_multiplier:.2f}{' (off)' if args.gaze_multiplier <= 0 else ''}")
    print(f" Wink (per eye):   {args.wink}")
    print(f" Face Detector:    {args.detector}")
    print("=" * 65)

    # 1. Initialize Webcam (en hilo separado para no bloquear el loop de inferencia)
    cap_raw = None if args.driving_video else open_webcam_robust(args.webcam_id, args.fps)
    if args.driving_video:
        cap = LoopingVideoSource(args.driving_video)
        print(f"[*] Fuente de conducción: vídeo en bucle {args.driving_video}")
    elif cap_raw is None:
        if args.dry_run > 0:
            print("[*] Generating synthetic webcam stream for benchmark dry-run...")
            cap = cap_raw
        else:
            print(f"[!] No se pudo abrir la cámara física: {args.webcam_id}")
            sys.exit(1)
    else:
        cap = ThreadedWebcam(cap_raw)

    # Standard webcam output dimensions (640x480) for Zoom / WebRTC compatibility
    out_w, out_h = 640, 480

    # 2. Initialize LivePortrait Pipeline
    pipeline = LivePortraitCamPipeline(
        source_image_path=args.source_image,
        device_id=0,
        flag_pasteback=args.pasteback,
        flag_compile=args.compile,
        driving_multiplier=args.driving_multiplier,
        skin_noise=args.skin_noise,
        flag_fp32_generator=args.fp32_generator,
        flag_lip_retargeting=args.lip_retargeting,
        lip_multiplier=args.lip_multiplier,
        gaze_multiplier=args.gaze_multiplier,
        flag_wink=args.wink,
        flag_seamless=args.seamless,
        output_size=(out_w, out_h),
        backend=args.backend,
        detector=args.detector,
        trt_engine_dir=args.trt_engine_dir
    )

    # 3. Initialize Virtual Camera (pyvirtualcam)
    virtual_cam = None
    if not args.no_virtualcam:
        try:
            import pyvirtualcam
            cam = pyvirtualcam.Camera(
                width=out_w,
                height=out_h,
                fps=args.fps,
                device=args.output_device,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
            virtual_cam = VirtualCamSender(cam)
            print(f"[+] Streaming to Virtual Camera active on: {cam.device} ({out_w}x{out_h} @ {args.fps} FPS)")
        except Exception as e:
            print(f"[!] Warning: Failed to open virtual camera ({args.output_device}): {e}")
            print("    Continuing with live processing and metrics...")

    # 4. Initialize Preview Window (si está habilitada)
    preview_enabled = not args.no_preview
    window_name = "LivePortrait Preview [c: Centrar | q: Salir]"
    if preview_enabled:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, out_w, out_h)
            print(f"[+] Ventana de Preview activa: '{window_name}'")
        except Exception as e:
            print(f"[!] Warning: No se pudo abrir la ventana GUI ({e}). Continuando sin preview...")
            preview_enabled = False

    # Signal handlers for clean termination
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("\n[*] Interrupción capturada (SIGINT). Saliendo limpiamente...")
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    stdin_listener = StdinCommandListener()

    print("\n" + "=" * 65)
    print(" [>>>] PIPELINE EN EJECUCIÓN (CON VENTANA DE PREVIEW)")
    print("=" * 65)
    print(" Controles interactivos (en la ventana de preview o en la terminal):")
    print("   c        -> Centrar / recalibrar pose neutral (mira de frente a la cámara)")
    print("   q        -> Salir del preview y cerrar el programa")
    print("   m+ / m-  -> Subir / bajar sensibilidad de cabeza (driving multiplier)")
    print("   l+ / l-  -> Subir / bajar sensibilidad de articulación de labios\n")

    frame_count = 0
    t_start = time.perf_counter()
    latency_records = []
    frame_times = []
    recalibration_feedback_time = 0.0

    try:
        while running:
            # Read webcam frame: espera un frame NUEVO (no re-procesa duplicados)
            if cap is not None and cap.isOpened():
                ret, frame_bgr = cap.read(wait_new=True)
                if not ret or frame_bgr is None:
                    continue
            else:
                # Synthetic dummy face for headless / dry-run testing
                frame_bgr = cv2.resize(cv2.imread(args.source_image), (640, 480))

            # La latencia se mide desde que el frame está disponible (sin la espera a la cámara)
            t_frame_start = time.perf_counter()

            # Run inference
            t_infer_start = time.perf_counter()
            out_bgr = pipeline.process_frame(frame_bgr)
            # Solo sincronizamos cuando vamos a reportar métricas (cada 30 frames).
            # Sincronizar SIEMPRE serializa CPU y GPU en cada iteración y es
            # el principal responsable de que se sienta "trabado" en vez de fluido.
            if (frame_count + 1) % 30 == 0:
                torch.cuda.synchronize()
            t_infer_end = time.perf_counter()

            if out_bgr is None:
                # Face temporarily undetected, retain fallback
                out_bgr = cv2.cvtColor(pipeline.source_rgb, cv2.COLOR_RGB2BGR)

            # Format frame maintaining natural aspect ratio (prevents facial deformation)
            out_formatted = format_frame_for_output(out_bgr, out_w, out_h, is_pasteback=args.pasteback)

            # Send clean, unadorned video stream to virtual cam for Zoom/Discord/WebRTC
            # (el envío a ritmo constante lo hace el hilo de VirtualCamSender)
            if virtual_cam is not None:
                virtual_cam.submit(out_formatted)

            # Render interactive GUI preview window if enabled
            if preview_enabled:
                display_frame = out_formatted.copy()
                if time.perf_counter() < recalibration_feedback_time:
                    cv2.putText(display_frame, "POSE CENTRADA / RECALIBRADA", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('c'), ord('C')):
                    pipeline.x_d_0_info = None
                    pipeline.last_lmk = None
                    recalibration_feedback_time = time.perf_counter() + 1.2
                    print("[*] [Preview 'c'] Pose neutral centrada y recalibrada.")
                elif key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                    print("[*] [Preview 'q'] Cerrando LivePortrait...")
                    running = False

            # Measure performance
            t_frame_end = time.perf_counter()
            frame_ms = (t_frame_end - t_frame_start) * 1000.0
            infer_ms = (t_infer_end - t_infer_start) * 1000.0
            latency_records.append((frame_ms, infer_ms))
            frame_times.append(t_frame_end)
            frame_count += 1

            if frame_count % 30 == 0:
                recent_fps = 29.0 / max(frame_times[-1] - frame_times[-30], 1e-6)
                avg_infer = np.mean([x[1] for x in latency_records[-30:]])
                status_str = "TRACKING OK" if pipeline.is_tracking else "NO ROSTRO DETECTADO"
                print(f"[*] Latency: {frame_ms:.1f} ms | GPU Infer: {avg_infer:.1f} ms | Effective: {recent_fps:.1f} FPS | Estado: {status_str}")

            # Comandos por stdin (recalibrar / sensibilidad / salir) sin bloquear el loop
            cmd = stdin_listener.poll()
            if cmd == "c":
                pipeline.x_d_0_info = None  # Fuerza recalibración en el próximo frame válido
                pipeline.last_lmk = None
                recalibration_feedback_time = time.perf_counter() + 1.2
                print("[*] [Terminal 'c'] Recalibrando... mira de frente a la cámara con pose neutral.")
            elif cmd in ("q", "quit", "exit"):
                print("[*] [Terminal 'q'] Cerrando LivePortrait...")
                running = False
            elif cmd == "m+":
                pipeline.driving_multiplier = min(1.20, round(pipeline.driving_multiplier + 0.05, 2))
                print(f"[*] Sensibilidad de cabeza: {pipeline.driving_multiplier:.2f}")
            elif cmd == "m-":
                pipeline.driving_multiplier = max(0.20, round(pipeline.driving_multiplier - 0.05, 2))
                print(f"[*] Sensibilidad de cabeza: {pipeline.driving_multiplier:.2f}")
            elif cmd == "l+":
                pipeline.lip_multiplier = min(3.00, round(pipeline.lip_multiplier + 0.10, 2))
                print(f"[*] Sensibilidad de labios: {pipeline.lip_multiplier:.2f}")
            elif cmd == "l-":
                pipeline.lip_multiplier = max(0.20, round(pipeline.lip_multiplier - 0.10, 2))
                print(f"[*] Sensibilidad de labios: {pipeline.lip_multiplier:.2f}")

            # Check dry-run duration limit
            if args.dry_run > 0 and (time.perf_counter() - t_start) >= args.dry_run:
                print(f"[*] Benchmark dry-run duration reached ({args.dry_run}s).")
                break

    finally:
        total_time = time.perf_counter() - t_start
        if preview_enabled:
            cv2.destroyAllWindows()
        if cap is not None and cap.isOpened():
            cap.release()
        if virtual_cam is not None:
            virtual_cam.close()

        if latency_records:
            avg_latency = np.mean([x[0] for x in latency_records])
            avg_infer = np.mean([x[1] for x in latency_records])
            actual_fps = frame_count / total_time
            print("\n" + "=" * 65)
            print(" PERFORMANCE BENCHMARK SUMMARY")
            print("=" * 65)
            print(f" Total Frames Processed: {frame_count}")
            print(f" Elapsed Time:           {total_time:.2f} s")
            print(f" Average Total Latency:  {avg_latency:.2f} ms/frame")
            print(f" P95 Total Latency:      {np.percentile([x[0] for x in latency_records], 95):.2f} ms/frame")
            print(f" Average GPU Inference:  {avg_infer:.2f} ms/frame")
            print(f" Throughput:             {actual_fps:.2f} FPS")
            print("=" * 65)


if __name__ == "__main__":
    main()
