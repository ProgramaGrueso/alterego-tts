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

# Ensure liveportrait_src is discoverable in PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "liveportrait_src")
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


class LivePortraitCamPipeline:
    def __init__(self, source_image_path: str, device_id: int = 0, flag_pasteback: bool = False, flag_compile: bool = False, driving_multiplier: float = 0.50, skin_noise: float = 2.5, flag_fp32_generator: bool = False, flag_lip_retargeting: bool = True, lip_multiplier: float = 1.00):
        self.device_id = device_id
        self.flag_pasteback = flag_pasteback
        self.driving_multiplier = driving_multiplier
        self.skin_noise = float(skin_noise)
        self.flag_fp32_generator = flag_fp32_generator
        self.flag_lip_retargeting = flag_lip_retargeting
        self.lip_multiplier = float(lip_multiplier)

        # Facial keypoint indices in LivePortrait latent expression space
        self.LIP_INDICES = [6, 12, 14, 17, 19, 20]
        self.FACE_INDICES = [i for i in range(21) if i not in self.LIP_INDICES]

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
        self.c_s_eyes_tensor = torch.from_numpy(self.c_s_eyes).half().to(f"cuda:{device_id}")
        self.c_s_eye_mean = float(self.c_s_eyes.mean())
        self.c_d_eye_0 = None

        # Precompute source lip ratio for lip retargeting
        self.c_s_lip = calc_lip_close_ratio(self.source_lmk[None])
        self.c_s_lip_tensor = torch.from_numpy(self.c_s_lip).half().to(f"cuda:{device_id}")

        # Precompute pasteback mask and seamlessClone ROI if requested
        if self.flag_pasteback:
            h, w = self.source_rgb.shape[:2]
            self.mask_ori_float = prepare_paste_back(
                self.inf_cfg.mask_crop, self.M_c2o, dsize=(w, h)
            )
            self.mask_u8 = (self.mask_ori_float[..., 0] * 255).astype(np.uint8)
            bx, by, bw, bh = cv2.boundingRect(self.mask_u8)
            self.paste_bbox = (bx, by, bw, bh)
            self.paste_center = (bx + bw // 2, by + bh // 2)
            self.mask_roi = self.mask_u8[by:by+bh, bx:bx+bw]
        else:
            self.mask_ori_float = None
            self.mask_u8 = None
            self.paste_bbox = None
            self.paste_center = None
            self.mask_roi = None

        print("[+] Source avatar features pre-computed and cached in VRAM.")

        # Reference driving motion cache (calibrated on initial detection)
        self.x_d_0_info = None
        self.R_d_0 = None
        self.last_lmk = None
        self.is_tracking = False

    def calibrate_neutral_pose(self, x_d_info, lmk):
        """Calibrates neutral expression, eye openness, lip baseline, and head pose from driving webcam."""
        self.x_d_0_info = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in x_d_info.items()}
        r_eyes = calc_eye_close_ratio(lmk[None])
        self.c_d_eye_0 = max(float(r_eyes.mean()), 0.15)
        r_lip = calc_lip_close_ratio(lmk[None])
        self.c_d_lip_0 = max(float(r_lip[0, 0]), 0.02)
        self.smoother_lmk.reset()
        self.smoother_angles.reset()
        self.smoother_exp_face.reset()
        self.smoother_exp_lip.reset()
        self.smoother_t.reset()
        self.smoother_scale.reset()
        self.smoother_eye.reset()
        self.smoother_lip_seal.reset()
        print(f"[+] Pose neutra calibrada (Sensibilidad cabeza: {self.driving_multiplier:.2f} | Labios: {self.lip_multiplier:.2f} | Ojos: {self.c_d_eye_0:.2f} | Boca reposo: {self.c_d_lip_0:.2f}).")

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Executes LivePortrait deformation on one driving frame.
        Returns animated output frame (BGR, uint8).
        """
        # Safety check: if webcam is covered / pitch dark, report no tracking
        if frame_bgr.mean() < 12.0:
            self.last_lmk = None
            self.is_tracking = False
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. Face tracking / alignment
        if self.last_lmk is None:
            # First frame or re-acquisition: run detector
            src_face = self.cropper.face_analysis_wrapper.get(
                frame_bgr, flag_do_landmark_2d_106=True, direction="large-small"
            )
            if len(src_face) == 0:
                self.is_tracking = False
                return None
            lmk = src_face[0].landmark_2d_106
            lmk = self.cropper.human_landmark_runner.run(frame_rgb, lmk)
            self.last_lmk = lmk
        else:
            # Tracking mode: pass previous landmark as prior (fast ONNX execution ~2ms)
            lmk = self.cropper.human_landmark_runner.run(frame_rgb, self.last_lmk)
            if lmk is None or lmk.min() < -50 or lmk.max() > max(frame_bgr.shape[:2]) + 50:
                # Re-detect if tracking was lost or landmark collapsed
                src_face = self.cropper.face_analysis_wrapper.get(
                    frame_bgr, flag_do_landmark_2d_106=True, direction="large-small"
                )
                if len(src_face) == 0:
                    self.last_lmk = None
                    self.is_tracking = False
                    return None
                lmk = self.cropper.human_landmark_runner.run(frame_rgb, src_face[0].landmark_2d_106)
            self.last_lmk = lmk

        # Apply temporal smoothing to facial landmarks to eliminate crop jitter
        lmk = self.smoother_lmk.update(lmk)

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
            self.is_tracking = False
            return None

        img_driving_crop_256 = cv2.resize(ret_crop["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)

        # 3. Driving keypoints inference
        with torch.inference_mode():
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

            # Deadband filter: micro-movements < 0.30 degrees are suppressed (preserves calm posture when still)
            deadband_angle = 0.30
            angles_raw = torch.where(torch.abs(angles_raw) < deadband_angle, angles_raw * 0.25, angles_raw)
            angles_smooth = self.smoother_angles.update(angles_raw)

            # Apply motion multiplier to rotation
            angles_damped = angles_smooth * self.driving_multiplier
            pitch_new = self.x_s_info["pitch"] + angles_damped[:, 0:1]
            yaw_new = self.x_s_info["yaw"] + angles_damped[:, 1:2]
            roll_new = self.x_s_info["roll"] + angles_damped[:, 2:3]
            R_new = get_rotation_matrix(pitch_new, yaw_new, roll_new).half()

            # 5. Expression delta (Capa 1: Lipsync reactivo desacoplado para labios + estabilidad facial)
            delta_raw = x_d_i_info["exp"] - self.x_d_0_info["exp"]

            # Región facial (cejas, pómulos): deadband suave anti-jitter y filtro One-Euro suave
            delta_face = delta_raw.clone()
            deadband_face = 0.0045
            delta_face = torch.where(torch.abs(delta_face) < deadband_face, delta_face * 0.25, delta_face)
            delta_face_smooth = self.smoother_exp_face.update(delta_face)

            # Región labial (articulación vocal / fonemas): deadband anti-muecas en reposo + One-Euro adaptativo
            delta_lip_raw = delta_raw.clone()
            deadband_lip = 0.0055
            delta_lip_raw = torch.where(torch.abs(delta_lip_raw) < deadband_lip, delta_lip_raw * 0.20, delta_lip_raw)
            delta_lip_smooth = self.smoother_exp_lip.update(delta_lip_raw)

            # Fusión de componentes:
            # - Resto del rostro escala con driving_multiplier (control natural de expresiones secundarias)
            # - Labios escalan con lip_multiplier (articulación nítida e independiente de la cabeza)
            delta_combined = delta_face_smooth.clone()
            for idx in self.FACE_INDICES:
                delta_combined[:, idx, :] = delta_face_smooth[:, idx, :] * self.driving_multiplier
            for idx in self.LIP_INDICES:
                delta_combined[:, idx, :] = delta_lip_smooth[:, idx, :] * self.lip_multiplier

            delta_new = self.x_s_info["exp"] + delta_combined

            # 6. Translation and scale (smoothed and controlled)
            t_raw = x_d_i_info["t"] - self.x_d_0_info["t"]
            t_smooth = self.smoother_t.update(t_raw)
            t_new = self.x_s_info["t"] + t_smooth * self.driving_multiplier
            t_new[..., 2].fill_(0)

            scale_raw = (x_d_i_info["scale"] / self.x_d_0_info["scale"] - 1.0) * self.driving_multiplier + 1.0
            scale_smooth = self.smoother_scale.update(scale_raw)
            scale_new = self.x_s_info["scale"] * scale_smooth

            x_d_i_new = (scale_new * (self.x_c_s @ R_new + delta_new) + t_new).half()

            # 7. Eye Retargeting: Natural blink closure ONLY, strictly preventing bulging/crazy eyes
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
                blink_prog = float(np.clip((0.82 - rel_eye) / (0.82 - 0.45), 0.0, 1.0))
                blink_w = blink_prog * blink_prog * (3.0 - 2.0 * blink_prog)
                target_val = (1.0 - blink_w) * self.c_s_eye_mean + blink_w * 0.03
                target_tensor = torch.tensor([[target_val]]).half().to(f"cuda:{self.device_id}")
                target_filtered = self.smoother_eye.update(target_tensor)
                combined_eye = torch.cat([self.c_s_eyes_tensor, target_filtered], dim=1)
                delta_eye = self.wrapper.retarget_eye(self.x_s.half(), combined_eye)
                x_d_i_new = x_d_i_new + delta_eye * (blink_w * 0.85)
            else:
                # Fully open: delta_eye is zero. Eyes remain 100% natural, never bulging!
                self.smoother_eye.reset()

            # Capa 2: Lip Retargeting Asistido (Asistencia inteligente de sellado sin sobre-apertura)
            # Solo interviene para cerrar suavemente si la boca está en reposo; al hablar no suma apertura extra
            if self.inf_cfg.flag_lip_retargeting and self.c_d_lip_0 is not None:
                r_lip_raw = float(calc_lip_close_ratio(lmk[None])[0, 0])
                r_lip_cur = float(self.smoother_lip_seal.update(np.array([r_lip_raw]))[0])
                if r_lip_cur < self.c_d_lip_0 * 1.15:
                    c_d_lip_tensor = torch.tensor([[r_lip_cur]], dtype=torch.float16, device=f"cuda:{self.device_id}")
                    combined_lip = torch.cat([self.c_s_lip_tensor, c_d_lip_tensor], dim=1)
                    delta_lip = self.wrapper.retarget_lip(self.x_s.half(), combined_lip)
                    seal_factor = max(0.0, 1.0 - (r_lip_cur / (self.c_d_lip_0 * 1.15)))
                    x_d_i_new = x_d_i_new + delta_lip * (0.08 * seal_factor)

            # 8. Stitching
            if self.inf_cfg.flag_stitching:
                x_d_i_new = self.wrapper.stitching(self.x_s, x_d_i_new).half()

            # 9. Warping and SPADE Generator Decoding
            out = self.wrapper.warp_decode(self.f_s, self.x_s, x_d_i_new)
            out_crop = self.wrapper.parse_output(out["out"])[0]  # HxWx3 uint8 RGB

            # Skin texture grain to reduce plastic / over-smoothed appearance
            if self.skin_noise > 0.0:
                noise = np.random.normal(0, self.skin_noise, out_crop.shape).astype(np.int16)
                out_crop = np.clip(out_crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self.is_tracking = True

        # 10. Formatting and Pasteback (with seamlessClone & fallback)
        if self.flag_pasteback and self.mask_ori_float is not None:
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
        else:
            return cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)


def format_frame_for_output(frame: np.ndarray, target_w: int = 640, target_h: int = 480, is_pasteback: bool = False) -> np.ndarray:
    """
    Ensures natural aspect ratio when streaming to virtual camera.
    Prevents horizontal stretching, fat/squished faces, and distortion.
    """
    h, w = frame.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h

    if is_pasteback:
        if abs(current_ratio - target_ratio) < 0.05:
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        elif current_ratio < target_ratio:
            # Source is narrower/taller than target (e.g. 1:1 image to 4:3 screen)
            crop_h = int(w / target_ratio)
            y_offset = max(0, int((h - crop_h) * 0.35))  # Keep head and upper torso
            cropped = frame[y_offset:y_offset + crop_h, :]
            return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            # Source is wider than target
            crop_w = int(h * target_ratio)
            x_offset = (w - crop_w) // 2
            cropped = frame[:, x_offset:x_offset + crop_w]
            return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        # Square cropped face mode: maintain 1:1 face proportion with blurred background canvas
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
    parser.add_argument("--pasteback", action="store_true",
                        help="Paste animated face back into full portrait frame.")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for maximum inference speed (>40-60 FPS).")
    parser.add_argument("--driving-multiplier", "-m", type=float, default=0.50,
                        help="Head pose and general motion intensity multiplier (default: 0.50). Values between 0.40-0.60 provide stable, subtle motion.")
    parser.add_argument("--lip-multiplier", type=float, default=1.00,
                        help="Lip speech articulation multiplier (default: 1.00). Adjusts amplitude of mouth visemes/phonemes without head motion interference.")
    parser.add_argument("--lip-retargeting", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable secondary landmark lip-seal assistance (default: True, use --no-lip-retargeting to disable).")
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
    return parser.parse_args()


class ThreadedWebcam:
    """
    Lee la webcam en un hilo aparte y siempre expone el frame MÁS RECIENTE.
    Evita que el loop de inferencia GPU tenga que esperar a cap.read() cada
    vez, que es la causa más común de "stutter" en pipelines real-time.
    """
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        # Espera al primer frame real antes de continuar
        for _ in range(50):
            if self.frame is not None:
                break
            time.sleep(0.02)

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame is not None, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


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
    print(f" Full Pasteback:   {args.pasteback}")
    print(f" Torch Compile:    {args.compile}")
    print(f" Motion Mult:      {args.driving_multiplier}")
    print(f" Lip Mult:         {args.lip_multiplier}")
    print(f" Lip Retargeting:  {args.lip_retargeting}")
    print(f" Skin Grain Noise: {args.skin_noise}")
    print(f" FP32 Generator:   {args.fp32_generator}")
    print("=" * 65)

    # 1. Initialize Webcam (en hilo separado para no bloquear el loop de inferencia)
    cap_raw = open_webcam_robust(args.webcam_id, args.fps)
    if cap_raw is None:
        if args.dry_run > 0:
            print("[*] Generating synthetic webcam stream for benchmark dry-run...")
            cap = cap_raw
        else:
            print(f"[!] No se pudo abrir la cámara física: {args.webcam_id}")
            sys.exit(1)
    else:
        cap = ThreadedWebcam(cap_raw)

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
        lip_multiplier=args.lip_multiplier
    )

    # Standard webcam output dimensions (640x480) for Zoom / WebRTC compatibility
    out_w, out_h = 640, 480

    # 3. Initialize Virtual Camera (pyvirtualcam)
    virtual_cam = None
    if not args.no_virtualcam:
        try:
            import pyvirtualcam
            virtual_cam = pyvirtualcam.Camera(
                width=out_w,
                height=out_h,
                fps=args.fps,
                device=args.output_device,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
            print(f"[+] Streaming to Virtual Camera active on: {virtual_cam.device} ({out_w}x{out_h} @ {args.fps} FPS)")
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
    recalibration_feedback_time = 0.0

    try:
        while running:
            t_frame_start = time.perf_counter()

            # Read webcam frame
            if cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    continue
            else:
                # Synthetic dummy face for headless / dry-run testing
                frame_bgr = cv2.resize(cv2.imread(args.source_image), (640, 480))

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
            if virtual_cam is not None:
                virtual_cam.send(out_formatted)
                virtual_cam.sleep_until_next_frame()

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
            frame_count += 1

            if frame_count % 30 == 0:
                recent_fps = 1000.0 / np.mean([x[0] for x in latency_records[-30:]])
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
        if cap.isOpened():
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
            print(f" Average GPU Inference:  {avg_infer:.2f} ms/frame")
            print(f" Throughput:             {actual_fps:.2f} FPS")
            print("=" * 65)


if __name__ == "__main__":
    main()
