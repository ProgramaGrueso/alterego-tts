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
from src.utils.crop import paste_back, prepare_paste_back, crop_image
from src.utils.retargeting_utils import calc_eye_close_ratio


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
    def __init__(self, source_image_path: str, device_id: int = 0, flag_pasteback: bool = False, flag_compile: bool = False, driving_multiplier: float = 0.65):
        self.device_id = device_id
        self.flag_pasteback = flag_pasteback
        self.driving_multiplier = driving_multiplier

        # Anti-jitter One-Euro filters tailored for each facial/pose component
        self.smoother_lmk = OneEuroFilter(min_cutoff=0.8, beta=0.02)
        self.smoother_angles = OneEuroFilter(min_cutoff=0.4, beta=0.02)
        self.smoother_exp = OneEuroFilter(min_cutoff=0.7, beta=0.06)
        self.smoother_t = OneEuroFilter(min_cutoff=0.4, beta=0.01)
        self.smoother_scale = OneEuroFilter(min_cutoff=0.3, beta=0.01)
        self.smoother_eye = OneEuroFilter(min_cutoff=2.5, beta=0.10)

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
        self.inf_cfg.flag_lip_retargeting = False
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
        self.wrapper.spade_generator.half()
        self.wrapper.motion_extractor.half()
        if self.wrapper.stitching_retargeting_module:
            for k in self.wrapper.stitching_retargeting_module:
                self.wrapper.stitching_retargeting_module[k].half()

        # 3. Pre-process static source avatar (One-Time Execution)
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

        # Precompute pasteback mask if requested
        if self.flag_pasteback:
            h, w = self.source_rgb.shape[:2]
            self.mask_ori_float = prepare_paste_back(
                self.inf_cfg.mask_crop, self.M_c2o, dsize=(w, h)
            )
        else:
            self.mask_ori_float = None

        print("[+] Source avatar features pre-computed and cached in VRAM.")

        # Reference driving motion cache (calibrated on initial detection)
        self.x_d_0_info = None
        self.R_d_0 = None
        self.last_lmk = None
        self.is_tracking = False

    def calibrate_neutral_pose(self, x_d_info, lmk):
        """Calibrates neutral expression, eye openness, and head pose from driving webcam."""
        self.x_d_0_info = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in x_d_info.items()}
        r_eyes = calc_eye_close_ratio(lmk[None])
        self.c_d_eye_0 = max(float(r_eyes.mean()), 0.15)
        self.smoother_lmk.reset()
        self.smoother_angles.reset()
        self.smoother_exp.reset()
        self.smoother_t.reset()
        self.smoother_scale.reset()
        self.smoother_eye.reset()
        print(f"[+] Pose neutra calibrada (Sensibilidad: {self.driving_multiplier:.2f} | Apertura ojos: {self.c_d_eye_0:.2f}).")

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

            # Deadband filter: micro-movements < 0.4 degrees are suppressed (kills sensor tremor)
            deadband_angle = 0.4
            angles_raw = torch.where(torch.abs(angles_raw) < deadband_angle, angles_raw * 0.25, angles_raw)
            angles_smooth = self.smoother_angles.update(angles_raw)

            # Apply motion multiplier to rotation
            angles_damped = angles_smooth * self.driving_multiplier
            pitch_new = self.x_s_info["pitch"] + angles_damped[:, 0:1]
            yaw_new = self.x_s_info["yaw"] + angles_damped[:, 1:2]
            roll_new = self.x_s_info["roll"] + angles_damped[:, 2:3]
            R_new = get_rotation_matrix(pitch_new, yaw_new, roll_new).half()

            # 5. Expression delta (smoothed and damped to avoid mouth/eye twitching)
            delta_raw = x_d_i_info["exp"] - self.x_d_0_info["exp"]
            # Deadband on tiny resting expression jitter
            delta_raw = torch.where(torch.abs(delta_raw) < 0.005, delta_raw * 0.3, delta_raw)
            delta_smooth = self.smoother_exp.update(delta_raw)
            delta_new = self.x_s_info["exp"] + delta_smooth * self.driving_multiplier

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

            # 8. Stitching
            if self.inf_cfg.flag_stitching:
                x_d_i_new = self.wrapper.stitching(self.x_s, x_d_i_new).half()

            # 9. Warping and SPADE Generator Decoding (FP16 on Ada Lovelace)
            out = self.wrapper.warp_decode(self.f_s, self.x_s, x_d_i_new)
            out_crop = self.wrapper.parse_output(out["out"])[0]  # HxWx3 uint8 RGB

        self.is_tracking = True

        # 7. Formatting and Pasteback
        if self.flag_pasteback and self.mask_ori_float is not None:
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
        bg = cv2.resize(frame, (target_w, target_h))
        bg = cv2.GaussianBlur(bg, (51, 51), 0)
        x_off = (target_w - rw) // 2
        y_off = (target_h - rh) // 2
        bg[y_off:y_off + rh, x_off:x_off + rw] = resized
        return bg


def draw_preview_hud(avatar_bgr: np.ndarray, cam_bgr: np.ndarray, is_tracking: bool, is_calibrated: bool, brightness: float, driving_mult: float = 0.65) -> np.ndarray:
    """
    Renders status HUD and PiP webcam thumbnail for the local preview window only.
    The virtual camera receives the clean video without overlays.
    """
    canvas = avatar_bgr.copy()
    h, w = canvas.shape[:2]

    # 1. PiP webcam inset in top-right corner
    pip_w, pip_h = 160, 120
    pip_cam = cv2.resize(cam_bgr, (pip_w, pip_h))
    margin = 15
    x1, y1 = w - pip_w - margin, margin
    x2, y2 = x1 + pip_w, y1 + pip_h

    border_col = (0, 220, 0) if is_tracking else (0, 0, 255)
    canvas[y1:y2, x1:x2] = pip_cam
    cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), border_col, 2)
    cv2.putText(canvas, "TU WEBCAM", (x1 + 6, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "TU WEBCAM", (x1 + 6, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # 2. Status badge top-left
    if brightness < 15.0:
        msg = "CAMARA OSCURA / TAPA CERRADA"
        color = (0, 0, 240)
    elif not is_tracking:
        msg = "BUSCANDO TU ROSTRO..."
        color = (0, 140, 255)
    else:
        calib_str = "CALIBRADO" if is_calibrated else "PRESIONA C"
        msg = f"TRACKING OK ({calib_str})"
        color = (0, 200, 0)

    # Semi-transparent dark pill background for readability
    overlay = canvas.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + 340, margin + 38), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.rectangle(canvas, (margin, margin), (margin + 340, margin + 38), color, 1)
    cv2.putText(canvas, msg, (margin + 12, margin + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Hotkey hints at bottom
    hud_hints = f"[C] Calibrar  |  [-/+] Sensibilidad: {driving_mult:.2f}  |  [Q] Salir"
    cv2.putText(canvas, hud_hints, (margin, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def find_default_virtual_cam():
    for node in ["/dev/video2", "/dev/video10", "/dev/video3", "/dev/video4"]:
        if os.path.exists(node):
            return node
    return "/dev/video2"


def parse_args():
    default_img = "avatar2.png" if os.path.exists("avatar2.png") else "sample_avatar.jpg"
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
    parser.add_argument("--preview", action="store_true",
                        help="Show local OpenCV preview window.")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for maximum inference speed (>40-60 FPS).")
    parser.add_argument("--driving-multiplier", "-m", type=float, default=0.65,
                        help="Facial expression and motion intensity multiplier (default: 0.65). Values between 0.50-0.70 provide stable, subtle motion.")
    parser.add_argument("--dry-run", type=float, default=0.0,
                        help="Run in benchmark mode for N seconds and exit with FPS statistics.")
    parser.add_argument("--no-virtualcam", action="store_true",
                        help="Disable virtual camera output (useful for testing without v4l2loopback).")
    return parser.parse_args()


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
    print("=" * 65)

    # 1. Initialize Webcam
    cap = open_webcam_robust(args.webcam_id, args.fps)
    if cap is None:
        if args.dry_run > 0:
            print("[*] Generating synthetic webcam stream for benchmark dry-run...")
        else:
            print(f"[!] No se pudo abrir la cámara física: {args.webcam_id}")
            sys.exit(1)

    # 2. Initialize LivePortrait Pipeline
    pipeline = LivePortraitCamPipeline(
        source_image_path=args.source_image,
        device_id=0,
        flag_pasteback=args.pasteback,
        flag_compile=args.compile,
        driving_multiplier=args.driving_multiplier
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

    # Signal handlers for clean termination
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("\n[*] Interruption caught (SIGINT). Exiting gracefully...")
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    print("\n[>>>] Pipeline running. Press 'q' to quit, 'c' to recalibrate neutral pose.\n")

    frame_count = 0
    t_start = time.perf_counter()
    latency_records = []

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

            # Local preview window if requested (includes PiP camera and status HUD)
            if args.preview:
                try:
                    preview_img = draw_preview_hud(
                        avatar_bgr=out_formatted,
                        cam_bgr=frame_bgr,
                        is_tracking=pipeline.is_tracking,
                        is_calibrated=(pipeline.x_d_0_info is not None),
                        brightness=frame_bgr.mean(),
                        driving_mult=pipeline.driving_multiplier
                    )
                    cv2.imshow("LivePortrait Avatar Preview (Press q to exit)", preview_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        pipeline.x_d_0_info = None  # Trigger recalibration
                        pipeline.last_lmk = None
                        print("[*] Calibración reseteada. Mira de frente a la cámara con pose neutral...")
                    elif key in [ord('-'), ord('_')]:
                        pipeline.driving_multiplier = max(0.20, round(pipeline.driving_multiplier - 0.05, 2))
                        print(f"[*] Sensibilidad reducida a: {pipeline.driving_multiplier:.2f}")
                    elif key in [ord('+'), ord('='), ord(']')]:
                        pipeline.driving_multiplier = min(1.20, round(pipeline.driving_multiplier + 0.05, 2))
                        print(f"[*] Sensibilidad aumentada a: {pipeline.driving_multiplier:.2f}")
                except cv2.error as e:
                    print(f"[!] Warning: GUI display error in cv2.imshow ({e}). Disabling preview.")
                    args.preview = False

            # Check dry-run duration limit
            if args.dry_run > 0 and (time.perf_counter() - t_start) >= args.dry_run:
                print(f"[*] Benchmark dry-run duration reached ({args.dry_run}s).")
                break

    finally:
        total_time = time.perf_counter() - t_start
        if cap.isOpened():
            cap.release()
        if virtual_cam is not None:
            virtual_cam.close()
        if args.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

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
