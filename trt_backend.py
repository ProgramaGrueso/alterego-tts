#!/usr/bin/env python3
"""
trt_backend.py - Backend TensorRT opcional para run_live_cam.py (--backend trt).

Carga motores TensorRT 8.x generados con la imagen Docker de FasterLivePortrait
(ver README, sección "Backend TensorRT") y sustituye, con la misma interfaz, los
módulos PyTorch equivalentes del LivePortraitWrapper:

  motion_extractor.trt          -> wrapper.motion_extractor
  warping_spade-fix.trt         -> wrapper.warp_decode (warping + SPADE en un motor)
  stitching{,_eye,_lip}.trt     -> wrapper.stitching_retargeting_module[...]
  landmark.trt                  -> cropper.human_landmark_runner.session

El resto del pipeline (filtros, retargeting, pasteback) no cambia. Los motores
TRT solo se pueden cargar con la MISMA versión de TensorRT con la que se crearon,
por eso el runtime es el wheel pip `tensorrt` 8.x instalado en el venv (no pacman).
"""

import ctypes
import os
import sys

import numpy as np
import torch

trt = None
_TRT_TO_TORCH = {}
_LIB_HANDLES = []


def _load_trt_libs(private_lib_dir: str):
    """
    Importa TensorRT 8.6 (wheels pip tensorrt-bindings/tensorrt-libs) evitando tocar las
    librerías CUDA de PyTorch: libnvinfer_plugin.so.8 exige libcudnn.so.8, pero torch usa
    cuDNN 9, así que se precarga un cuDNN 8 privado (RTLD_GLOBAL) antes del import. Las
    sublibrerías de inferencia las usa el plugin InstanceNormalization_TRT del warping.
    """
    global trt
    if trt is not None:
        return
    for name in ("libcudnn_ops_infer.so.8", "libcudnn_cnn_infer.so.8", "libcudnn.so.8"):
        lib = os.path.join(private_lib_dir, name)
        if os.path.exists(lib):
            _LIB_HANDLES.append(ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL))
    try:
        import tensorrt as _trt
    except ImportError:
        try:
            import tensorrt_bindings as _trt
        except ImportError as e:
            raise RuntimeError(
                f"No se pudo importar TensorRT 8.6 ({e}). Instálalo en el venv y extrae libcudnn.so.8 "
                f"en {private_lib_dir} (ver README, sección 'Backend TensorRT')."
            ) from e
    trt = _trt
    if not _TRT_TO_TORCH:
        _TRT_TO_TORCH.update({
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        })


class TRTEngine:
    """Motor TensorRT con buffers de E/S preasignados como tensores torch en GPU."""

    def __init__(self, engine_path: str, logger):
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"No se pudo deserializar {engine_path}. ¿Se generó con otra versión de TensorRT "
                f"(runtime actual: {trt.__version__})?"
            )
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.buffers = [], [], {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[-1])
                self.context.set_input_shape(name, shape)
            dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            # Tensores normales (no "inference tensors") para poder escribirlos dentro y fuera
            # de torch.inference_mode(): el LandmarkRunner se llama fuera de él.
            with torch.inference_mode(False):
                buf = torch.empty(shape, dtype=dtype, device="cuda")
            self.buffers[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

    def __call__(self, feed: dict) -> dict:
        """feed: nombre -> tensor (se copia al buffer, convirtiendo dtype). Devuelve los buffers de salida."""
        for name, value in feed.items():
            with torch.no_grad():
                self.buffers[name].copy_(value.reshape(self.buffers[name].shape))
        # Mismo stream que torch: el orden respecto a las operaciones torch queda garantizado
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT: fallo en execute_async_v3")
        return {name: self.buffers[name] for name in self.outputs}


class MotionExtractorTRT:
    """Sustituto de wrapper.motion_extractor: devuelve el mismo dict de salidas crudas."""

    def __init__(self, engine: TRTEngine):
        self.engine = engine
        self.in_name = engine.inputs[0]

    def __call__(self, x):
        out = self.engine({self.in_name: x})
        return {k: v.clone() for k, v in out.items()}


class MLPTRT:
    """Sustituto de los MLP de stitching / retargeting de ojos y labios."""

    def __init__(self, engine: TRTEngine):
        self.engine = engine
        self.in_name = engine.inputs[0]
        self.out_name = engine.outputs[0]

    def __call__(self, feat):
        return self.engine({self.in_name: feat})[self.out_name].to(feat.dtype)

    # El wrapper llama a .half() sobre los módulos; aquí no aplica
    def half(self):
        return self


class WarpDecodeTRT:
    """Sustituto de wrapper.warp_decode (warping + SPADE en un único motor)."""

    def __init__(self, engine: TRTEngine):
        self.engine = engine
        self._feature_ptr = None

    def __call__(self, feature_3d, kp_source, kp_driving):
        feed = {"kp_source": kp_source, "kp_driving": kp_driving}
        # feature_3d (8 MB) es constante por avatar: solo se copia cuando cambia
        if feature_3d.data_ptr() != self._feature_ptr:
            feed["feature_3d"] = feature_3d
            self._feature_ptr = feature_3d.data_ptr()
        out = self.engine(feed)["out"]
        return {"out": out.float()}


class LandmarkSessionTRT:
    """Sustituto de la onnxruntime.InferenceSession del LandmarkRunner (misma API run())."""

    def __init__(self, engine: TRTEngine, output_order):
        self.engine = engine
        self.output_order = output_order

    def run(self, _output_names, feed):
        feed_t = {k: torch.from_numpy(np.ascontiguousarray(v)).cuda() for k, v in feed.items()}
        out = self.engine(feed_t)
        return [out[name].float().cpu().numpy() for name in self.output_order]


def install_trt_backend(wrapper, cropper, engine_dir: str, plugin_path: str = None, use_landmark: bool = True):
    """Reemplaza in-place los módulos del wrapper/cropper por motores TensorRT."""
    _load_trt_libs(os.path.join(os.path.dirname(os.path.abspath(engine_dir)), "trt_libs"))
    logger = trt.Logger(trt.Logger.ERROR)
    plugin_path = plugin_path or os.path.join(engine_dir, "libgrid_sample_3d_plugin.so")
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Plugin GridSample3D no encontrado: {plugin_path}")
    _LIB_HANDLES.append(ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL))
    trt.init_libnvinfer_plugins(logger, "")

    def load(name):
        path = os.path.join(engine_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Motor TensorRT no encontrado: {path}")
        return TRTEngine(path, logger)

    wrapper.motion_extractor = MotionExtractorTRT(load("motion_extractor.trt"))
    wrapper.warp_decode = WarpDecodeTRT(load("warping_spade-fix.trt"))
    if wrapper.stitching_retargeting_module is not None:
        wrapper.stitching_retargeting_module = {
            "stitching": MLPTRT(load("stitching.trt")),
            "eye": MLPTRT(load("stitching_eye.trt")),
            "lip": MLPTRT(load("stitching_lip.trt")),
        }
    if use_landmark:
        runner = cropper.human_landmark_runner
        order = [o.name for o in runner.session.get_outputs()]
        runner.session = LandmarkSessionTRT(load("landmark.trt"), order)
    print(f"[+] Backend TensorRT {trt.__version__} activo (motores en {engine_dir}).", file=sys.stderr)
