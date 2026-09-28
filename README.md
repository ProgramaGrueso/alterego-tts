# LivePortrait Real-Time Virtual Camera

Animación facial de avatares en tiempo real a partir de tu webcam física, transmitiendo la señal procesada directamente como una cámara web virtual en Linux (`v4l2loopback`) para aplicaciones como Discord, OBS, Google Meet o Zoom.

Acelerado mediante GPU NVIDIA (CUDA / Tensor Cores) y estabilizado mediante **Filtro 1€ (One-Euro Filter)** para evitar micro-temblores y latencia en el seguimiento facial.

---

## 🚀 Características

- **Streaming en tiempo real a cámara virtual**: Salida directa a nodos V4L2 (`/dev/videoX`) usando `pyvirtualcam`.
- **Filtro One-Euro adaptativo**: Suavizado cinemático de rotación de cabeza, traslación y expresiones faciales (ojos y boca).
- **Modo Pasteback**: Re-inserción del rostro animado sobre la imagen original de fondo completo.
- **Aceleración PyTorch & ONNX**: Inferencia optimizada con fallback inteligente y soporte para `torch.compile`.
- **Diagnóstico integrado**: Script `test_env.py` para validar CUDA, dispositivos de audio y cámara virtual antes de iniciar.

---

## 📋 Requisitos del Sistema

- **SO**: Linux (probado en entornos Wayland / X11).
- **GPU**: NVIDIA con soporte CUDA (ej. RTX 30xx / 40xx o superior).
- **Kernel Linux**: Módulo `v4l2loopback` para soporte de cámara virtual.
- **Python**: 3.10 o 3.11.

---

## 🛠️ Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

### 2. Configurar el Entorno Virtual
Se recomienda usar `uv` o `venv`:

```bash
# Con uv (muy rápido)
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# O con python venv estándar:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Descargar los Pesos Preentrenados (Pretrained Weights)
Los modelos pesados no se almacenan en el repositorio de Git. Descárgalos automáticamente desde Hugging Face ejecutando:

```bash
chmod +x download_weights.sh
./download_weights.sh
```

*(Esto descargará los modelos base de LivePortrait y Buffalo_L de InsightFace en `liveportrait_src/pretrained_weights/`).*

### 4. Configurar el Dispositivo de Cámara Virtual (V4L2)
Carga el módulo de kernel para crear el dispositivo virtual de video:

```bash
sudo modprobe v4l2loopback exclusive_caps=1 card_label="AI Virtual Cam"
```

Verifica el nodo creado (normalmente `/dev/video2`):
```bash
v4l2-ctl --list-devices
```

---

## 🧪 Verificación del Entorno

Antes de correr el pipeline completo, ejecuta el test de diagnóstico:

```bash
python test_env.py
```

Comprobará:
1. Reconocimiento de GPU NVIDIA y cálculo tensorial en CUDA.
2. Detección de micrófonos/dispositivos de entrada de audio (`sounddevice`).
3. Inicialización y envío de frames al nodo de cámara virtual (`pyvirtualcam`).

---

## 🎥 Uso

### Ejecución rápida (Comando Global)
Puedes iniciar la aplicación directamente desde cualquier terminal ejecutando:

```bash
avatar
```

*(Consulta [`AVATAR_CLI.md`](file:///mnt/Datos/Proyecto/avatar/AVATAR_CLI.md) para detalles completos de arquitectura y opciones).*

O invocando directamente el script con Python:

```bash
python run_live_cam.py
```

### Opciones y Parámetros Principales

```bash
python run_live_cam.py \
  --source-image "mi_avatar.png" \
  --webcam-id 0 \
  --output-device /dev/video2 \
  --fps 30 \
  --driving-multiplier 0.50 \
  --pasteback \
  --preview
```

| Parámetro | Descripción | Por defecto |
| :--- | :--- | :--- |
| `--source-image`, `-s` | Ruta de la imagen estática del avatar a animar | `sample_avatar.jpg` |
| `--webcam-id`, `-w` | ID numérico (ej. `0`) o nodo (`/dev/video0`) de tu webcam real | `0` |
| `--output-device`, `-o` | Nodo de la cámara virtual V4L2 | `/dev/video2` |
| `--fps` | Tasa de fotogramas objetivo | `30` |
| `--driving-multiplier`, `-m` | Intensidad de movimiento de cabeza y rostro (0.40 - 0.60 es óptimo) | `0.50` |
| `--lip-multiplier` | Articulación de labios (fonemas/habla) | `1.00` |
| `--pasteback` | Re-inserta el rostro en el retrato completo con fondo (alpha blend con feather en GPU) | Desactivado |
| `--seamless` | Con `--pasteback`: usa `cv2.seamlessClone` a resolución completa (lento, ~0.7 FPS con avatares 2048 px) | Desactivado |
| `--no-preview` | Desactiva la ventana local de OpenCV (el preview está activo por defecto) | Preview activo |
| `--compile` | Activa `torch.compile` (requiere Triton instalado en el venv) | Desactivado |
| `--no-virtualcam` | Ejecuta solo en ventana sin requerir el módulo v4l2 | Desactivado |
| `--backend` | Backend de inferencia: `torch` o `trt` (motores TensorRT, ver abajo) | `torch` |
| `--trt-engine-dir` | Carpeta con los motores `.trt` y el plugin GridSample3D | `liveportrait_src/pretrained_weights/faster_liveportrait/trt_engines` |
| `--detector` | Detector para (re)adquirir el rostro: `insightface` o `mediapipe` | `insightface` |
| `--driving-video` | Usa un vídeo en bucle (a su FPS nativo) en lugar de la webcam; benchmarks reproducibles | — |
| `--dry-run N` | Modo benchmark: corre N segundos e imprime el resumen de rendimiento | `0` (desactivado) |

---

## ⚡ Backend TensorRT (`--backend trt`, opcional)

Sustituye motion extractor, warping+SPADE, stitching/retargeting y landmark por motores
TensorRT 8.6 generados con la imagen Docker de [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait).
El backend `torch` sigue siendo el predeterminado. En una RTX 4070 Ti Super el warping+SPADE
pasa de ~33 ms a ~14.5 ms y el pipeline completo llega a 30 FPS (límite de la webcam).

> **No instales TensorRT con pacman**: Arch/CachyOS trae TensorRT 10.x y los motores y el
> plugin `GridSample3D` de FasterLivePortrait requieren TensorRT 8.x.

### 1. Docker + nvidia-container-toolkit (CachyOS)

```bash
sudo pacman -S --needed docker nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER   # cierra sesión y vuelve a entrar
docker pull shaoguo/faster_liveportrait:v3   # ~14 GB descargados, ~39 GB en disco
```

Si `docker run --gpus=all ...` falla con `failed to fulfil mount request: open /usr/lib/libnvidia-...so.X: no such file or directory`,
la especificación CDI quedó desactualizada tras actualizar el driver NVIDIA. Regenérala:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### 2. Descargar los ONNX y generar los motores

```bash
.venv/bin/hf download warmshao/FasterLivePortrait --include "liveportrait_onnx/*" \
    --local-dir liveportrait_src/pretrained_weights/faster_liveportrait
scripts/build_trt_engines.sh   # ~7 min; deja los .trt en .../faster_liveportrait/trt_engines/
```

`scripts/build_trt_engines.sh` ejecuta `scripts/onnx2trt.py` dentro del contenedor (TensorRT 8.6.1.6).
Es una variante del conversor de FasterLivePortrait que construye InstanceNorm como capa nativa y sin
tácticas cuDNN, para que el runtime del host no necesite cuDNN 8 completo. Los motores dependen de la GPU
y de la versión de TensorRT: regenéralos si cambias de tarjeta.

### 3. Runtime TensorRT 8.6 en el venv (sin tocar el sistema)

```bash
uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match "tensorrt-bindings==8.6.1" "tensorrt-libs==8.6.1"

# libnvinfer_plugin.so.8 enlaza contra libcudnn.so.8, pero PyTorch usa cuDNN 9.
# Se extrae SOLO el loader de cuDNN 8 (142 KB) a una carpeta privada que trt_backend.py precarga:
D=liveportrait_src/pretrained_weights/faster_liveportrait/trt_libs; mkdir -p $D
curl -L -o /tmp/cudnn8.whl https://files.pythonhosted.org/packages/79/fb/8071b1c82db9b38dccbd11dfaa16aaa4b239d2deff5af711d8b01467554f/nvidia_cudnn_cu12-8.9.7.29-py3-none-manylinux1_x86_64.whl
unzip -j -o /tmp/cudnn8.whl nvidia/cudnn/lib/libcudnn.so.8 -d $D && rm /tmp/cudnn8.whl
```

Verificación (compara salidas y latencias TRT vs PyTorch):

```bash
.venv/bin/python bench/test_trt_parity.py
python run_live_cam.py --backend trt
```

### Ejecutar dentro del contenedor (alternativa, no probada con este repo)

Si prefieres trabajar dentro de la imagen, pasa la webcam, la cámara virtual y la GPU:

```bash
docker run -it --rm --gpus=all \
    --device /dev/video0 --device /dev/video2 \
    -v "$PWD":/root/avatar shaoguo/faster_liveportrait:v3 /bin/bash
```

La imagen no trae las dependencias de este proyecto (pyvirtualcam, etc.); la vía probada es la nativa descrita arriba.

---

## 🧑 Detector MediaPipe (`--detector mediapipe`, opcional)

Usa el FaceLandmarker de MediaPipe como prior para (re)adquirir el rostro en lugar de InsightFace.
El seguimiento frame a frame no cambia (sigue siendo el LandmarkRunner); solo acelera la re-adquisición
tras perder la cara o recalibrar (~17.6 ms vs ~47.8 ms, `bench/bench_detectors.py`).

```bash
uv pip install --python .venv/bin/python mediapipe
mkdir -p liveportrait_src/pretrained_weights/mediapipe
curl -L -o liveportrait_src/pretrained_weights/mediapipe/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

---

## 📊 Benchmarks

`bench/run_bench.sh <etiqueta> [flags]` ejecuta el benchmark estándar (30 s, sin cámara virtual ni preview)
con `d6.mp4` en bucle a 30 FPS como webcam y guarda el resumen en `bench/results/<etiqueta>.txt`.
Latencias medidas desde que el frame está disponible (RTX 4070 Ti Super, avatar 2048x2048):

| Configuración | FPS | Latencia media | Latencia p95 | Inferencia |
| :--- | ---: | ---: | ---: | ---: |
| Original, recorte | 13.99 | 71.45 ms | 70.00 ms | 68.43 ms |
| Original, `--pasteback` (seamlessClone) | 0.66 | 1509.82 ms | 1656.69 ms | 1499.66 ms |
| Fase 1–2, recorte (torch) | 19.86 | 50.31 ms | 50.63 ms | 47.3 ms |
| Fase 2, `--pasteback` GPU (torch) | 20.99 | 47.61 ms | 48.47 ms | 47.58 ms |
| Fase 3, recorte `--backend trt` | 29.91 | 26.18 ms | 27.79 ms | 23.17 ms |
| Fase 3, `--pasteback --backend trt` | 29.90 | 22.94 ms | 23.47 ms | 22.90 ms |

Con TensorRT el límite pasa a ser la fuente (30 FPS). Otros tests: `bench/test_face_loss.py`
(continuidad al perder/recuperar el rostro), `bench/test_trt_parity.py`, `bench/bench_detectors.py`.

---

## 📂 Estructura del Proyecto

```text
.
├── download_weights.sh      # Script para descargar pesos de Hugging Face
├── requirements.txt         # Dependencias del proyecto
├── run_live_cam.py          # Pipeline principal de inferencia en tiempo real
├── trt_backend.py           # Backend TensorRT opcional (--backend trt)
├── scripts/                 # Generación de motores TensorRT (Docker)
├── bench/                   # Benchmarks, tests funcionales y resultados por fase
├── test_env.py              # Script de validación de hardware y v4l2
├── liveportrait_src/        # Código fuente del motor LivePortrait
│   ├── pretrained_weights/  # Pesos preentrenados (gestionados vía Hugging Face)
│   └── src/                 # Módulos de animación, cropper y retargeting
├── sample_avatar.jpg        # Imagen de prueba
└── .gitignore               # Configuración de exclusión para Git
```

---

## 📄 Licencia

Este proyecto integra modelos y herramientas bajo sus respectivas licencias de investigación y uso de código abierto (LivePortrait / InsightFace).
