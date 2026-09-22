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

*(Consulta [`AVATAR_CLI.md`](file:///run/media/juang/Datos/Proyecto/avatar/AVATAR_CLI.md) para detalles completos de arquitectura y opciones).*

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
| `--pasteback` | Re-inserta el rostro en el retrato completo con fondo | Desactivado |
| `--preview` | Muestra una ventana local de OpenCV para previsualización | Desactivado |
| `--compile` | Activa `torch.compile` para máxima tasa de FPS | Desactivado |
| `--no-virtualcam` | Ejecuta solo en ventana sin requerir el módulo v4l2 | Desactivado |

---

## 📂 Estructura del Proyecto

```text
.
├── download_weights.sh      # Script para descargar pesos de Hugging Face
├── requirements.txt         # Dependencias del proyecto
├── run_live_cam.py          # Pipeline principal de inferencia en tiempo real
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
