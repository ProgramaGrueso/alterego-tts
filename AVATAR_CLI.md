# Implementación del Comando Global `avatar`

Este documento detalla los cambios realizados para permitir el inicio inmediato de la aplicación escribiendo simplemente `avatar` en cualquier terminal del sistema, explicando **qué** se hizo, **cómo** se implementó y **por qué** se tomaron estas decisiones de diseño.

---

## 1. Resumen de lo Realizado

Se configuró un comando ejecutable global en el sistema (`avatar`) que:
1. Permite iniciar el pipeline de animación facial en tiempo real desde cualquier terminal y cualquier ruta de trabajo.
2. Utiliza automáticamente el entorno virtual aislado (`.venv`) con PyTorch, CUDA y todas las dependencias sin requerir activación manual (`source .venv/bin/activate`).
3. Abre por defecto la ventana interactiva de previsualización con HUD y atajos de teclado (`--preview` activo por omisión, desactivable con `--no-preview`).
4. Resuelve de forma inteligente la imagen estática del avatar (`avatar.png` prioritario, fallback a `sample_avatar.jpg`), incluso si el comando se invoca fuera del directorio del repositorio.
5. Detecta proactivamente si el módulo de kernel para cámara virtual (`v4l2loopback`) está cargado y ofrece instrucciones de carga sin interrumpir el funcionamiento local.

---

## 2. Cómo se Implementó (Arquitectura y Código)

### A. Lanzador en el PATH del Usuario (`~/.local/bin/avatar`)

Se creó un script ejecutable Bash en [`/home/juang/.local/bin/avatar`](file:///home/juang/.local/bin/avatar):

```bash
#!/usr/bin/env bash
set -e

PROJECT_DIR="/run/media/juang/Datos/Proyecto/avatar"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
MAIN_SCRIPT="$PROJECT_DIR/run_live_cam.py"

# 1. Validación del entorno virtual
if [ ! -x "$VENV_PYTHON" ]; then
    echo -e "\033[1;31m[!] Error:\033[0m Intérprete de Python no encontrado en: $VENV_PYTHON" >&2
    exit 1
fi

# 2. Compatibilidad con Wayland / XWayland
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# 3. Carga automática de la cámara virtual (v4l2loopback)
if ! lsmod | grep -q "^v4l2loopback " 2>/dev/null; then
    case "$*" in
        *--help*|*-h*|*--no-virtualcam*) ;;
        *)
            echo -e "\033[1;34m[i]\033[0m Activando cámara virtual (v4l2loopback)..."
            if sudo modprobe v4l2loopback exclusive_caps=1 card_label="AI Virtual Cam"; then
                echo -e "\033[1;32m[+]\033[0m Dispositivo 'AI Virtual Cam' activado con éxito.\n"
            else
                echo -e "\033[1;33m[!]\033[0m No se pudo cargar 'v4l2loopback'. Iniciando sin cámara virtual...\n" >&2
            fi
            ;;
    esac
fi

# 4. Reemplazo de proceso
exec "$VENV_PYTHON" "$MAIN_SCRIPT" "$@"
```

Se le otorgaron permisos de ejecución mediante:
```bash
chmod +x /home/juang/.local/bin/avatar
```

### B. Ajustes en [`run_live_cam.py`](file:///run/media/juang/Datos/Proyecto/avatar/run_live_cam.py)

1. **Resolución de ruta del avatar canónico**:
   Se implementó la función `find_default_source_image()` para localizar de manera absoluta los archivos de imagen dentro del proyecto (`avatar.png`, `avatar2.png` o `sample_avatar.jpg`), evitando errores de archivo no encontrado al ejecutar desde directorios ajenos (como `~` o `/tmp`).

2. **Resolución bidireccional de imágenes personalizadas**:
   En `LivePortraitCamPipeline.__init__`, si el usuario pasa una ruta relativa mediante `-s imagen.jpg`, el sistema verifica primero si existe en el directorio de trabajo actual (`CWD`) y, si no, busca en el directorio del proyecto (`CURRENT_DIR`).

3. **Previsualización por defecto y soporte de `--no-preview`**:
   Se actualizó el argumento `--preview` mediante `argparse.BooleanOptionalAction` con `default=True`. Esto garantiza que `avatar` abra la ventana interactiva automáticamente y a su vez soporte `--no-preview` para entornos headless.

---

## 3. Justificación Técnica: ¿Por qué se hizo así?

### 1. ¿Por qué un binario en `~/.local/bin` en lugar de un `alias` en el shell?
* **Compatibilidad Multi-Shell**: Tu shell principal es `fish` (`/bin/fish`), pero los entornos pueden invocar `bash`, `zsh` o subprocesos. Un `alias` en `.bashrc` o una función en `config.fish` queda restringido exclusivamente a esa shell interactiva.
* **Estándar XDG / POSIX**: El directorio `~/.local/bin` forma parte del estándar de ejecutables locales de usuario en Linux y ya se encuentra configurado en tu `$PATH` global. Cualquier terminal, script o lanzador gráfico puede encontrar el comando inmediatamente.

### 2. ¿Por qué el uso de `exec` en el script Bash?
* **Gestión limpia de señales Unix**: Al usar `exec`, el intérprete de Python reemplaza directamente al proceso Bash en lugar de ejecutarse como un proceso hijo.
* **Liberación de memoria y hardware**: Cuando presionas `Ctrl + C` o envías un `SIGINT` / `SIGTERM`, la señal llega directamente a Python sin intermediarios, permitiendo que los bloques `finally` de OpenCV (`cap.release()`, `destroyAllWindows()`) y de CUDA limpien la VRAM de la GPU y liberen el nodo de la cámara física.

### 3. ¿Por qué invocar directamente `$PROJECT_DIR/.venv/bin/python`?
* **Aislamiento sin efectos secundarios**: No se contamina el entorno global de la sesión ni se requiere modificar el estado del shell activo con `source .venv/bin/activate`. El ejecutable de Python dentro del virtualenv ya tiene configurados internamente sus `sys.prefix` y rutas a `site-packages`.

### 4. ¿Por qué `QT_QPA_PLATFORM=xcb`?
* **Estabilidad en Wayland**: En distribuciones Linux modernas con Wayland (como CachyOS / Arch / Fedora), los backends GUI de OpenCV basados en Qt pueden fallar al intentar comunicarse con el compositor Wayland nativo. Configurar `xcb` fuerza el renderizado a través de XWayland, garantizando que la ventana de previsualización abra de forma consistente y sin bloqueos.

### 5. ¿Por qué aviso informativo no bloqueante para `v4l2loopback`?
* **Tolerancia a fallos**: No todos los casos de uso requieren transmitir a una reunión virtual; a veces solo se desea probar expresiones o calibrar el avatar localmente. Bloquear el arranque si falta el módulo de kernel perjudicaría la experiencia de usuario. El script advierte claramente cómo activarlo con `sudo modprobe` pero continúa con la ejecución normal.

---

## 4. Ejemplos de Uso

Desde cualquier terminal o carpeta:

```bash
# 1. Uso estándar (inicia con avatar por defecto y ventana de previsualización)
avatar

# 2. Con imagen de avatar personalizada
avatar -s ~/Pictures/mi_retrato.png

# 3. Modo headless (sin ventana gráfica, solo emisión a cámara virtual)
avatar --no-preview

# 4. Modo alta velocidad con torch.compile
avatar --compile

# 5. Con re-inserción de cuerpo/fondo completo (pasteback)
avatar --pasteback

# 6. Ajustar sensibilidad de movimiento y apertura de labios (base: 1.50)
avatar -m 0.65 --lip-multiplier 1.80

# 7. Ver todas las opciones disponibles
avatar --help
```

### Controles durante la ejecución (en la ventana de previsualización)
* **`C`**: Recalibrar posición neutral (mira a la cámara con boca relajada y cerrada).
* **`+` / `-`**: Aumentar o disminuir la sensibilidad de giro de cabeza.
* **`,` / `.`**: Aumentar o disminuir la sensibilidad de apertura de labios y boca.
* **`Q`**: Salir y cerrar la aplicación limpiamente.
