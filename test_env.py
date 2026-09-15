#!/usr/bin/env python3
"""
test_env.py - Script de verificación para entorno Deep Learning / Real-time Lip-sync
Verifica:
1. PyTorch + CUDA + GPU (Ada Lovelace / sm_89)
2. sounddevice (detección de micrófonos/dispositivos de entrada)
3. pyvirtualcam (emisión a dispositivo V4L2 virtual)
"""

import sys
import torch
import sounddevice as sd

def test_cuda():
    print("=" * 60)
    print("1. COMPROBACIÓN DE PYTORCH Y CUDA")
    print("=" * 60)
    print(f"PyTorch versión: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        cuda_version = torch.version.cuda
        print(f"Dispositivos CUDA detectados: {device_count}")
        print(f"Nombre de GPU: {device_name}")
        print(f"Versión de CUDA (PyTorch): {cuda_version}")
        print(f"Compute Capability: sm_{capability[0]}{capability[1]}")
        
        # Test tensor allocation and execution on GPU
        x = torch.randn((1000, 1000), device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print("Test de cálculo tensorial en GPU: EXITOSO")
    else:
        print("[!] CUDA NO está disponible para PyTorch.")
    print()

def test_audio():
    print("=" * 60)
    print("2. COMPROBACIÓN DE DISPOSITIVOS DE AUDIO (sounddevice)")
    print("=" * 60)
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        print(f"Dispositivo de entrada predeterminado (ID): {default_in}")
        
        input_devices = [
            (i, d["name"], d["max_input_channels"], d["default_samplerate"])
            for i, d in enumerate(devices)
            if d.get("max_input_channels", 0) > 0
        ]
        
        if input_devices:
            print(f"Se encontraron {len(input_devices)} dispositivo(s) de entrada:")
            for idx, name, channels, srate in input_devices:
                pref = "-> [DEFAULT]" if idx == default_in else "  "
                print(f"{pref} ID {idx}: {name} ({channels} canales, {int(srate)} Hz)")
        else:
            print("[!] No se encontraron dispositivos de entrada de audio activos.")
    except Exception as e:
        print(f"[!] Error al consultar sounddevice: {e}")
    print()

def test_virtualcam():
    print("=" * 60)
    print("3. COMPROBACIÓN DE CÁMARA VIRTUAL (pyvirtualcam)")
    print("=" * 60)
    import numpy as np
    
    try:
        import pyvirtualcam
        
        # Intento de instanciar un frame buffer de prueba (640x480, 30 fps, BGR/RGB)
        print("Intentando inicializar cámara virtual...")
        with pyvirtualcam.Camera(width=640, height=480, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
            print(f"Cámara virtual abierta con éxito en el nodo: {cam.device}")
            print(f"Resolución configurada: {cam.width}x{cam.height} @ {cam.fps} FPS")
            
            # Enviar 5 frames de prueba
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_frame[:] = (0, 255, 0)  # Frame verde
            for _ in range(5):
                cam.send(dummy_frame)
                cam.sleep_until_next_frame()
            print("Envío de frames de prueba: EXITOSO")
            
    except ModuleNotFoundError:
        print("[!] pyvirtualcam no está instalado.")
    except RuntimeError as e:
        print(f"[!] No se pudo abrir la cámara virtual: {e}")
        print("    Causa probable: El módulo de kernel 'v4l2loopback' no ha sido cargado aún.")
        print("    Ejecuta: sudo modprobe v4l2loopback exclusive_caps=1 card_label=\"AI Virtual Cam\"")
    except Exception as e:
        print(f"[!] Error inesperado al probar pyvirtualcam: {e}")
    print()

if __name__ == "__main__":
    print(f"Entorno Python: {sys.executable}\n")
    test_cuda()
    test_audio()
    test_virtualcam()
