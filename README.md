# Contador de coches

Cuenta vehículos únicos que pasan por una cámara o vídeo durante una ventana de tiempo configurable. Usa YOLOv8 + ByteTrack para detección y seguimiento.

## Requisitos

- Python 3.10+
- ffmpeg en el PATH (necesario para descargar streams de YouTube)
- (Opcional) GPU NVIDIA con CUDA — mucho más rápido

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

La primera ejecución descarga el modelo YOLO (~6 MB) automáticamente.

### GPU NVIDIA (opcional, ~10x más rápido)

`pip install -r requirements.txt` instala torch en CPU. Para usar tu GPU:

```powershell
.\.venv\Scripts\pip.exe uninstall -y torch torchvision
.\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verifica con:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

## Uso

```powershell
# Livestream de YouTube (graba el trozo con yt-dlp y procesa)
.\.venv\Scripts\python.exe contar.py --source "https://www.youtube.com/watch?v=VIDEO_ID"

# Archivo local
.\.venv\Scripts\python.exe contar.py --source ruta\al\video.mp4

# URL .mp4 directa (jamcams TfL, etc.)
.\.venv\Scripts\python.exe contar.py --source "https://.../traffic.mp4"

# Cámara IP por RTSP
.\.venv\Scripts\python.exe contar.py --source "rtsp://usuario:pass@192.168.1.10:554/stream1"

# Webcam del portátil (índice 0)
.\.venv\Scripts\python.exe contar.py --source 0
```

Pulsa `q` mientras corre para terminar antes.

### Flags

| Flag              | Default        | Descripción                                                              |
|-------------------|----------------|--------------------------------------------------------------------------|
| `--source`        | (obligatorio)  | URL/path/índice de webcam                                                |
| `--duration`      | `30`           | Segundos a procesar                                                      |
| `--conf`          | `0.5`          | Confianza mínima de detección (0..1)                                     |
| `--min-frames`    | `5`            | Frames mínimos para contar un track (modo sin línea)                     |
| `--line-y`        | `None`         | Píxel Y de la línea horizontal de conteo. Activa modo línea + dirección. |
| `--line-x`        | `None`         | Píxel X de la línea vertical. Mutuamente excluyente con `--line-y`.      |
| `--preview-frame` | off            | Guarda el primer frame como PNG y sale. Usar para elegir `--line-y/x`.   |
| `--model`         | `yolov8n.pt`   | Pesos del modelo (`yolov8s.pt` más preciso, más lento)                   |
| `--output-dir`    | `output`       | Carpeta de salida                                                        |
| `--no-save`       | off            | No guardar vídeo ni JSON                                                 |
| `--no-display`    | off            | No abrir ventana (headless)                                              |

### Modos de conteo

**Sin línea (default):** cuenta IDs únicos del tracker. Usa `--conf` y `--min-frames` para filtrar falsos positivos.

**Con línea (`--line-y` o `--line-x`):** solo cuenta vehículos que cruzan una línea virtual. Mucho más preciso y reporta dirección.

Para elegir el píxel de la línea:

```powershell
# 1. Guarda el primer frame
.\.venv\Scripts\python.exe contar.py --source "URL" --preview-frame

# 2. Abre output/preview_*.png, mira el píxel Y o X donde quieres la línea
# 3. Vuelve a lanzar con esa coordenada
.\.venv\Scripts\python.exe contar.py --source "URL" --line-y 400 --duration 30
```

La salida en modo línea desglosa por dirección:
- `--line-y`: `down` (de arriba a abajo) y `up` (al revés)
- `--line-x`: `right` (de izquierda a derecha) y `left` (al revés)

### Ajustando precisión (modo sin línea)

- Si cuenta **demasiados**: subir `--conf 0.6` y/o `--min-frames 10`.
- Si cuenta **muy pocos**: bajar `--conf 0.4` y/o `--min-frames 3`.

## Salida

- `output/YYYY-MM-DD_HH-MM-SS.mp4` — vídeo anotado con cajas e IDs.
- `output/YYYY-MM-DD_HH-MM-SS.json` — total, desglose por clase, IDs únicos, `min_frames` usado.
- `output/buffer_YYYY-MM-DD_HH-MM-SS.ts` — solo para YouTube: trozo descargado del live.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
