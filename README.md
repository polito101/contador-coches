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
| `--line`          | `None`         | Segmento de conteo `'x1,y1,x2,y2'`. Activa modo línea + dirección.       |
| `--line-y`        | `None`         | Atajo: segmento horizontal a lo ancho de toda la imagen en la fila Y.    |
| `--line-x`        | `None`         | Atajo: segmento vertical a lo alto de toda la imagen en la columna X.    |
| `--preview-frame` | off            | Guarda el primer frame como PNG y sale (para elegir coordenadas a ojo).  |
| `--pick-line`     | off            | Abre el primer frame, dibujas el segmento con click-and-drag.            |
| `--lines-file`    | `lines.json`   | Archivo donde se cachean los segmentos por fuente.                       |
| `--no-cached-line`| off            | Ignora cualquier segmento cacheado para esta fuente.                     |
| `--model`         | `yolov8n.pt`   | Pesos del modelo (`yolov8s.pt` más preciso, más lento)                   |
| `--output-dir`    | `output`       | Carpeta de salida                                                        |
| `--no-save`       | off            | No guardar vídeo ni JSON                                                 |
| `--no-display`    | off            | No abrir ventana (headless)                                              |

### Modos de conteo

**Sin línea (default):** cuenta IDs únicos del tracker. Usa `--conf` y `--min-frames` para filtrar falsos positivos.

**Con línea (`--line`, `--line-y`, `--line-x` o `--pick-line`):** solo cuenta vehículos cuyo movimiento cruza el **segmento** entre A y B. Mucho más preciso y reporta dirección.

**Opción rápida — picker interactivo:**

```powershell
.\.venv\Scripts\python.exe contar.py --source "URL" --pick-line --duration 30
```

Se abre el primer frame en una ventana. **Mantén pulsado** el botón del ratón en A y **arrastra** hasta B; ves un preview amarillo del segmento real (no se extiende a toda la pantalla). **Suelta** para fijar. **Enter** confirma, **R** rehace, **Esc** cancela.

El segmento se **guarda automáticamente** en `lines.json` asociado a la URL/path de `--source`. Próximas ejecuciones con esa misma fuente reutilizan el segmento sin abrir el picker:

```powershell
# 1ª vez: dibujas
.\.venv\Scripts\python.exe contar.py --source "URL" --pick-line --duration 30

# Siguientes: no flags de línea, se carga sola
.\.venv\Scripts\python.exe contar.py --source "URL" --duration 30
# → "Using cached segment from lines.json: (120, 300) -> (520, 310)"
```

Para rehacerla, lanza `--pick-line` otra vez (se sobrescribe). Para ignorar la caché, `--no-cached-line`.

**Opción manual — segmento explícito:**

```powershell
# 1. Guarda el primer frame
.\.venv\Scripts\python.exe contar.py --source "URL" --preview-frame

# 2. Abre output/preview_*.png, anota los píxeles de A y B
# 3. Lanza con esas coordenadas
.\.venv\Scripts\python.exe contar.py --source "URL" --line "120,300,520,310" --duration 30
```

Los atajos `--line-y N` y `--line-x N` dibujan segmentos que ocupan todo el ancho/alto.

La dirección reportada depende de la orientación dominante del segmento:
- segmento mayoritariamente horizontal → `down`/`up`
- segmento mayoritariamente vertical → `right`/`left`

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

## Versioning discipline

`contador-coches` se distribuye como paquete instalable (`pip install
git+https://.../contador-coches.git@v0.1.0`) y `live-bets` lo pinea por **tag
inmutable**. La disciplina de versionado es estricta porque downstream
(live-bets) construye `indexer_version = "<package_semver>+<model_basename>"`
y ese string entra en el `clip_id` content-addressed (D-06, D-08): un cambio
silencioso en `min_frames`, `conf` o el modelo YOLO sin bump de semver
produciría dos clips con `clip_id` idéntico y resultados distintos —
romperíamos auditoría e idempotencia.

### Reglas

1. **Bump de semver obligatorio** cuando cualquiera de estas cosas cambia:
   - El valor por defecto de `min_frames` o `conf` en
     `count_vehicles_minframes` / `count_vehicles_linecrossing`.
   - El archivo binario de los pesos YOLO bundled
     (`contador_coches/weights/yolov8n.pt`).
   - La firma pública de `count_vehicles_minframes`,
     `count_vehicles_linecrossing`, `resolve_source` o
     `get_default_model_path`.
   - El output JSON shape (campos `total`, `breakdown`, `frames_processed`,
     `duration_real`, etc.).
2. **Los tags git nunca se mueven.** Bump = nuevo tag (`v0.1.1`, `v0.2.0`).
   Mover un tag existente rompería la trazabilidad de cualquier clip ya
   indexado contra esa versión (RESEARCH §1 Pitfall #1, threat T-02-04).
3. **Bump major (`v1.0.0`)** cuando la firma pública o el output shape
   rompe compatibilidad. Bump minor (`v0.2.0`) cuando añades funcionalidad
   manteniendo backwards-compat. Bump patch (`v0.1.1`) para fixes que NO
   cambien el output para el mismo input — si cambia el output, NO es patch.

### Workflow

```bash
# 1. Edita el código + actualiza version en pyproject.toml + __init__.py
# 2. Commit
git add -A
git commit -m "feat: bump to 0.2.0 — new min_frames default"
# 3. Tag (anotado, nunca lightweight)
git tag -a v0.2.0 -m "v0.2.0 — semver bump per versioning discipline"
# 4. (Opcional) push si hay remote configurado
# git push origin master v0.2.0
```

### Consumo desde live-bets

`live-bets/pyproject.toml` declara la dependencia con el tag pineado:

```toml
dependencies = [
    "contador-coches @ git+file:///C/Users/pobom/contador-coches@v0.1.0",
    # En producción: git+https://github.com/<user>/contador-coches.git@v0.1.0
]
```

Cambiar el pin (`@v0.1.0` → `@v0.2.0`) es una decisión deliberada del
consumer; un `git push` que mueva el tag rompería el contrato.
