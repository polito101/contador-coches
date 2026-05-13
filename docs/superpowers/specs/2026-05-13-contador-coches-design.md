# Contador de vehículos por cámara — Diseño

**Fecha:** 2026-05-13
**Autor:** Pol Bonet (con Claude)
**Estado:** Aprobado para implementación

## Objetivo

Programa que se conecta a una cámara (RTSP, archivo o URL HTTP de vídeo), procesa 30 segundos de imagen, cuenta cuántos vehículos únicos han pasado y deja un vídeo de salida para verificación manual.

## Alcance

- Una sola cámara/stream por ejecución.
- Ventana de tiempo fija: 30 segundos (configurable por flag).
- Cuenta agregada de **todos los vehículos** (coches, motos, camiones, autobuses) en un único total. Internamente desglosado por clase, pero el resultado principal es un total.
- Ejecución local en el ordenador del usuario (Windows, GPU NVIDIA disponible).

## Fuera de alcance (no v1)

- Conteo por línea virtual / dirección del tráfico.
- Multi-cámara simultánea.
- API/servicio web.
- Reentrenamiento del modelo o clases personalizadas.
- Persistencia en base de datos.

## Stack técnico

- **Python 3.10+**
- **OpenCV** (`opencv-python`) — abrir el stream (URL HTTP, RTSP o fichero), mostrar la ventana en vivo, escribir el vídeo de salida.
- **Ultralytics YOLOv8** (`ultralytics`) — detección + tracking. Modelo por defecto `yolov8n.pt` (rápido); configurable a `yolov8s.pt` para más precisión.
- **PyTorch con CUDA** — instalado como dependencia de ultralytics; aprovecha la GPU NVIDIA automáticamente.

## Flujo del programa

1. Parsear argumentos: `--source`, `--duration` (default 30s), `--model` (default `yolov8n.pt`), `--no-save`, `--no-display`.
2. Cargar modelo YOLOv8.
3. Abrir el stream con `cv2.VideoCapture(source)`. Validar que se abre correctamente; si no, error claro.
4. Preparar `VideoWriter` con el mismo tamaño y FPS del stream (a menos que `--no-save`). Si los FPS del stream no son fiables (típico en RTSP), fijar 25 FPS.
5. Bucle hasta que pasen `duration` segundos *del tiempo procesado* (no wall clock — más estable si el stream se ralentiza):
   - Leer frame. Si `ret` es False, romper.
   - `results = model.track(frame, persist=True, classes=[2,3,5,7], verbose=False)` — clases COCO: car, motorcycle, bus, truck.
   - Para cada detección con `track_id`: añadir el ID al `set` correspondiente a su clase.
   - Anotar el frame con cajas + ID + clase (`results[0].plot()`).
   - Si `--no-display` no está, `cv2.imshow(frame_anotado)`. Permitir salida temprana con `q`.
   - Si guarda, `writer.write(frame_anotado)`.
6. Cerrar recursos (capture, writer, ventana).
7. Calcular total = suma de tamaños de los sets por clase.
8. Imprimir resumen en consola:
   ```
   Han pasado 47 vehículos en 30.0 s
     coches:     38
     motos:       4
     camiones:    3
     autobuses:   2
   Vídeo guardado en: output/2026-05-13_18-22-11.mp4
   ```
9. Escribir `output/<timestamp>.json` con: source, duration_real, total, desglose, lista de track IDs por clase, modelo usado.

## Estrategia de conteo

**Vehículos únicos por track ID.** Mantener un `set()` de `track_id`s vistos durante la ventana, por clase. El total es la suma de los tamaños de los sets.

- **Por qué:** ByteTrack (integrado en YOLOv8) asigna un ID estable a cada objeto mientras está en el frame. Contar IDs únicos es la forma más simple de "cuántos coches diferentes han aparecido".
- **Limitación conocida:** si un vehículo se ocluye totalmente y reaparece, puede recibir un nuevo ID (sobre-conteo). Para tráfico fluido visto desde una cámara fija es aceptable. Mitigación futura: línea de conteo virtual (v2).

## Estructura de archivos

```
contador-coches/
├── contar.py              # script principal
├── requirements.txt       # opencv-python, ultralytics
├── README.md              # cómo instalar y ejecutar
├── docs/
│   └── superpowers/specs/2026-05-13-contador-coches-design.md
├── output/                # .gitignore — vídeos y json generados
└── .gitignore
```

`contar.py` se mantiene como un único archivo para v1 (script lineal, ~150 líneas). Si crece, partir en `detector.py` + `cli.py`.

## Configuración / argumentos CLI

| Flag           | Default        | Descripción                                       |
|----------------|----------------|---------------------------------------------------|
| `--source`     | (obligatorio)  | URL RTSP/HTTP o ruta a archivo de vídeo           |
| `--duration`   | `30`           | Segundos a procesar                               |
| `--model`      | `yolov8n.pt`   | Pesos del modelo YOLO                             |
| `--no-save`    | `False`        | No guardar vídeo ni JSON                          |
| `--no-display` | `False`        | No abrir ventana (útil para ejecución headless)   |
| `--output-dir` | `output`       | Carpeta de salida                                 |

## Manejo de errores

- **Stream no se abre:** mensaje claro indicando la URL/path y posibles causas (URL incorrecta, sin internet, credenciales RTSP). Exit code 1.
- **Modelo no descargable:** ultralytics descarga el modelo la primera vez. Si falla (sin red), error claro.
- **Frame nulo durante ejecución:** romper bucle, no falla; reportar duración real procesada.
- **GPU no disponible:** caer a CPU automáticamente (comportamiento por defecto de ultralytics). Avisar por consola.

## Plan de pruebas

1. **Prueba inicial con el .mp4 público de TfL** (`https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.08953.mp4?i=jvqch`). Verificar visualmente con el vídeo guardado.
2. **Prueba con archivo local** descargando un clip de tráfico conocido.
3. **Prueba con `--duration 5`** para iterar rápido.
4. **Prueba con `--no-display`** para confirmar que funciona sin GUI.

## Riesgos y mitigaciones

- **Sobre-conteo por re-IDs tras oclusión:** aceptado en v1; documentado.
- **Vídeo .mp4 público dura menos de 30s:** si el stream se acaba antes, terminar bonito y reportar duración real.
- **Primer arranque lento:** descarga del modelo (~6 MB para yolov8n). Documentar en README.
- **Codec mp4v puede no reproducirse en todos los players:** funciona en VLC/Windows Media Player modernos; aceptable.
