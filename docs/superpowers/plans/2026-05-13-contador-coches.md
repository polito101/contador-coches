# Contador de Vehículos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that reads 30 seconds from a video source (RTSP/HTTP/file), counts unique vehicles using YOLOv8 + tracking, shows a live window, and saves an annotated video + JSON summary for verification.

**Architecture:** Single-file script (`contar.py`) using OpenCV for I/O and Ultralytics YOLOv8 with built-in ByteTrack for detection+tracking. Unique track IDs per vehicle class are accumulated in `set()`s during the time window. At the end the total and per-class breakdown are printed and a JSON summary is written. The annotated frame (with bounding boxes and IDs drawn) is shown live and written to disk.

**Tech Stack:** Python 3.10+, OpenCV (`opencv-python`), Ultralytics (`ultralytics`), PyTorch with CUDA (transitive dep).

**Spec:** `docs/superpowers/specs/2026-05-13-contador-coches-design.md`

---

## File structure

```
contador-coches/
├── contar.py              # Main CLI script (created in Task 4-7)
├── requirements.txt       # Pinned dependencies (Task 1)
├── README.md              # Install + run instructions (Task 9)
├── .gitignore             # Ignore output/, venv/, __pycache__/ (Task 1)
├── tests/
│   └── test_counting.py   # Unit tests for the counting helpers (Task 3)
├── output/                # Generated videos + JSON (gitignored)
└── docs/
    └── superpowers/
        ├── specs/2026-05-13-contador-coches-design.md
        └── plans/2026-05-13-contador-coches.md
```

**Design note:** The plan keeps everything in `contar.py` as the spec specifies, but the *pure logic* (vehicle ID accumulation, JSON building) is exposed as small functions so we can unit-test them without OpenCV or a real stream. Side-effectful code (capture, display, write) lives in a `main()` function called only when `__name__ == "__main__"`.

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Create requirements.txt**

```
opencv-python==4.10.0.84
ultralytics==8.3.40
yt-dlp==2024.11.18
```

- [ ] **Step 2: Create .gitignore**

```
# Python
__pycache__/
*.pyc
.venv/
venv/

# Project outputs
output/

# Models cached by ultralytics
*.pt

# OS
Thumbs.db
.DS_Store
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: scaffold project (requirements, gitignore)"
```

---

## Task 2: Virtual environment + install

**Files:** none (environment setup)

- [ ] **Step 1: Create and activate virtualenv (PowerShell)**

Run:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Expected: prompt now shows `(.venv)`.

If `Activate.ps1` is blocked by execution policy, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

- [ ] **Step 2: Upgrade pip and install dependencies**

Run:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Expected: install completes without errors. `ultralytics` pulls in `torch`. On Windows + NVIDIA, the default `torch` wheel uses CUDA if drivers are present.

- [ ] **Step 3: Verify install**

Run:
```powershell
python -c "import cv2, ultralytics; print(cv2.__version__, ultralytics.__version__)"
```

Expected: prints two version strings, no traceback.

- [ ] **Step 4: Verify CUDA is detected (informational)**

Run:
```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Expected: `cuda: True <GPU name>`. If `False`, the script will still work on CPU — just slower. Do not block on this.

- [ ] **Step 5: Add pytest for tests**

Run:
```powershell
pip install pytest==8.3.3
```

Then add to `requirements.txt` a new line:
```
pytest==8.3.3
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "chore: pin pytest in requirements"
```

---

## Task 3: Pure counting logic + tests (TDD)

We extract the only logic that's not just plumbing: maintaining the per-class set of track IDs and producing the summary dict. This lets us TDD without needing a camera.

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_counting.py`
- Create: `contar.py` (only the helpers in this task)

- [ ] **Step 1: Create tests/__init__.py (empty file)**

```python
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_counting.py`:

```python
from contar import VehicleCounter, VEHICLE_CLASSES


def test_counter_starts_empty():
    c = VehicleCounter()
    assert c.total() == 0
    assert c.breakdown() == {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}


def test_counter_adds_unique_ids_per_class():
    c = VehicleCounter()
    c.add(track_id=1, class_id=2)   # car
    c.add(track_id=2, class_id=2)   # car
    c.add(track_id=3, class_id=7)   # truck
    assert c.total() == 3
    assert c.breakdown() == {"car": 2, "motorcycle": 0, "bus": 0, "truck": 1}


def test_counter_dedupes_same_id_same_class():
    c = VehicleCounter()
    for _ in range(50):
        c.add(track_id=5, class_id=2)
    assert c.total() == 1
    assert c.breakdown()["car"] == 1


def test_counter_ignores_non_vehicle_class():
    c = VehicleCounter()
    c.add(track_id=1, class_id=0)   # person (COCO id 0) — not in vehicle set
    assert c.total() == 0


def test_counter_same_id_different_class_counts_twice():
    # Edge: tracker reused an id across classes. Treat them as different vehicles.
    c = VehicleCounter()
    c.add(track_id=1, class_id=2)
    c.add(track_id=1, class_id=7)
    assert c.total() == 2


def test_summary_dict_shape():
    c = VehicleCounter()
    c.add(track_id=1, class_id=2)
    c.add(track_id=2, class_id=5)
    summary = c.summary(
        source="test.mp4",
        duration_real=12.3,
        model="yolov8n.pt",
    )
    assert summary["total"] == 2
    assert summary["source"] == "test.mp4"
    assert summary["duration_real"] == 12.3
    assert summary["model"] == "yolov8n.pt"
    assert summary["breakdown"]["car"] == 1
    assert summary["breakdown"]["bus"] == 1
    assert set(summary["track_ids"]["car"]) == {1}
    assert set(summary["track_ids"]["bus"]) == {2}


def test_vehicle_classes_constant():
    # Spec: COCO ids car=2, motorcycle=3, bus=5, truck=7
    assert VEHICLE_CLASSES == {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run:
```powershell
pytest tests/test_counting.py -v
```

Expected: `ImportError` / `ModuleNotFoundError` because `contar.py` doesn't exist yet.

- [ ] **Step 4: Write minimal implementation**

Create `contar.py`:

```python
"""Contador de vehículos — count unique vehicles passing through a video stream."""

from __future__ import annotations

VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleCounter:
    """Tracks unique (track_id, class_id) pairs per vehicle class."""

    def __init__(self) -> None:
        self._ids_by_class: dict[str, set[int]] = {
            name: set() for name in VEHICLE_CLASSES.values()
        }

    def add(self, track_id: int, class_id: int) -> None:
        name = VEHICLE_CLASSES.get(class_id)
        if name is None:
            return
        self._ids_by_class[name].add(track_id)

    def total(self) -> int:
        return sum(len(s) for s in self._ids_by_class.values())

    def breakdown(self) -> dict[str, int]:
        return {name: len(ids) for name, ids in self._ids_by_class.items()}

    def summary(self, source: str, duration_real: float, model: str) -> dict:
        return {
            "source": source,
            "duration_real": duration_real,
            "model": model,
            "total": self.total(),
            "breakdown": self.breakdown(),
            "track_ids": {
                name: sorted(ids) for name, ids in self._ids_by_class.items()
            },
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```powershell
pytest tests/test_counting.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add contar.py tests/
git commit -m "feat: add VehicleCounter with unique-track-id accumulation"
```

---

## Task 4: CLI argument parsing

**Files:**
- Modify: `contar.py` (add `parse_args` + main entry point)

- [ ] **Step 1: Add argument parser to contar.py**

Append to `contar.py`:

```python
import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Count unique vehicles in a video stream.")
    p.add_argument("--source", required=True,
                   help="RTSP/HTTP URL or path to a video file.")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Seconds of video to process (default: 30).")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Ultralytics model weights (default: yolov8n.pt).")
    p.add_argument("--output-dir", default="output",
                   help="Where to write annotated video + JSON (default: output).")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing video and JSON.")
    p.add_argument("--no-display", action="store_true",
                   help="Skip opening the OpenCV window (headless mode).")
    return p.parse_args(argv)
```

- [ ] **Step 2: Add a quick test for parse_args**

Append to `tests/test_counting.py`:

```python
from contar import parse_args


def test_parse_args_defaults():
    args = parse_args(["--source", "foo.mp4"])
    assert args.source == "foo.mp4"
    assert args.duration == 30.0
    assert args.model == "yolov8n.pt"
    assert args.output_dir == "output"
    assert args.no_save is False
    assert args.no_display is False


def test_parse_args_overrides():
    args = parse_args([
        "--source", "rtsp://x", "--duration", "5",
        "--model", "yolov8s.pt", "--no-save", "--no-display",
    ])
    assert args.source == "rtsp://x"
    assert args.duration == 5.0
    assert args.model == "yolov8s.pt"
    assert args.no_save is True
    assert args.no_display is True
```

- [ ] **Step 3: Run tests**

Run:
```powershell
pytest tests/test_counting.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 4: Commit**

```bash
git add contar.py tests/test_counting.py
git commit -m "feat: add CLI argument parsing"
```

---

## Task 5: Main loop — capture, detect, track, count

This is the biggest task. It wires OpenCV + Ultralytics together. There is no easy unit test for the live loop, so we rely on visual verification with the TfL sample in Task 8.

**Files:**
- Modify: `contar.py`

- [ ] **Step 1: Add main() with capture + tracking loop**

Append to `contar.py`:

```python
import json
import sys
import time
from datetime import datetime

import cv2
from ultralytics import YOLO


def resolve_source(source: str) -> str | int:
    """Translate a user-supplied source into something OpenCV can open.

    - Integer string ("0", "1") -> int (webcam index).
    - YouTube URL -> direct HLS/MP4 URL via yt-dlp.
    - Anything else -> returned as-is (file path, RTSP, HTTP .mp4).
    """
    # Webcam index
    try:
        return int(source)
    except ValueError:
        pass

    if "youtube.com" in source or "youtu.be" in source:
        from yt_dlp import YoutubeDL
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            # Prefer HLS (live) or MP4 (VOD). Cap height for performance.
            "format": "best[height<=720][protocol^=m3u8]/best[height<=720]",
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=False)
            url = info.get("url")
            if not url:
                print(f"ERROR: could not extract stream URL from YouTube source.",
                      file=sys.stderr)
                sys.exit(1)
            return url

    return source


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a video source. Raises SystemExit with a clear message on failure."""
    src = resolve_source(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: could not open source '{source}'. "
              f"Check the URL/path, network, or credentials.", file=sys.stderr)
        sys.exit(1)
    return cap


def make_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Loading model {args.model}...")
    model = YOLO(args.model)

    cap = open_capture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_stream = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_stream if 1.0 < fps_stream < 120.0 else 25.0

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer: cv2.VideoWriter | None = None
    video_path: Path | None = None
    json_path: Path | None = None
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"{timestamp}.mp4"
        json_path = output_dir / f"{timestamp}.json"
        writer = make_writer(video_path, width, height, fps)

    counter = VehicleCounter()
    vehicle_class_ids = list(VEHICLE_CLASSES.keys())

    print(f"Processing {args.duration}s from {args.source} ...")
    start = time.monotonic()
    frame_count = 0
    duration_real = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Stream ended early.")
                break

            results = model.track(
                frame,
                persist=True,
                classes=vehicle_class_ids,
                verbose=False,
            )
            r = results[0]

            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().tolist()
                clss = r.boxes.cls.int().cpu().tolist()
                for tid, cid in zip(ids, clss):
                    counter.add(track_id=tid, class_id=cid)

            annotated = r.plot()

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("contador-coches (q para salir)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Interrupted by user.")
                    break

            frame_count += 1
            duration_real = time.monotonic() - start
            if duration_real >= args.duration:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Report
    breakdown = counter.breakdown()
    print()
    print(f"Han pasado {counter.total()} vehículos en {duration_real:.1f} s "
          f"({frame_count} frames procesados)")
    print(f"  coches:    {breakdown['car']:>4}")
    print(f"  motos:     {breakdown['motorcycle']:>4}")
    print(f"  camiones:  {breakdown['truck']:>4}")
    print(f"  autobuses: {breakdown['bus']:>4}")

    if json_path is not None:
        summary = counter.summary(
            source=args.source,
            duration_real=round(duration_real, 3),
            model=args.model,
        )
        summary["frames_processed"] = frame_count
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Vídeo guardado en:  {video_path}")
        print(f"JSON guardado en:   {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Sanity check — confirm tests still pass (no regressions in helpers)**

Run:
```powershell
pytest tests/test_counting.py -v
```

Expected: all 9 tests still pass.

- [ ] **Step 3: Confirm the script's `--help` works**

Run:
```powershell
python contar.py --help
```

Expected: argparse help text listing every flag. No traceback.

- [ ] **Step 4: Commit**

```bash
git add contar.py
git commit -m "feat: implement capture+track+count main loop"
```

---

## Task 6: Smoke test with a tiny duration

Before testing with the real TfL stream, run for 3 seconds against the same source to catch silly bugs (path wrong, model fails to download, etc.) fast.

**Files:** none (manual verification)

- [ ] **Step 1: Run a 3-second smoke test against TfL sample**

Run:
```powershell
python contar.py --source "https://www.youtube.com/watch?v=M3EYAY2MftI" --duration 3
```

Expected:
- First run downloads `yolov8n.pt` automatically (~6 MB).
- A window pops up showing the TfL clip with bounding boxes.
- After ~3s of processed video, prints `Han pasado N vehículos en ~3.0 s` and a file path under `output/`.
- Exit code 0.

If the window doesn't appear, check that you're not in a remote/headless shell. If the model download fails, check internet.

- [ ] **Step 2: Inspect the saved files**

Run:
```powershell
ls output/
```

Open the newest `.mp4` in VLC or Windows Media Player. You should see the same annotated frames the live window showed.

Open the newest `.json` in any editor; confirm it contains `total`, `breakdown`, `track_ids`, `frames_processed`.

- [ ] **Step 3: No commit needed (no code changed). Move on if smoke test passes.**

---

## Task 7: Full 30-second run

**Files:** none (manual verification)

- [ ] **Step 1: Run the spec-required scenario**

Run:
```powershell
python contar.py --source "https://www.youtube.com/watch?v=M3EYAY2MftI"
```

Expected:
- Script runs for ~30 seconds (or until the clip ends — TfL jamcams are ~10s long, so likely "Stream ended early" appears, which is correct behavior).
- Prints final total + per-class breakdown.
- Writes a video + JSON under `output/`.

- [ ] **Step 2: Open the saved video and visually count vehicles yourself**

Compare your manual count to the script's reported total. They should be in the same ballpark. Wild discrepancies (e.g., 100 vs 5) mean tracking is creating new IDs per frame — investigate before continuing.

- [ ] **Step 3: Try the headless mode**

Run:
```powershell
python contar.py --source "https://www.youtube.com/watch?v=M3EYAY2MftI" --no-display --duration 5
```

Expected: no window opens, but final stats still print and files are saved.

- [ ] **Step 4: Try --no-save**

Run:
```powershell
python contar.py --source "https://www.youtube.com/watch?v=M3EYAY2MftI" --no-save --duration 5
```

Expected: window shows up, prints stats at end, but `output/` has no new files from this run.

- [ ] **Step 5: No commit needed.**

---

## Task 8: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Contador de coches

Cuenta vehículos únicos que pasan por una cámara o vídeo durante una ventana de tiempo configurable.

## Requisitos

- Python 3.10+
- (Opcional) GPU NVIDIA con drivers CUDA — la detección va mucho más rápida.

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

La primera ejecución descarga automáticamente el modelo YOLO (~6 MB).

## Uso

```powershell
# Vídeo de prueba público de las jamcams de Transport for London
python contar.py --source "https://www.youtube.com/watch?v=M3EYAY2MftI"

# Archivo local
python contar.py --source ruta\al\video.mp4

# Cámara IP por RTSP
python contar.py --source "rtsp://usuario:password@192.168.1.10:554/stream1"

# Webcam del portátil (índice 0)
python contar.py --source 0

# Livestream de YouTube (se resuelve automáticamente con yt-dlp)
python contar.py --source "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Flags

| Flag           | Default        | Descripción                                       |
|----------------|----------------|---------------------------------------------------|
| `--source`     | (obligatorio)  | URL RTSP/HTTP o ruta a archivo o índice de webcam |
| `--duration`   | `30`           | Segundos a procesar                               |
| `--model`      | `yolov8n.pt`   | Pesos del modelo YOLO (`yolov8s.pt` más preciso)  |
| `--output-dir` | `output`       | Carpeta de salida                                 |
| `--no-save`    | off            | No guardar vídeo ni JSON                          |
| `--no-display` | off            | No abrir ventana (headless)                       |

Pulsa `q` mientras corre para terminar antes de tiempo.

## Salida

- `output/YYYY-MM-DD_HH-MM-SS.mp4` — vídeo anotado con cajas e IDs.
- `output/YYYY-MM-DD_HH-MM-SS.json` — resumen con total, desglose por clase y lista de IDs únicos.

## Tests

```powershell
pytest -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with install + usage"
```

---

## Task 9: Final verification

**Files:** none

- [ ] **Step 1: Run the test suite from scratch**

Run:
```powershell
pytest -v
```

Expected: 9 passed.

- [ ] **Step 2: Run one final end-to-end check**

Run:
```powershell
python contar.py --source "https://www.youtube.com/watch?v=M3EYAY2MftI" --duration 10
```

Expected:
- Live window shows annotated frames.
- Final summary printed.
- `output/<timestamp>.mp4` and `.json` written.

- [ ] **Step 3: Confirm git status is clean**

Run:
```powershell
git status
```

Expected: `nothing to commit, working tree clean` (apart from `output/` which is gitignored).

---

## Done criteria

- All 9 unit tests pass.
- `python contar.py --source <TfL URL>` runs end-to-end, shows live window, prints a sensible total, and produces an `.mp4` + `.json` in `output/`.
- Manual visual count of the saved video roughly matches the reported total.
- `--no-display` and `--no-save` flags both work.
