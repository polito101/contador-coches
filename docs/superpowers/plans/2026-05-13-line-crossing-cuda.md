# Línea de conteo + dirección + GPU — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the "unique track IDs in window" counting strategy with a virtual line-crossing counter that also reports direction (up/down or left/right), and reinstall PyTorch with CUDA so the GPU is actually used.

**Architecture:** A new `LineCrossingCounter` class observes each track's centroid per frame, remembers the previous position, and only counts a track the first time it crosses the configured line. Direction is recorded based on the side it came from. When `--line-y`/`--line-x` is not set, the existing `VehicleCounter` is used (backwards compatible). A small `--preview-frame` helper lets the user grab the first frame as PNG to choose pixel coordinates.

**Tech Stack:** Python, OpenCV, Ultralytics YOLOv8, PyTorch with CUDA 12.1.

**Context for the engineer:** The project lives in `C:\Users\pobom\contador-coches\`. The venv is at `.venv\`. Use `./.venv/Scripts/python.exe` for all Python commands (the bare `python` is a broken Windows Store stub). Current `contar.py` has `VehicleCounter` (with `min_frames` filter) and a YouTube source path that records via yt-dlp.

---

## File structure

```
contador-coches/
├── contar.py              # MODIFIED — add LineCrossingCounter, CLI flags, preview mode
├── requirements.txt       # MODIFIED — switch torch to CUDA wheel index
├── tests/
│   └── test_counting.py   # MODIFIED — new tests for LineCrossingCounter and new CLI flags
├── README.md              # MODIFIED — document --line-y/--line-x, --preview-frame, CUDA install
└── docs/superpowers/plans/2026-05-13-line-crossing-cuda.md   # this plan
```

Everything stays in `contar.py` for now. If it grows past ~300 lines, future work can split out `counters.py`, but YAGNI for now.

---

## Task 0: Reinstall PyTorch with CUDA 12.1

PyTorch was installed as CPU-only by default. With the user's NVIDIA GPU available, switching to the CUDA build gives ~10x speedup on YOLO inference.

**Files:**
- Modify: `C:\Users\pobom\contador-coches\requirements.txt`

- [ ] **Step 0.1: Uninstall existing CPU torch**

Run:
```bash
cd /c/Users/pobom/contador-coches && ./.venv/Scripts/pip.exe uninstall -y torch torchvision
```
Expected: both uninstalled. No errors.

- [ ] **Step 0.2: Install CUDA 12.1 torch wheels**

Run:
```bash
./.venv/Scripts/pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Expected: ~2 GB download, succeeds. Allow timeout 600000ms.

- [ ] **Step 0.3: Verify CUDA is now detected**

Run:
```bash
./.venv/Scripts/python.exe -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
Expected: `cuda available: True` and a GPU name (e.g. RTX something).

If `False`, abort and report: most likely the NVIDIA driver is too old. Do not proceed with the rest of the plan — the user can stay on CPU.

- [ ] **Step 0.4: Confirm existing tests still pass**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: 12 passed.

- [ ] **Step 0.5: Update requirements.txt with a comment about the CUDA index**

The `pip install -r requirements.txt` default flow gets the CPU build. Add a comment header so future users know how to get the CUDA build.

Read `requirements.txt`. At the top, before the existing pins, insert:

```
# To install with CUDA 12.1 GPU support, run AFTER `pip install -r requirements.txt`:
#   pip uninstall -y torch torchvision
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Without this, torch will be CPU-only (works but ~10x slower for YOLO inference).
```

- [ ] **Step 0.6: Commit**

```bash
git add requirements.txt
git -c user.name="Pol Bonet" commit -m "chore: document CUDA torch install procedure"
```

---

## Task 1: Tests for LineCrossingCounter (TDD — failing first)

We write all tests for the new counter before implementing it.

**Files:**
- Modify: `tests/test_counting.py`

- [ ] **Step 1.1: Append failing tests to `tests/test_counting.py`**

Append:

```python


from contar import LineCrossingCounter


def test_line_y_counts_top_to_bottom_crossing():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=80)   # car above line
    c.observe(track_id=1, class_id=2, cx=55, cy=120)  # now below -> crossed down
    assert c.total() == 1
    bd = c.breakdown()
    assert bd["car"]["down"] == 1
    assert bd["car"]["up"] == 0


def test_line_y_counts_bottom_to_top_crossing():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=150)
    c.observe(track_id=1, class_id=2, cx=55, cy=50)
    assert c.total() == 1
    assert c.breakdown()["car"]["up"] == 1
    assert c.breakdown()["car"]["down"] == 0


def test_line_y_no_count_when_not_crossing():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=20)
    c.observe(track_id=1, class_id=2, cx=55, cy=30)
    assert c.total() == 0


def test_line_y_no_double_count_same_track():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=80)
    c.observe(track_id=1, class_id=2, cx=55, cy=120)  # cross
    c.observe(track_id=1, class_id=2, cx=60, cy=80)   # cross back
    c.observe(track_id=1, class_id=2, cx=65, cy=120)  # cross again
    # Still counted once — first crossing wins.
    assert c.total() == 1


def test_line_x_counts_left_to_right_crossing():
    c = LineCrossingCounter(line_x=200)
    c.observe(track_id=1, class_id=5, cx=180, cy=50)  # bus left of line
    c.observe(track_id=1, class_id=5, cx=220, cy=55)  # now right -> crossed "right"
    assert c.total() == 1
    assert c.breakdown()["bus"]["right"] == 1
    assert c.breakdown()["bus"]["left"] == 0


def test_line_x_counts_right_to_left_crossing():
    c = LineCrossingCounter(line_x=200)
    c.observe(track_id=1, class_id=5, cx=220, cy=50)
    c.observe(track_id=1, class_id=5, cx=180, cy=55)
    assert c.breakdown()["bus"]["left"] == 1


def test_line_counter_ignores_non_vehicle_class():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=0, cx=50, cy=80)   # person
    c.observe(track_id=1, class_id=0, cx=55, cy=120)
    assert c.total() == 0


def test_line_counter_requires_exactly_one_axis():
    import pytest as _pt
    with _pt.raises(ValueError):
        LineCrossingCounter()
    with _pt.raises(ValueError):
        LineCrossingCounter(line_y=100, line_x=200)


def test_line_counter_first_observation_does_not_count():
    # A track that appears already past the line shouldn't be counted retroactively.
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=120)
    assert c.total() == 0


def test_line_counter_summary_shape():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=80)
    c.observe(track_id=1, class_id=2, cx=55, cy=120)
    summary = c.summary(source="x.mp4", duration_real=2.5, model="yolov8n.pt")
    assert summary["total"] == 1
    assert summary["source"] == "x.mp4"
    assert summary["duration_real"] == 2.5
    assert summary["model"] == "yolov8n.pt"
    assert summary["line_y"] == 100
    assert summary["line_x"] is None
    assert summary["breakdown"]["car"]["down"] == 1
    assert summary["breakdown"]["car"]["up"] == 0
    assert summary["track_ids"]["car"]["down"] == [1]
```

- [ ] **Step 1.2: Run tests to confirm they FAIL**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_counting.py -v
```
Expected: `ImportError: cannot import name 'LineCrossingCounter' from 'contar'` (or similar). Existing 12 tests still pass.

- [ ] **Step 1.3: No commit yet** — these tests are intentionally failing. We commit them together with the implementation in Task 2.

---

## Task 2: Implement LineCrossingCounter

**Files:**
- Modify: `contar.py`

- [ ] **Step 2.1: Add LineCrossingCounter class to contar.py**

In `contar.py`, immediately after the existing `VehicleCounter` class (before `import argparse`), append:

```python


class LineCrossingCounter:
    """Counts unique vehicles that cross a virtual line, with direction.

    Exactly one of `line_y` or `line_x` must be set.
    - `line_y=N` → horizontal line at row N. Reports "down" (top→bottom) and "up" (bottom→top).
    - `line_x=N` → vertical line at column N. Reports "right" (left→right) and "left" (right→left).

    A track is counted on its first crossing only; subsequent crossings by the
    same track_id are ignored.
    """

    def __init__(self, line_y: int | None = None, line_x: int | None = None) -> None:
        if (line_y is None) == (line_x is None):
            raise ValueError("Pass exactly one of line_y or line_x.")
        self.line_y = line_y
        self.line_x = line_x
        self._last_pos: dict[int, tuple[int, int]] = {}
        # Per class: {direction_name: set(track_ids that crossed in that direction)}
        if line_y is not None:
            dirs = ("down", "up")
        else:
            dirs = ("right", "left")
        self._dirs = dirs
        self._crossed: dict[str, dict[str, set[int]]] = {
            name: {d: set() for d in dirs}
            for name in VEHICLE_CLASSES.values()
        }
        # Track IDs that have already been counted (in any direction) — prevents
        # double-counting if a vehicle crosses back.
        self._already_counted: set[int] = set()

    def observe(self, track_id: int, class_id: int, cx: int, cy: int) -> None:
        name = VEHICLE_CLASSES.get(class_id)
        if name is None:
            return
        prev = self._last_pos.get(track_id)
        self._last_pos[track_id] = (cx, cy)
        if prev is None or track_id in self._already_counted:
            return
        px, py = prev
        if self.line_y is not None:
            if py < self.line_y <= cy:
                self._crossed[name]["down"].add(track_id)
                self._already_counted.add(track_id)
            elif py > self.line_y >= cy:
                self._crossed[name]["up"].add(track_id)
                self._already_counted.add(track_id)
        else:
            if px < self.line_x <= cx:
                self._crossed[name]["right"].add(track_id)
                self._already_counted.add(track_id)
            elif px > self.line_x >= cx:
                self._crossed[name]["left"].add(track_id)
                self._already_counted.add(track_id)

    def total(self) -> int:
        return sum(
            len(self._crossed[n][d])
            for n in self._crossed
            for d in self._dirs
        )

    def breakdown(self) -> dict[str, dict[str, int]]:
        return {
            n: {d: len(self._crossed[n][d]) for d in self._dirs}
            for n in self._crossed
        }

    def summary(self, source: str, duration_real: float, model: str) -> dict:
        return {
            "source": source,
            "duration_real": duration_real,
            "model": model,
            "line_y": self.line_y,
            "line_x": self.line_x,
            "total": self.total(),
            "breakdown": self.breakdown(),
            "track_ids": {
                n: {d: sorted(self._crossed[n][d]) for d in self._dirs}
                for n in self._crossed
            },
        }
```

- [ ] **Step 2.2: Run tests to confirm new ones PASS and old ones still PASS**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_counting.py -v
```
Expected: 22 passed (12 original + 10 new).

- [ ] **Step 2.3: Commit**

```bash
git add contar.py tests/test_counting.py
git -c user.name="Pol Bonet" commit -m "feat: add LineCrossingCounter with direction tracking"
```

---

## Task 3: CLI flags `--line-y`, `--line-x`, `--preview-frame`

**Files:**
- Modify: `contar.py` (parse_args)
- Modify: `tests/test_counting.py` (new arg tests)

- [ ] **Step 3.1: Add the new flags to `parse_args` in `contar.py`**

In `contar.py`, find the existing `parse_args` function. After the line:
```python
    p.add_argument("--min-frames", type=int, default=5,
                   help="Minimum frames a track must appear in to be counted "
                        "(default: 5). Higher = filters out ephemeral re-IDs.")
```

Insert before `return p.parse_args(argv)`:

```python
    p.add_argument("--line-y", type=int, default=None,
                   help="Horizontal counting line at this pixel row. When set, "
                        "use line-crossing mode (overrides --min-frames).")
    p.add_argument("--line-x", type=int, default=None,
                   help="Vertical counting line at this pixel column. Mutually "
                        "exclusive with --line-y.")
    p.add_argument("--preview-frame", action="store_true",
                   help="Save the first frame of the source as PNG to "
                        "<output-dir>/preview_<timestamp>.png and exit. Use to "
                        "pick pixel coordinates for --line-y/--line-x.")
```

- [ ] **Step 3.2: Add tests for the new flags**

Append to `tests/test_counting.py`:

```python


def test_parse_args_line_y_default_none():
    args = parse_args(["--source", "foo.mp4"])
    assert args.line_y is None
    assert args.line_x is None
    assert args.preview_frame is False


def test_parse_args_line_y_set():
    args = parse_args(["--source", "foo.mp4", "--line-y", "400"])
    assert args.line_y == 400
    assert args.line_x is None


def test_parse_args_line_x_set():
    args = parse_args(["--source", "foo.mp4", "--line-x", "600"])
    assert args.line_y is None
    assert args.line_x == 600


def test_parse_args_preview_frame():
    args = parse_args(["--source", "foo.mp4", "--preview-frame"])
    assert args.preview_frame is True
```

- [ ] **Step 3.3: Run tests**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_counting.py -v
```
Expected: 26 passed (22 + 4).

- [ ] **Step 3.4: Confirm `--help` shows the new flags**

Run:
```bash
./.venv/Scripts/python.exe contar.py --help
```
Expected: `--line-y`, `--line-x`, `--preview-frame` all listed in the help output.

- [ ] **Step 3.5: Commit**

```bash
git add contar.py tests/test_counting.py
git -c user.name="Pol Bonet" commit -m "feat: add CLI flags --line-y, --line-x, --preview-frame"
```

---

## Task 4: Implement `--preview-frame` mode

This is short-circuit logic in `main()`: open the source, read one frame, write a PNG, exit. No model loading needed.

**Files:**
- Modify: `contar.py` (main function)

- [ ] **Step 4.1: Add preview-frame short-circuit at the top of `main()`**

In `contar.py`, find:
```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Loading model {args.model}...")
    model = YOLO(args.model)
```

Replace those four lines with:

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.line_y is not None and args.line_x is not None:
        print("ERROR: pass either --line-y or --line-x, not both.", file=sys.stderr)
        return 2

    if args.preview_frame:
        return _preview_frame(args)

    print(f"Loading model {args.model}...")
    model = YOLO(args.model)
```

- [ ] **Step 4.2: Add the `_preview_frame` helper above `main()` in `contar.py`**

Place this function immediately before `def main(...)`:

```python
def _preview_frame(args: argparse.Namespace) -> int:
    """Save the first frame of the source as PNG, then exit.

    Skips model loading and tracking — useful for picking line-y/line-x pixel
    coordinates visually.
    """
    cap = open_capture(args.source, args.duration, args.output_dir)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("ERROR: could not read a frame from the source.", file=sys.stderr)
        return 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = output_dir / f"preview_{timestamp}.png"
    cv2.imwrite(str(out_path), frame)
    print(f"First frame saved to: {out_path}")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]} (width x height)")
    print("Open the file, pick a Y or X pixel for your counting line, then run:")
    print(f"  python contar.py --source <same> --line-y <row>")
    return 0
```

- [ ] **Step 4.3: Smoke test the preview mode**

Run:
```bash
./.venv/Scripts/python.exe contar.py --source "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.08953.mp4?i=jvqch" --preview-frame
```
Expected: prints "First frame saved to: output/preview_*.png" and exits 0. The PNG should exist and be openable.

- [ ] **Step 4.4: Run all tests**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: 26 passed.

- [ ] **Step 4.5: Commit**

```bash
git add contar.py
git -c user.name="Pol Bonet" commit -m "feat: implement --preview-frame to extract first frame"
```

---

## Task 5: Wire LineCrossingCounter into the main loop

When `--line-y` or `--line-x` is set, use `LineCrossingCounter` instead of `VehicleCounter`. Also draw the line on the annotated frames.

**Files:**
- Modify: `contar.py` (main function)

- [ ] **Step 5.1: Replace counter selection and observation logic**

In `contar.py`, find the existing block in `main()`:

```python
    counter = VehicleCounter(min_frames=args.min_frames)
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
                conf=args.conf,
                verbose=False,
            )
            r = results[0]

            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().tolist()
                clss = r.boxes.cls.int().cpu().tolist()
                for tid, cid in zip(ids, clss):
                    counter.add(track_id=tid, class_id=cid)

            annotated = r.plot()
```

Replace it with:

```python
    use_line = args.line_y is not None or args.line_x is not None
    counter: VehicleCounter | LineCrossingCounter
    if use_line:
        counter = LineCrossingCounter(line_y=args.line_y, line_x=args.line_x)
    else:
        counter = VehicleCounter(min_frames=args.min_frames)
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
                conf=args.conf,
                verbose=False,
            )
            r = results[0]

            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().tolist()
                clss = r.boxes.cls.int().cpu().tolist()
                if use_line:
                    # xywh: [cx, cy, w, h] (centroid format)
                    xywh = r.boxes.xywh.cpu().numpy()
                    for tid, cid, (cx, cy, _, _) in zip(ids, clss, xywh):
                        counter.observe(track_id=tid, class_id=cid,
                                        cx=int(cx), cy=int(cy))
                else:
                    for tid, cid in zip(ids, clss):
                        counter.add(track_id=tid, class_id=cid)

            annotated = r.plot()
            if use_line:
                if args.line_y is not None:
                    cv2.line(annotated, (0, args.line_y),
                             (annotated.shape[1], args.line_y), (0, 255, 255), 2)
                else:
                    cv2.line(annotated, (args.line_x, 0),
                             (args.line_x, annotated.shape[0]), (0, 255, 255), 2)
```

- [ ] **Step 5.2: Update the final summary print to handle both counter types**

In `contar.py`, find the existing block (after the try/finally that releases resources):

```python
    breakdown = counter.breakdown()
    print()
    print(f"Han pasado {counter.total()} vehículos en {duration_real:.1f} s "
          f"({frame_count} frames procesados)")
    print(f"  coches:    {breakdown['car']:>4}")
    print(f"  motos:     {breakdown['motorcycle']:>4}")
    print(f"  camiones:  {breakdown['truck']:>4}")
    print(f"  autobuses: {breakdown['bus']:>4}")
```

Replace with:

```python
    breakdown = counter.breakdown()
    print()
    print(f"Han pasado {counter.total()} vehículos en {duration_real:.1f} s "
          f"({frame_count} frames procesados)")
    if use_line:
        # breakdown is dict[class, dict[direction, count]]
        dir_names = ("down", "up") if args.line_y is not None else ("right", "left")
        header = f"  {'clase':<10} {dir_names[0]:>6} {dir_names[1]:>6}"
        print(header)
        for label, key in [("coches", "car"), ("motos", "motorcycle"),
                            ("camiones", "truck"), ("autobuses", "bus")]:
            d = breakdown[key]
            print(f"  {label:<10} {d[dir_names[0]]:>6} {d[dir_names[1]]:>6}")
    else:
        print(f"  coches:    {breakdown['car']:>4}")
        print(f"  motos:     {breakdown['motorcycle']:>4}")
        print(f"  camiones:  {breakdown['truck']:>4}")
        print(f"  autobuses: {breakdown['bus']:>4}")
```

- [ ] **Step 5.3: Run all tests**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: 26 passed.

- [ ] **Step 5.4: Smoke test in headless mode without line (backwards compat)**

Run:
```bash
./.venv/Scripts/python.exe contar.py --source "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.08953.mp4?i=jvqch" --duration 5 --no-display
```
Expected: same flat summary as before ("coches:   N", etc.). No errors.

- [ ] **Step 5.5: Smoke test with a horizontal line**

First, see what a good Y is for the TfL clip by saving the preview:
```bash
./.venv/Scripts/python.exe contar.py --source "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.08953.mp4?i=jvqch" --preview-frame
```
Open the saved PNG to read its dimensions (printed in the output too). Pick a Y value in the middle of the road area. For TfL clips at 288x352 (these jamcams are small), Y around 180 typically works.

Then run with the line:
```bash
./.venv/Scripts/python.exe contar.py --source "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.08953.mp4?i=jvqch" --duration 5 --no-display --line-y 180
```
Expected: summary shows columns `down` and `up` with per-class counts. A yellow line was drawn in the saved video.

If the count is 0/0, the chosen Y may not intersect actual vehicle paths — pick a different Y from the preview PNG and retry. This is a sanity check, not a strict pass/fail.

- [ ] **Step 5.6: Commit**

```bash
git add contar.py
git -c user.name="Pol Bonet" commit -m "feat: wire LineCrossingCounter into main loop with line overlay"
```

---

## Task 6: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 6.1: Update README with the new features**

Read `README.md`. Replace the entire "Flags" table with:

```markdown
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
```

Then, find the "Ajustando precisión" section and replace it with:

```markdown
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
```

Finally, find the existing "Instalación" section and append a new subsection:

```markdown
### GPU NVIDIA (opcional, ~10x más rápido)

`pip install -r requirements.txt` instala torch en CPU. Para usar tu GPU:

```powershell
.\.venv\Scripts\pip.exe uninstall -y torch torchvision
.\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verifica con:
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```
```

- [ ] **Step 6.2: Commit**

```bash
git add README.md
git -c user.name="Pol Bonet" commit -m "docs: document line-counting mode and GPU install"
```

---

## Task 7: Final verification

**Files:** none

- [ ] **Step 7.1: Run the full test suite**

Run:
```bash
./.venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: 26 passed.

- [ ] **Step 7.2: End-to-end test — line mode with YouTube**

Run preview to pick a Y:
```bash
./.venv/Scripts/python.exe contar.py --source "https://www.youtube.com/watch?v=M3EYAY2MftI" --preview-frame
```
Open the PNG, pick a Y value, then:
```bash
./.venv/Scripts/python.exe contar.py --source "https://www.youtube.com/watch?v=M3EYAY2MftI" --duration 30 --no-display --line-y <Y>
```

Expected: prints per-direction counts and writes `output/<timestamp>.mp4` (with line overlay) + `.json` (with `line_y`, breakdown by direction).

- [ ] **Step 7.3: End-to-end test — backwards compatibility (no line)**

Run:
```bash
./.venv/Scripts/python.exe contar.py --source "https://www.youtube.com/watch?v=M3EYAY2MftI" --duration 10 --no-display
```
Expected: classic flat output ("coches: N", "motos: N", ...). No regression.

- [ ] **Step 7.4: Confirm GPU is being used (informational)**

Run:
```bash
./.venv/Scripts/python.exe -c "from ultralytics import YOLO; m = YOLO('yolov8n.pt'); print('device after load:', m.device)"
```
Expected: prints `device after load: cuda:0` (or similar). If `cpu`, GPU isn't being picked up despite torch being CUDA-enabled — investigate.

- [ ] **Step 7.5: Confirm git status is clean**

Run:
```bash
git status
```
Expected: nothing to commit, working tree clean (apart from gitignored `output/`).

---

## Done criteria

- 26 tests pass.
- `--preview-frame` saves a PNG of the first frame and exits.
- `--line-y N` counts only vehicles that cross row N; output splits "down"/"up" per class.
- `--line-x N` works similarly with "right"/"left".
- The saved video shows the yellow counting line drawn at the configured position.
- The JSON summary includes `line_y`/`line_x` and per-direction track ID lists.
- Without `--line-y`/`--line-x`, behavior is identical to before (backwards compatible).
- `torch.cuda.is_available()` returns True; YOLO reports `device: cuda:0`.
