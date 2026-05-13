"""Contador de vehículos — count unique vehicles passing through a video stream."""

from __future__ import annotations

import os
# Increase OpenCV/ffmpeg timeouts and reconnect on HLS read failures.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rw_timeout;30000000|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5",
)

VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleCounter:
    """Counts unique vehicles per class, requiring a minimum number of frames
    of presence before a track is considered a real vehicle (filters
    ephemeral false positives like flickering detections on signs).
    """

    def __init__(self, min_frames: int = 1) -> None:
        self.min_frames = min_frames
        self._frames_by_class: dict[str, dict[int, int]] = {
            name: {} for name in VEHICLE_CLASSES.values()
        }

    def add(self, track_id: int, class_id: int) -> None:
        name = VEHICLE_CLASSES.get(class_id)
        if name is None:
            return
        self._frames_by_class[name][track_id] = (
            self._frames_by_class[name].get(track_id, 0) + 1
        )

    def _kept_ids(self, name: str) -> list[int]:
        return [
            tid for tid, n in self._frames_by_class[name].items()
            if n >= self.min_frames
        ]

    def total(self) -> int:
        return sum(len(self._kept_ids(n)) for n in self._frames_by_class)

    def breakdown(self) -> dict[str, int]:
        return {n: len(self._kept_ids(n)) for n in self._frames_by_class}

    def summary(self, source: str, duration_real: float, model: str) -> dict:
        return {
            "source": source,
            "duration_real": duration_real,
            "model": model,
            "min_frames": self.min_frames,
            "total": self.total(),
            "breakdown": self.breakdown(),
            "track_ids": {
                n: sorted(self._kept_ids(n)) for n in self._frames_by_class
            },
        }


class LineCrossingCounter:
    """Counts unique vehicles that cross a line SEGMENT (not an infinite line).

    The segment is defined by two endpoints `p1` and `p2`. A vehicle is
    counted the first time its centroid path between consecutive frames
    intersects this segment. Direction is reported as ``down``/``up`` for
    a mostly-horizontal segment, or ``right``/``left`` for a mostly-vertical
    one — independent of which way the user drew the segment.
    """

    def __init__(self, p1: tuple[int, int], p2: tuple[int, int]) -> None:
        if p1 == p2:
            raise ValueError("Segment endpoints must differ.")
        # Normalize so direction labels don't depend on drag order.
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dx) >= abs(dy):
            if dx < 0:
                p1, p2 = p2, p1
            self._dirs: tuple[str, str] = ("down", "up")
        else:
            if dy < 0:
                p1, p2 = p2, p1
            self._dirs = ("right", "left")
        self.p1 = p1
        self.p2 = p2
        self._last_pos: dict[int, tuple[int, int]] = {}
        self._crossed: dict[str, dict[str, set[int]]] = {
            name: {d: set() for d in self._dirs}
            for name in VEHICLE_CLASSES.values()
        }
        self._already_counted: set[int] = set()

    @staticmethod
    def _orient(a: tuple[int, int], b: tuple[int, int],
                c: tuple[int, int]) -> int:
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0

    @classmethod
    def _segments_intersect(cls, a1, a2, b1, b2) -> bool:
        # Strict (non-collinear) intersection. Collinear-on-segment cases are
        # rare for centroid paths and we don't need to count them specially.
        o1 = cls._orient(a1, a2, b1)
        o2 = cls._orient(a1, a2, b2)
        o3 = cls._orient(b1, b2, a1)
        o4 = cls._orient(b1, b2, a2)
        return o1 != o2 and o3 != o4

    def observe(self, track_id: int, class_id: int, cx: int, cy: int) -> None:
        name = VEHICLE_CLASSES.get(class_id)
        if name is None:
            return
        prev = self._last_pos.get(track_id)
        self._last_pos[track_id] = (cx, cy)
        if prev is None or track_id in self._already_counted:
            return
        curr = (cx, cy)
        if not self._segments_intersect(prev, curr, self.p1, self.p2):
            return
        # Direction: sign of (p1→p2) × (p1→prev) after normalization.
        # Horizontal-ish (normalized dx > 0): negative => prev above => "down".
        # Vertical-ish   (normalized dy > 0): positive => prev left  => "right".
        side = self._orient(self.p1, self.p2, prev)
        if self._dirs[0] == "down":
            direction = "down" if side < 0 else "up"
        else:
            direction = "right" if side > 0 else "left"
        self._crossed[name][direction].add(track_id)
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
            "line": {"p1": list(self.p1), "p2": list(self.p2)},
            "total": self.total(),
            "breakdown": self.breakdown(),
            "track_ids": {
                n: {d: sorted(self._crossed[n][d]) for d in self._dirs}
                for n in self._crossed
            },
        }


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
    p.add_argument("--conf", type=float, default=0.5,
                   help="Detection confidence threshold 0..1 (default: 0.5). "
                        "Higher = fewer false positives.")
    p.add_argument("--min-frames", type=int, default=5,
                   help="Minimum frames a track must appear in to be counted "
                        "(default: 5). Higher = filters out ephemeral re-IDs.")
    p.add_argument("--line", type=str, default=None,
                   help="Counting segment as 'x1,y1,x2,y2'. Enables line-crossing "
                        "mode (overrides --min-frames). Use --pick-line for "
                        "interactive selection.")
    p.add_argument("--line-y", type=int, default=None,
                   help="Shorthand: full-width horizontal line at row Y.")
    p.add_argument("--line-x", type=int, default=None,
                   help="Shorthand: full-height vertical line at column X.")
    p.add_argument("--preview-frame", action="store_true",
                   help="Save the first frame of the source as PNG to "
                        "<output-dir>/preview_<timestamp>.png and exit. Use to "
                        "pick pixel coordinates for --line-y/--line-x.")
    p.add_argument("--pick-line", action="store_true",
                   help="Open the first frame in a window and let you click "
                        "two points to define the counting line interactively. "
                        "Snaps to horizontal/vertical based on dominant axis.")
    return p.parse_args(argv)


import json
import sys
import time
from datetime import datetime

import cv2
from ultralytics import YOLO


def resolve_source(source: str, duration: float = 30.0, output_dir: str = "output") -> str | int:
    """Translate a user-supplied source into something OpenCV can open.

    - Integer string ("0", "1") -> int (webcam index).
    - YouTube URL -> downloads `duration + 5` seconds via yt-dlp to
      `<output_dir>/buffer_<timestamp>.ts` and returns that local path.
    - Anything else -> returned as-is (file path, RTSP, HTTP .mp4/.ts).
    """
    try:
        return int(source)
    except ValueError:
        pass

    if "youtube.com" in source or "youtu.be" in source:
        import subprocess
        from datetime import datetime as _dt
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        buffer_path = Path(output_dir) / f"buffer_{_dt.now().strftime('%Y-%m-%d_%H-%M-%S')}.ts"
        record_seconds = int(duration + 5)
        print(f"Recording {record_seconds}s of livestream via yt-dlp to {buffer_path} ...")
        ytdlp_exe = Path(sys.executable).parent / "yt-dlp.exe"
        cmd = [
            str(ytdlp_exe),
            "-f", "best[height<=720]",
            "--hls-use-mpegts",
            "--downloader", "ffmpeg",
            "--downloader-args", f"ffmpeg:-t {record_seconds}",
            "-o", str(buffer_path),
            source,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not buffer_path.exists() or buffer_path.stat().st_size == 0:
            print(f"ERROR: yt-dlp failed to record stream.\nstdout: {result.stdout}\nstderr: {result.stderr}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Recorded {buffer_path.stat().st_size // 1024} KiB.")
        return str(buffer_path)

    return source


def open_capture(source: str, duration: float, output_dir: str) -> cv2.VideoCapture:
    src = resolve_source(source, duration=duration, output_dir=output_dir)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: could not open source '{source}'. "
              f"Check the URL/path, network, or credentials.", file=sys.stderr)
        sys.exit(1)
    return cap


def make_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


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


def _snap_line(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[str, int]:
    """Snap a two-point line to the dominant axis.

    Returns ("y", y) for a horizontal line at row y, or ("x", x) for a vertical
    line at column x.
    """
    x1, y1 = p1
    x2, y2 = p2
    if abs(x2 - x1) >= abs(y2 - y1):
        return "y", (y1 + y2) // 2
    return "x", (x1 + x2) // 2


_MIN_DRAG_PX = 10  # ignore accidental clicks shorter than this in both axes


def _pick_line_interactive(frame) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Open a window, let user click-and-drag to define a counting line segment.

    Returns (p1, p2) endpoints, or None if cancelled.
    """
    state: dict = {"p1": None, "p2": None, "dragging": False, "hover": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["p1"] = (x, y)
            state["p2"] = None
            state["dragging"] = True
            state["hover"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["hover"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            if state["p1"] is not None:
                dx = abs(x - state["p1"][0])
                dy = abs(y - state["p1"][1])
                if dx >= _MIN_DRAG_PX or dy >= _MIN_DRAG_PX:
                    state["p2"] = (x, y)
                else:
                    state["p1"] = None  # accidental click

    win = "Pick segment  (drag A->B,  Enter=ok, R=redo, Esc=cancel)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    h, w = frame.shape[:2]

    while True:
        display = frame.copy()
        if state["p1"] is None:
            msg = "Click and drag from A to B to draw the segment"
        elif state["dragging"]:
            cv2.circle(display, state["p1"], 5, (0, 0, 255), -1)
            if state["hover"] is not None:
                cv2.line(display, state["p1"], state["hover"], (0, 255, 255), 2)
            msg = "Release to confirm"
        elif state["p2"] is not None:
            cv2.line(display, state["p1"], state["p2"], (0, 255, 255), 2)
            cv2.circle(display, state["p1"], 4, (0, 0, 255), -1)
            cv2.circle(display, state["p2"], 4, (0, 0, 255), -1)
            msg = (f"segment {state['p1']}->{state['p2']}  "
                   f"Enter=ok  R=redo  Esc=cancel")
        else:
            msg = "Click and drag from A to B to draw the segment"

        cv2.rectangle(display, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(display, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, display)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyWindow(win)
            return None
        if k in (13, 10) and state["p1"] is not None and state["p2"] is not None:
            p1, p2 = state["p1"], state["p2"]
            cv2.destroyWindow(win)
            return p1, p2
        if k in (ord("r"), ord("R")):
            state["p1"] = None
            state["p2"] = None
            state["dragging"] = False


def _parse_line_arg(s: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse a '--line x1,y1,x2,y2' string into two endpoints."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--line must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = (int(p) for p in parts)
    return (x1, y1), (x2, y2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    explicit_line_modes = sum(
        v is not None for v in (args.line, args.line_y, args.line_x)
    )
    if explicit_line_modes > 1:
        print("ERROR: pass only one of --line, --line-y, --line-x.", file=sys.stderr)
        return 2

    if args.preview_frame:
        return _preview_frame(args)

    print(f"Loading model {args.model}...")
    model = YOLO(args.model)

    cap = open_capture(args.source, args.duration, args.output_dir)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_stream = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_stream if 1.0 < fps_stream < 120.0 else 25.0

    # Resolve the counting segment (if any) from CLI flags or interactive picker.
    segment: tuple[tuple[int, int], tuple[int, int]] | None = None
    if args.line is not None:
        try:
            segment = _parse_line_arg(args.line)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            cap.release()
            return 2
    elif args.line_y is not None:
        segment = ((0, args.line_y), (width, args.line_y))
    elif args.line_x is not None:
        segment = ((args.line_x, 0), (args.line_x, height))
    elif args.pick_line:
        ok, first = cap.read()
        if not ok or first is None:
            print("ERROR: could not read first frame for line picker.", file=sys.stderr)
            cap.release()
            return 1
        segment = _pick_line_interactive(first)
        if segment is None:
            print("Line picker cancelled.", file=sys.stderr)
            cap.release()
            return 2
        print(f"Segment picked: {segment[0]} -> {segment[1]}")

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

    use_line = segment is not None
    counter: VehicleCounter | LineCrossingCounter
    if use_line:
        counter = LineCrossingCounter(p1=segment[0], p2=segment[1])
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
            if use_line and segment is not None:
                cv2.line(annotated, segment[0], segment[1], (0, 255, 255), 2)

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

    breakdown = counter.breakdown()
    print()
    print(f"Han pasado {counter.total()} vehículos en {duration_real:.1f} s "
          f"({frame_count} frames procesados)")
    if use_line:
        # breakdown is dict[class, dict[direction, count]]
        dir_names = counter._dirs  # ("down","up") or ("right","left")
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
