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
        if line_y is not None:
            dirs = ("down", "up")
        else:
            dirs = ("right", "left")
        self._dirs = dirs
        self._crossed: dict[str, dict[str, set[int]]] = {
            name: {d: set() for d in dirs}
            for name in VEHICLE_CLASSES.values()
        }
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Loading model {args.model}...")
    model = YOLO(args.model)

    cap = open_capture(args.source, args.duration, args.output_dir)
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
