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
    try:
        return int(source)
    except ValueError:
        pass

    if "youtube.com" in source or "youtu.be" in source:
        from yt_dlp import YoutubeDL
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[height<=720][protocol^=m3u8]/best[height<=720]",
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=False)
            url = info.get("url")
            if not url:
                print("ERROR: could not extract stream URL from YouTube source.",
                      file=sys.stderr)
                sys.exit(1)
            return url

    return source


def open_capture(source: str) -> cv2.VideoCapture:
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
