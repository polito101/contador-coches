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
