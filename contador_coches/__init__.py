"""contador-coches — YOLO+ByteTrack vehicle counter for live-bets.

Distribution name: ``contador-coches`` (hyphen).
Import name: ``contador_coches`` (underscore).

Public surface:
- ``__version__`` — semver, bumped per D-06 when any output of
  ``count_vehicles_minframes`` would change for the same input (changes to
  ``min_frames``, ``conf``, or the YOLO model file), or when public function
  signatures change. Tags are immutable: bumping creates a new
  ``v0.1.1`` / ``v0.2.0`` rather than moving ``v0.1.0``.
- ``count_vehicles_minframes`` — headless minimum-frames counter.
- ``count_vehicles_linecrossing`` — headless line-crossing counter.
- ``resolve_source`` — yt-dlp helper for YouTube live URLs.
- ``get_default_model_path`` — resolves the bundled ``yolov8n.pt`` via
  ``importlib.resources`` (no CWD assumptions).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__version__ = "0.1.0"

# Re-export the public callables from the implementation module.
# Lazy heavy imports (cv2, ultralytics) live inside the function bodies so
# ``import contador_coches`` stays cheap for callers that only need
# ``__version__`` or ``get_default_model_path`` (RESEARCH §2 Landmine).
from .contar import (
    count_vehicles_linecrossing,
    count_vehicles_minframes,
    resolve_source,
)


def get_default_model_path() -> Path:
    """Resolve the bundled YOLOv8 nano weights as a filesystem path.

    Uses ``importlib.resources`` so the path works whether the package is
    installed editable (``pip install -e .``) or from a wheel.
    """
    return Path(str(files("contador_coches.weights") / "yolov8n.pt"))


__all__ = [
    "__version__",
    "count_vehicles_linecrossing",
    "count_vehicles_minframes",
    "get_default_model_path",
    "resolve_source",
]
