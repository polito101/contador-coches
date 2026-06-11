"""contador-coches — YOLO+ByteTrack vehicle counter for live-bets.

Distribution name: ``contador-coches`` (hyphen).
Import name: ``contador_coches`` (underscore).

Public surface:
- ``__version__`` — semver, bumped per D-06 when any output of
  ``count_vehicles_minframes`` would change for the same input (changes to
  ``min_frames``, ``conf``, or the YOLO model), or when public function
  signatures change. Tags are immutable: bumping creates a new
  ``v0.2.0`` / ``v0.3.0`` rather than moving an existing tag.
- ``count_vehicles_minframes`` — headless minimum-frames counter.
- ``count_vehicles_linecrossing`` — headless line-crossing counter.
- ``resolve_source`` — yt-dlp helper for YouTube live URLs.
- ``get_default_model_path`` — the default YOLO model name (``yolo26x.pt``),
  auto-downloaded + cached by ultralytics on first use (no CWD assumptions).
"""

from __future__ import annotations

from pathlib import Path

# v0.2.0 — switched the detection model yolov8n.pt -> yolo26x.pt (YOLO26, best
# variant; unanimous decision 2026-06-10, same model for cars and birds). The
# model change alters detection output for the same input, so this MUST be a
# semver bump (it propagates a new indexer_version '+yolo26x' -> re-index).
__version__ = "0.3.0"

# Re-export the public callables from the implementation module.
# Lazy heavy imports (cv2, ultralytics) live inside the function bodies so
# ``import contador_coches`` stays cheap for callers that only need
# ``__version__`` or ``get_default_model_path`` (RESEARCH §2 Landmine).
from .contar import (
    count_vehicles_linecrossing,
    count_vehicles_minframes,
    resolve_source,
)

# The default YOLO model. A bare name (not a bundled file) so ultralytics
# downloads + caches the official weight on first use — yolo26x.pt is ~118 MB,
# too large to ship in the package/git (>GitHub's 100 MB/file limit). The
# weight is COCO-pretrained, so vehicle (2/3/5/7) detection needs no retraining.
_DEFAULT_MODEL = "yolo26x.pt"


def get_default_model_path() -> Path:
    """Return the default YOLO model identifier (``yolo26x.pt``).

    Not a filesystem path to a bundled file: ultralytics resolves this name and
    auto-downloads/caches the weight on first ``YOLO("yolo26x.pt")`` call. Kept
    as a ``Path`` for caller compatibility; ``.stem`` ("yolo26x") feeds the
    live-bets ``indexer_version``.
    """
    return Path(_DEFAULT_MODEL)


__all__ = [
    "__version__",
    "count_vehicles_linecrossing",
    "count_vehicles_minframes",
    "get_default_model_path",
    "resolve_source",
]
