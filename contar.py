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
