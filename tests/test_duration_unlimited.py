"""Regression tests for ``duration <= 0`` meaning 'process the whole clip'.

The headless counters (`count_vehicles_minframes`, `count_vehicles_linecrossing`)
are used by live-bets' offline ingestion, which passes ``duration=0.0`` to mean
"no wall-clock cap, count until the video ends". The old loop used a bare
``elapsed >= duration`` break, so ``duration=0`` stopped after a single frame —
yielding ``total=0`` for every clip (and, downstream, settlements liquidating at
zero). These tests pin the contract: ``duration <= 0`` processes every frame.

cv2 + ultralytics are faked via ``sys.modules`` so the test is deterministic
(no dependence on CPU speed — the exact failure mode a wall-clock cap causes)
and needs neither the ~200 MB CV stack nor a real video clip.
"""
from __future__ import annotations

import sys
import types

from contador_coches.contar import (
    count_vehicles_linecrossing,
    count_vehicles_minframes,
)


class _FakeArr:
    """Mimics a torch tensor's ``.int().cpu().tolist()`` / ``.cpu().numpy()`` chain."""

    def __init__(self, data: list) -> None:
        self._data = data

    def int(self) -> "_FakeArr":
        return self

    def cpu(self) -> "_FakeArr":
        return self

    def tolist(self) -> list:
        return list(self._data)

    def numpy(self) -> list:
        return list(self._data)


class _FakeBoxes:
    def __init__(self, ids: list, clss: list, xywh: list | None = None) -> None:
        self.id = _FakeArr(ids)
        self.cls = _FakeArr(clss)
        self.xywh = _FakeArr(xywh if xywh is not None else [])


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes


def _fake_cv2_module(num_frames: int) -> types.ModuleType:
    """A stand-in ``cv2`` whose ``VideoCapture`` yields exactly ``num_frames`` frames."""

    class _FakeCapture:
        def __init__(self, source) -> None:  # noqa: ANN001 - mirrors cv2 API
            self._remaining = num_frames

        def isOpened(self) -> bool:
            return True

        def read(self):
            if self._remaining <= 0:
                return False, None
            self._remaining -= 1
            return True, object()  # opaque frame; the fake model never inspects it

        def get(self, prop):  # noqa: ANN001 - cv2 API
            return 0.0

        def release(self) -> None:
            pass

    mod = types.ModuleType("cv2")
    mod.VideoCapture = _FakeCapture  # type: ignore[attr-defined]
    mod.CAP_PROP_FPS = 5  # type: ignore[attr-defined]
    return mod


def _fake_ultralytics_module(boxes_for_call) -> types.ModuleType:  # noqa: ANN001
    """A stand-in ``ultralytics`` whose ``YOLO(...).track(...)`` returns scripted boxes.

    ``boxes_for_call(i)`` is called with the 0-based frame index and must return
    a ``_FakeBoxes`` instance (the detections for that frame).
    """

    class _FakeYOLO:
        def __init__(self, model_path) -> None:  # noqa: ANN001
            self._i = 0

        def track(self, frame, **kwargs):  # noqa: ANN001
            boxes = boxes_for_call(self._i)
            self._i += 1
            return [_FakeResult(boxes)]

    mod = types.ModuleType("ultralytics")
    mod.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    return mod


def test_minframes_duration_zero_processes_whole_clip(monkeypatch):
    """duration=0 must process every frame, not stop after the first.

    A car (track id 1) is present in all 12 frames; with min_frames=5 it is only
    counted if >=5 frames are processed. The old 1-frame cap yielded total=0.
    """
    num_frames = 12
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module(num_frames))
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        _fake_ultralytics_module(lambda i: _FakeBoxes(ids=[1], clss=[2])),  # 2 == car
    )

    result = count_vehicles_minframes(
        source="fake.mp4",
        duration=0.0,
        model_path="ignored.pt",
        min_frames=5,
        conf=0.25,
    )

    assert result["frames_processed"] == num_frames
    assert result["total"] == 1
    assert result["breakdown"]["car"] == 1


def test_minframes_negative_duration_is_also_unlimited(monkeypatch):
    """duration<0 behaves the same as duration==0: no wall-clock cap."""
    num_frames = 7
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module(num_frames))
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        _fake_ultralytics_module(lambda i: _FakeBoxes(ids=[1], clss=[2])),
    )

    result = count_vehicles_minframes(
        source="fake.mp4",
        duration=-1.0,
        model_path="ignored.pt",
        min_frames=5,
        conf=0.25,
    )

    assert result["frames_processed"] == num_frames
    assert result["total"] == 1


def test_minframes_positive_duration_cap_still_honored(monkeypatch):
    """A positive duration must still cap processing (the >0 branch is unchanged).

    With duration tiny-but-positive and many frames available, the loop must stop
    on the wall-clock check rather than draining all frames.
    """
    num_frames = 10_000  # far more than can be processed under the cap
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module(num_frames))
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        _fake_ultralytics_module(lambda i: _FakeBoxes(ids=[1], clss=[2])),
    )

    result = count_vehicles_minframes(
        source="fake.mp4",
        duration=0.001,
        model_path="ignored.pt",
        min_frames=5,
        conf=0.25,
    )

    # The cap fires well before all 10k frames are read.
    assert result["frames_processed"] < num_frames


def test_linecrossing_duration_zero_processes_whole_clip(monkeypatch):
    """duration=0 must process every frame for the line-crossing counter too.

    Track 1 sits above the y=100 segment on frame 0 and below it from frame 1 on,
    so the crossing is only detectable once >=2 frames are processed. The old
    1-frame cap missed it entirely (total=0).
    """
    num_frames = 8
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_module(num_frames))

    def boxes_for_call(i):
        cy = 50 if i == 0 else 150  # above the line on frame 0, below it after
        return _FakeBoxes(ids=[1], clss=[2], xywh=[(50, cy, 10, 10)])

    monkeypatch.setitem(
        sys.modules, "ultralytics", _fake_ultralytics_module(boxes_for_call)
    )

    result = count_vehicles_linecrossing(
        source="fake.mp4",
        duration=0.0,
        p1=(0, 100),
        p2=(200, 100),
        model_path="ignored.pt",
        conf=0.25,
    )

    assert result["frames_processed"] == num_frames
    assert result["total"] == 1
    assert result["breakdown"]["car"]["down"] == 1
