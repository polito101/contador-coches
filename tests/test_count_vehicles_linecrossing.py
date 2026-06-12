"""Cross-repo Wave-0 tests for COUNT-01 (live-bets Phase 25).

`count_vehicles_linecrossing` must emit a per-crossing ``events`` list so the
live-bets widget counter can ramp in sync with the playing clip. Each event is
``{at, class, direction, track_id}`` where ``at`` is the *video-time* of the
crossing (``frame_index / fps`` seconds, NOT wall-clock).

The contract (live-bets 25-CONTEXT.md D-11 / D-12):
- ``events`` is the per-crossing decomposition of the SAME number ``total``
  already reports, so ``len(events) == total`` BY CONSTRUCTION (one event is
  appended at the single ``_already_counted.add`` crossing site).
- ``at`` is monotonic non-decreasing in the order the crossings are observed.

These tests drive ``LineCrossingCounter`` directly (no cv2 / ultralytics), so
they run on the dev box and CI without the ``[detector]`` extras. They reuse the
crossing geometry from ``tests/test_counting.py`` (a horizontal y=100 segment
spanning x=0..200; a centroid path from above to below it crosses it once).
"""
from __future__ import annotations

from contador_coches.contar import LineCrossingCounter


def _drive_three_crossings() -> LineCrossingCounter:
    """Three distinct tracks crossing the y=100 segment at increasing frames.

    fps is fixed at 10.0 so the expected ``at`` values are exact, terminating
    decimals (frame/10) — no float-rounding ambiguity in the assertions.

    - track 1 (car, down): above on frame 5, below on frame 6  -> at = 0.6
    - track 2 (motorcycle, down): above on frame 14, below on frame 15 -> at = 1.5
    - track 3 (bus, up): below on frame 24, above on frame 25 -> at = 2.5
    """
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100), fps=10.0)
    # track 1 — car (class 2), crosses downward between frame 5 and 6.
    c.observe(track_id=1, class_id=2, cx=50, cy=80, frame_index=5)
    c.observe(track_id=1, class_id=2, cx=55, cy=120, frame_index=6)
    # track 2 — motorcycle (class 3), crosses downward between frame 14 and 15.
    c.observe(track_id=2, class_id=3, cx=70, cy=70, frame_index=14)
    c.observe(track_id=2, class_id=3, cx=75, cy=130, frame_index=15)
    # track 3 — bus (class 5), crosses upward between frame 24 and 25.
    c.observe(track_id=3, class_id=5, cx=90, cy=150, frame_index=24)
    c.observe(track_id=3, class_id=5, cx=95, cy=50, frame_index=25)
    return c


def test_events_reconcile_with_total():
    """len(events) == total — events are the decomposition of the SAME number (D-12)."""
    c = _drive_three_crossings()
    assert c.total() == 3
    assert len(c.events) == c.total()


def test_events_have_float_at_and_are_monotonic():
    """Each event carries a float video-time ``at``, non-decreasing in record order."""
    c = _drive_three_crossings()
    ats = [e["at"] for e in c.events]
    assert all(isinstance(a, float) for a in ats)
    assert ats == sorted(ats), f"events must be monotonic non-decreasing, got {ats}"
    # at = frame_index / fps (the SECOND, below-the-line frame is the crossing frame).
    assert ats == [0.6, 1.5, 2.5]


def test_event_fields_match_observed_crossing():
    """class / direction / track_id are present and match what observe() saw."""
    c = _drive_three_crossings()
    by_track = {e["track_id"]: e for e in c.events}
    assert set(by_track) == {1, 2, 3}
    # Field types per the locked event shape {at: float, class: str, direction: str, track_id: int}.
    for ev in c.events:
        assert isinstance(ev["at"], float)
        assert isinstance(ev["class"], str)
        assert isinstance(ev["direction"], str)
        assert isinstance(ev["track_id"], int)
    assert by_track[1]["class"] == "car"
    assert by_track[1]["direction"] == "down"
    assert by_track[2]["class"] == "motorcycle"
    assert by_track[2]["direction"] == "down"
    assert by_track[3]["class"] == "bus"
    assert by_track[3]["direction"] == "up"


def test_no_crossing_yields_no_events():
    """A path that never crosses the segment records no events (and total 0)."""
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100), fps=10.0)
    c.observe(track_id=1, class_id=2, cx=50, cy=20, frame_index=0)
    c.observe(track_id=1, class_id=2, cx=55, cy=30, frame_index=1)
    assert c.total() == 0
    assert c.events == []


def test_recrossing_same_track_appends_one_event():
    """A track counted once never appends a second event (matches no-double-count)."""
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100), fps=10.0)
    c.observe(track_id=1, class_id=2, cx=50, cy=80, frame_index=2)
    c.observe(track_id=1, class_id=2, cx=55, cy=120, frame_index=3)  # cross (counted)
    c.observe(track_id=1, class_id=2, cx=60, cy=80, frame_index=4)   # cross back
    c.observe(track_id=1, class_id=2, cx=65, cy=120, frame_index=5)  # again
    assert c.total() == 1
    assert len(c.events) == 1
    assert c.events[0]["at"] == 0.3


def test_events_carry_normalized_xy_when_frame_size_known():
    """PLUS-ONE: with frame_size set, each event carries the crossing-frame
    centroid normalized to 0..1 (4 decimals) — the live-bets widget uses it to
    place the '+1' at the crossing point regardless of rendered video size."""
    c = LineCrossingCounter(
        p1=(0, 100), p2=(200, 100), fps=10.0, frame_size=(640, 360)
    )
    c.observe(track_id=1, class_id=2, cx=50, cy=80, frame_index=5)
    c.observe(track_id=1, class_id=2, cx=55, cy=120, frame_index=6)
    (ev,) = c.events
    # The CROSSING frame's centroid (the second observe) is the recorded point.
    assert ev["x"] == round(55 / 640, 4) == 0.0859
    assert ev["y"] == round(120 / 360, 4) == 0.3333
    assert 0.0 <= ev["x"] <= 1.0 and 0.0 <= ev["y"] <= 1.0


def test_events_omit_xy_without_frame_size():
    """Back-compat: no frame_size -> the keys are ABSENT (not null) so old
    consumers and byte-stable expectations never see them."""
    c = _drive_three_crossings()
    for ev in c.events:
        assert "x" not in ev and "y" not in ev


def test_total_at_and_existing_fields_unchanged_by_frame_size():
    """frame_size only ADDS keys — total / at / class / direction / track_id
    are byte-identical to the frame_size-less drive."""
    c = LineCrossingCounter(
        p1=(0, 100), p2=(200, 100), fps=10.0, frame_size=(640, 360)
    )
    c.observe(track_id=1, class_id=2, cx=50, cy=80, frame_index=5)
    c.observe(track_id=1, class_id=2, cx=55, cy=120, frame_index=6)
    c.observe(track_id=2, class_id=3, cx=70, cy=70, frame_index=14)
    c.observe(track_id=2, class_id=3, cx=75, cy=130, frame_index=15)
    c.observe(track_id=3, class_id=5, cx=90, cy=150, frame_index=24)
    c.observe(track_id=3, class_id=5, cx=95, cy=50, frame_index=25)
    assert c.total() == 3
    assert [e["at"] for e in c.events] == [0.6, 1.5, 2.5]
    stripped = [
        {k: v for k, v in e.items() if k not in ("x", "y")} for e in c.events
    ]
    assert stripped == _drive_three_crossings().events
