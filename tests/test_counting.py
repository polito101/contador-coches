from contar import VehicleCounter, VEHICLE_CLASSES


def test_counter_starts_empty():
    c = VehicleCounter()
    assert c.total() == 0
    assert c.breakdown() == {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}


def test_counter_adds_unique_ids_per_class():
    c = VehicleCounter()
    c.add(track_id=1, class_id=2)   # car
    c.add(track_id=2, class_id=2)   # car
    c.add(track_id=3, class_id=7)   # truck
    assert c.total() == 3
    assert c.breakdown() == {"car": 2, "motorcycle": 0, "bus": 0, "truck": 1}


def test_counter_dedupes_same_id_same_class():
    c = VehicleCounter()
    for _ in range(50):
        c.add(track_id=5, class_id=2)
    assert c.total() == 1
    assert c.breakdown()["car"] == 1


def test_counter_ignores_non_vehicle_class():
    c = VehicleCounter()
    c.add(track_id=1, class_id=0)   # person (COCO id 0)
    assert c.total() == 0


def test_counter_same_id_different_class_counts_twice():
    c = VehicleCounter()
    c.add(track_id=1, class_id=2)
    c.add(track_id=1, class_id=7)
    assert c.total() == 2


def test_summary_dict_shape():
    c = VehicleCounter()
    c.add(track_id=1, class_id=2)
    c.add(track_id=2, class_id=5)
    summary = c.summary(
        source="test.mp4",
        duration_real=12.3,
        model="yolov8n.pt",
    )
    assert summary["total"] == 2
    assert summary["source"] == "test.mp4"
    assert summary["duration_real"] == 12.3
    assert summary["model"] == "yolov8n.pt"
    assert summary["breakdown"]["car"] == 1
    assert summary["breakdown"]["bus"] == 1
    assert set(summary["track_ids"]["car"]) == {1}
    assert set(summary["track_ids"]["bus"]) == {2}


def test_vehicle_classes_constant():
    assert VEHICLE_CLASSES == {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


from contar import parse_args


def test_parse_args_defaults():
    args = parse_args(["--source", "foo.mp4"])
    assert args.source == "foo.mp4"
    assert args.duration == 30.0
    assert args.model == "yolov8n.pt"
    assert args.output_dir == "output"
    assert args.no_save is False
    assert args.no_display is False
    assert args.conf == 0.5
    assert args.min_frames == 5


def test_parse_args_overrides():
    args = parse_args([
        "--source", "rtsp://x", "--duration", "5",
        "--model", "yolov8s.pt", "--no-save", "--no-display",
        "--conf", "0.7", "--min-frames", "10",
    ])
    assert args.source == "rtsp://x"
    assert args.duration == 5.0
    assert args.model == "yolov8s.pt"
    assert args.no_save is True
    assert args.no_display is True
    assert args.conf == 0.7
    assert args.min_frames == 10


def test_counter_filters_below_min_frames():
    c = VehicleCounter(min_frames=5)
    # track 1: seen 5 times -> kept
    for _ in range(5):
        c.add(track_id=1, class_id=2)
    # track 2: seen 4 times -> dropped
    for _ in range(4):
        c.add(track_id=2, class_id=2)
    assert c.total() == 1
    assert c.breakdown()["car"] == 1


def test_counter_min_frames_default_keeps_everything():
    c = VehicleCounter()  # min_frames=1
    c.add(track_id=1, class_id=2)
    c.add(track_id=2, class_id=2)
    assert c.total() == 2


def test_summary_includes_min_frames():
    c = VehicleCounter(min_frames=3)
    summary = c.summary(source="x", duration_real=1.0, model="m")
    assert summary["min_frames"] == 3


from contar import LineCrossingCounter


def test_segment_counts_top_to_bottom_crossing():
    # Horizontal segment from (0,100) to (200,100). Vehicle crosses through it.
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100))
    c.observe(track_id=1, class_id=2, cx=50, cy=80)
    c.observe(track_id=1, class_id=2, cx=55, cy=120)
    assert c.total() == 1
    assert c.breakdown()["car"] == {"down": 1, "up": 0}


def test_segment_counts_bottom_to_top_crossing():
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100))
    c.observe(track_id=1, class_id=2, cx=50, cy=150)
    c.observe(track_id=1, class_id=2, cx=55, cy=50)
    assert c.breakdown()["car"]["up"] == 1


def test_segment_does_not_count_path_outside_segment():
    # Segment only spans x=0..200. Vehicle path crosses y=100 at x=400 — outside.
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100))
    c.observe(track_id=1, class_id=2, cx=400, cy=80)
    c.observe(track_id=1, class_id=2, cx=405, cy=120)
    assert c.total() == 0


def test_segment_no_count_when_not_crossing():
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100))
    c.observe(track_id=1, class_id=2, cx=50, cy=20)
    c.observe(track_id=1, class_id=2, cx=55, cy=30)
    assert c.total() == 0


def test_segment_no_double_count_same_track():
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100))
    c.observe(track_id=1, class_id=2, cx=50, cy=80)
    c.observe(track_id=1, class_id=2, cx=55, cy=120)  # cross
    c.observe(track_id=1, class_id=2, cx=60, cy=80)   # cross back
    c.observe(track_id=1, class_id=2, cx=65, cy=120)  # again
    assert c.total() == 1


def test_vertical_segment_left_to_right():
    c = LineCrossingCounter(p1=(200, 0), p2=(200, 200))
    c.observe(track_id=1, class_id=5, cx=180, cy=50)
    c.observe(track_id=1, class_id=5, cx=220, cy=55)
    assert c.breakdown()["bus"] == {"right": 1, "left": 0}


def test_vertical_segment_right_to_left():
    c = LineCrossingCounter(p1=(200, 0), p2=(200, 200))
    c.observe(track_id=1, class_id=5, cx=220, cy=50)
    c.observe(track_id=1, class_id=5, cx=180, cy=55)
    assert c.breakdown()["bus"]["left"] == 1


def test_diagonal_segment_uses_dominant_axis_naming():
    # Mostly-horizontal diagonal — labels should be down/up
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 110))
    c.observe(track_id=1, class_id=2, cx=100, cy=50)
    c.observe(track_id=1, class_id=2, cx=100, cy=200)
    bd = c.breakdown()
    assert "down" in bd["car"]
    assert bd["car"]["down"] == 1


def test_segment_direction_independent_of_drag_order():
    # Same segment, drawn either way, should yield the same direction labels.
    a = LineCrossingCounter(p1=(0, 100), p2=(200, 100))
    b = LineCrossingCounter(p1=(200, 100), p2=(0, 100))
    for c in (a, b):
        c.observe(track_id=1, class_id=2, cx=50, cy=50)
        c.observe(track_id=1, class_id=2, cx=55, cy=150)
    assert a.breakdown()["car"]["down"] == 1
    assert b.breakdown()["car"]["down"] == 1


def test_line_counter_ignores_non_vehicle_class():
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100))
    c.observe(track_id=1, class_id=0, cx=50, cy=80)
    c.observe(track_id=1, class_id=0, cx=55, cy=120)
    assert c.total() == 0


def test_line_counter_rejects_zero_length_segment():
    import pytest as _pt
    with _pt.raises(ValueError):
        LineCrossingCounter(p1=(10, 10), p2=(10, 10))


def test_line_counter_first_observation_does_not_count():
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100))
    c.observe(track_id=1, class_id=2, cx=50, cy=120)
    assert c.total() == 0


def test_line_counter_summary_shape():
    c = LineCrossingCounter(p1=(0, 100), p2=(200, 100))
    c.observe(track_id=1, class_id=2, cx=50, cy=80)
    c.observe(track_id=1, class_id=2, cx=55, cy=120)
    summary = c.summary(source="x.mp4", duration_real=2.5, model="yolov8n.pt")
    assert summary["total"] == 1
    assert summary["line"] == {"p1": [0, 100], "p2": [200, 100]}
    assert summary["breakdown"]["car"]["down"] == 1
    assert summary["track_ids"]["car"]["down"] == [1]


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


def test_parse_args_line_segment():
    args = parse_args(["--source", "foo.mp4", "--line", "10,20,300,40"])
    assert args.line == "10,20,300,40"


from contar import _parse_line_arg


def test_parse_line_arg_valid():
    assert _parse_line_arg("10,20,300,40") == ((10, 20), (300, 40))


def test_parse_line_arg_invalid():
    import pytest as _pt
    with _pt.raises(ValueError):
        _parse_line_arg("10,20,300")  # only 3 parts


from contar import _load_lines_cache, _save_line_to_cache, _segment_from_cache_entry


def test_lines_cache_roundtrip(tmp_path):
    p = tmp_path / "lines.json"
    assert _load_lines_cache(p) == {}
    _save_line_to_cache(p, "src.mp4", ((10, 20), (300, 40)))
    cache = _load_lines_cache(p)
    assert cache == {"src.mp4": [[10, 20], [300, 40]]}


def test_lines_cache_overwrites_same_source(tmp_path):
    p = tmp_path / "lines.json"
    _save_line_to_cache(p, "src.mp4", ((10, 20), (300, 40)))
    _save_line_to_cache(p, "src.mp4", ((1, 2), (3, 4)))
    assert _load_lines_cache(p) == {"src.mp4": [[1, 2], [3, 4]]}


def test_lines_cache_keeps_other_sources(tmp_path):
    p = tmp_path / "lines.json"
    _save_line_to_cache(p, "a", ((0, 0), (1, 1)))
    _save_line_to_cache(p, "b", ((2, 2), (3, 3)))
    cache = _load_lines_cache(p)
    assert set(cache.keys()) == {"a", "b"}


def test_lines_cache_corrupt_file_returns_empty(tmp_path):
    p = tmp_path / "lines.json"
    p.write_text("not json", encoding="utf-8")
    assert _load_lines_cache(p) == {}


def test_segment_from_cache_entry_valid():
    assert _segment_from_cache_entry([[1, 2], [3, 4]]) == ((1, 2), (3, 4))


def test_segment_from_cache_entry_invalid():
    assert _segment_from_cache_entry("nope") is None
    assert _segment_from_cache_entry([[1, 2]]) is None


def test_parse_args_pick_line():
    args = parse_args(["--source", "foo.mp4"])
    assert args.pick_line is False
    args = parse_args(["--source", "foo.mp4", "--pick-line"])
    assert args.pick_line is True


from contar import _snap_line


def test_snap_line_horizontal_dominant():
    assert _snap_line((10, 100), (200, 110)) == ("y", 105)


def test_snap_line_vertical_dominant():
    assert _snap_line((100, 10), (110, 200)) == ("x", 105)


def test_snap_line_equal_axes_prefers_horizontal():
    assert _snap_line((0, 0), (100, 100)) == ("y", 50)
