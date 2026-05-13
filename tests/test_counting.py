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


def test_line_y_counts_top_to_bottom_crossing():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=80)   # car above line
    c.observe(track_id=1, class_id=2, cx=55, cy=120)  # now below -> crossed down
    assert c.total() == 1
    bd = c.breakdown()
    assert bd["car"]["down"] == 1
    assert bd["car"]["up"] == 0


def test_line_y_counts_bottom_to_top_crossing():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=150)
    c.observe(track_id=1, class_id=2, cx=55, cy=50)
    assert c.total() == 1
    assert c.breakdown()["car"]["up"] == 1
    assert c.breakdown()["car"]["down"] == 0


def test_line_y_no_count_when_not_crossing():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=20)
    c.observe(track_id=1, class_id=2, cx=55, cy=30)
    assert c.total() == 0


def test_line_y_no_double_count_same_track():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=80)
    c.observe(track_id=1, class_id=2, cx=55, cy=120)  # cross
    c.observe(track_id=1, class_id=2, cx=60, cy=80)   # cross back
    c.observe(track_id=1, class_id=2, cx=65, cy=120)  # cross again
    # Still counted once — first crossing wins.
    assert c.total() == 1


def test_line_x_counts_left_to_right_crossing():
    c = LineCrossingCounter(line_x=200)
    c.observe(track_id=1, class_id=5, cx=180, cy=50)  # bus left of line
    c.observe(track_id=1, class_id=5, cx=220, cy=55)  # now right -> crossed "right"
    assert c.total() == 1
    assert c.breakdown()["bus"]["right"] == 1
    assert c.breakdown()["bus"]["left"] == 0


def test_line_x_counts_right_to_left_crossing():
    c = LineCrossingCounter(line_x=200)
    c.observe(track_id=1, class_id=5, cx=220, cy=50)
    c.observe(track_id=1, class_id=5, cx=180, cy=55)
    assert c.breakdown()["bus"]["left"] == 1


def test_line_counter_ignores_non_vehicle_class():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=0, cx=50, cy=80)   # person
    c.observe(track_id=1, class_id=0, cx=55, cy=120)
    assert c.total() == 0


def test_line_counter_requires_exactly_one_axis():
    import pytest as _pt
    with _pt.raises(ValueError):
        LineCrossingCounter()
    with _pt.raises(ValueError):
        LineCrossingCounter(line_y=100, line_x=200)


def test_line_counter_first_observation_does_not_count():
    # A track that appears already past the line shouldn't be counted retroactively.
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=120)
    assert c.total() == 0


def test_line_counter_summary_shape():
    c = LineCrossingCounter(line_y=100)
    c.observe(track_id=1, class_id=2, cx=50, cy=80)
    c.observe(track_id=1, class_id=2, cx=55, cy=120)
    summary = c.summary(source="x.mp4", duration_real=2.5, model="yolov8n.pt")
    assert summary["total"] == 1
    assert summary["source"] == "x.mp4"
    assert summary["duration_real"] == 2.5
    assert summary["model"] == "yolov8n.pt"
    assert summary["line_y"] == 100
    assert summary["line_x"] is None
    assert summary["breakdown"]["car"]["down"] == 1
    assert summary["breakdown"]["car"]["up"] == 0
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
