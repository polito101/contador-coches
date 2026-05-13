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
