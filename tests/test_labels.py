import pytest

from binarized_cv.data.labels import parse_yolo_label_file


def test_parses_multiple_boxes(tmp_path):
    label_path = tmp_path / "sample.txt"
    label_path.write_text("0 0.5 0.5 0.1 0.2\n0 0.25 0.75 0.05 0.05\n")

    boxes = parse_yolo_label_file(label_path)

    assert len(boxes) == 2
    assert boxes[0].class_id == 0
    assert boxes[0].cx == pytest.approx(0.5)
    assert boxes[1].w == pytest.approx(0.05)


def test_missing_file_returns_empty_list(tmp_path):
    assert parse_yolo_label_file(tmp_path / "missing.txt") == []


def test_empty_file_returns_empty_list(tmp_path):
    label_path = tmp_path / "empty.txt"
    label_path.write_text("")

    assert parse_yolo_label_file(label_path) == []


def test_malformed_line_raises(tmp_path):
    label_path = tmp_path / "bad.txt"
    label_path.write_text("0 0.5 0.5 0.1\n")

    with pytest.raises(ValueError):
        parse_yolo_label_file(label_path)
