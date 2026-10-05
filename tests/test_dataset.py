import dataclasses

import numpy as np
import pytest
import torch
from PIL import Image

from binarized_cv.data.dataset import MultispectralPersonDataset
from binarized_cv.data.records import PairRecord


def _write_pair(root):
    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(root / "rgb.jpg")
    Image.new("L", (32, 24), color=100).save(root / "ir.jpg")
    (root / "rgb.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    (root / "ir.txt").write_text("0 0.4 0.6 0.1 0.1\n")


def _record() -> PairRecord:
    return PairRecord(
        id="x",
        dataset="test",
        split="train",
        rgb_image="rgb.jpg",
        rgb_label="rgb.txt",
        ir_image="ir.jpg",
        ir_label="ir.txt",
    )


def test_getitem_resizes_both_modalities_to_img_size(tmp_path):
    _write_pair(tmp_path)
    dataset = MultispectralPersonDataset([_record()], tmp_path, img_size=(96, 128))

    item = dataset[0]

    assert item["rgb"].shape == (3, 96, 128)
    assert item["ir"].shape == (1, 96, 128)
    assert item["id"] == "x"
    assert item["dataset"] == "test"


def test_target_modality_rgb_uses_rgb_boxes(tmp_path):
    _write_pair(tmp_path)
    dataset = MultispectralPersonDataset([_record()], tmp_path, target_modality="rgb")

    targets = dataset[0]["targets"]

    torch.testing.assert_close(targets["boxes"], torch.tensor([[0.5, 0.5, 0.2, 0.3]]))
    assert targets["labels"].tolist() == [0]


def test_target_modality_ir_uses_ir_boxes(tmp_path):
    _write_pair(tmp_path)
    dataset = MultispectralPersonDataset([_record()], tmp_path, target_modality="ir")

    targets = dataset[0]["targets"]

    torch.testing.assert_close(targets["boxes"], torch.tensor([[0.4, 0.6, 0.1, 0.1]]))


def test_invalid_target_modality_rejected(tmp_path):
    with pytest.raises(ValueError):
        MultispectralPersonDataset([], tmp_path, target_modality="thermal")


def test_skips_records_unlabeled_in_target_modality(tmp_path):
    _write_pair(tmp_path)
    records = [_record(), dataclasses.replace(_record(), id="no_rgb", rgb_labeled=False)]

    assert [r.id for r in MultispectralPersonDataset(records, tmp_path).records] == ["x"]
    assert len(MultispectralPersonDataset(records, tmp_path, target_modality="ir")) == 2


def test_drop_misaligned_skips_only_checked_failures(tmp_path):
    _write_pair(tmp_path)
    records = [
        dataclasses.replace(_record(), id="ok", align_ok=True),
        dataclasses.replace(_record(), id="bad", align_ok=False),
        dataclasses.replace(_record(), id="unchecked"),
    ]

    kept = MultispectralPersonDataset(records, tmp_path, drop_misaligned=True).records

    assert [r.id for r in kept] == ["ok", "unchecked"]
    assert len(MultispectralPersonDataset(records, tmp_path)) == 3


# IR covers the middle half of the RGB frame: x' = 0.5x + 0.25S, y' = 0.5y + 0.25S (S = 640 space).
HALF = [[0.5, 0.0, 160.0], [0.0, 0.5, 160.0], [0.0, 0.0, 1.0]]


def _write_aligned_pair(root):
    rgb = np.zeros((80, 80, 3), np.uint8)
    rgb[20:60, 20:60] = 255  # the part of the scene IR sees
    Image.fromarray(rgb).save(root / "rgb.png")
    Image.new("L", (40, 40), color=200).save(root / "ir.png")
    # inside the IR footprint, half outside (clipped), fully outside (dropped)
    (root / "rgb.txt").write_text("0 0.5 0.5 0.1 0.1\n0 0.25 0.5 0.1 0.1\n0 0.1 0.1 0.05 0.05\n")
    (root / "ir.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return dataclasses.replace(_record(), rgb_image="rgb.png", ir_image="ir.png", ir_transform=HALF)


def test_ir_alignment_warp_keeps_rgb_grid_and_places_ir(tmp_path):
    record = _write_aligned_pair(tmp_path)
    item = MultispectralPersonDataset([record], tmp_path, img_size=(64, 64), ir_alignment="warp")[0]

    ir = item["ir"][0]
    assert ir[32, 32] == pytest.approx(200 / 255, abs=0.02)  # inside the footprint
    assert ir[4, 4] == 0 and ir[60, 60] == 0  # outside it
    assert len(item["targets"]["boxes"]) == 3  # nothing cropped


def test_ir_alignment_crop_fills_frame_and_clips_boxes(tmp_path):
    record = _write_aligned_pair(tmp_path)
    item = MultispectralPersonDataset([record], tmp_path, img_size=(64, 64), ir_alignment="crop")[0]

    assert item["ir"][0].min() > 0.7  # IR everywhere
    assert item["rgb"].mean() > 0.95  # the bright RGB centre now fills the frame
    boxes = item["targets"]["boxes"]
    torch.testing.assert_close(boxes[0], torch.tensor([0.5, 0.5, 0.2, 0.2]))  # 2x larger, centred
    assert boxes[1, 0] - boxes[1, 2] / 2 == pytest.approx(0.0, abs=1e-6)  # clipped at the left edge
    assert len(boxes) == 2  # the box outside the field of view is dropped


def test_ir_alignment_moves_ir_target_boxes_into_rgb_grid(tmp_path):
    record = _write_aligned_pair(tmp_path)
    item = MultispectralPersonDataset([record], tmp_path, img_size=(64, 64), target_modality="ir", ir_alignment="warp")[0]

    torch.testing.assert_close(item["targets"]["boxes"], torch.tensor([[0.5, 0.5, 0.1, 0.1]]))


def test_ir_alignment_falls_back_without_transform(tmp_path):
    _write_pair(tmp_path)
    item = MultispectralPersonDataset([_record()], tmp_path, img_size=(96, 128), ir_alignment="crop")[0]

    assert item["ir"].shape == (1, 96, 128)
    torch.testing.assert_close(item["targets"]["boxes"], torch.tensor([[0.5, 0.5, 0.2, 0.3]]))
