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


def _write_marked_pair(root):
    """64x64 pair, black except a bright square at x 8..15, y 24..39 in both
    modalities, with one box exactly on it."""
    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[24:40, 8:16] = 255
    ir = np.zeros((64, 64), np.uint8)
    ir[24:40, 8:16] = 200
    Image.fromarray(rgb).save(root / "rgb.png")
    Image.fromarray(ir).save(root / "ir.png")
    (root / "rgb.txt").write_text("0 0.1875 0.5 0.125 0.25\n")
    (root / "ir.txt").write_text("0 0.1875 0.5 0.125 0.25\n")
    return dataclasses.replace(_record(), rgb_image="rgb.png", ir_image="ir.png")


def test_augment_flip_mirrors_both_modalities_and_boxes(tmp_path):
    record = _write_marked_pair(tmp_path)
    plain = MultispectralPersonDataset([record], tmp_path, img_size=(64, 64))[0]
    item = MultispectralPersonDataset(
        [record], tmp_path, img_size=(64, 64), augment={"fliplr": 1.0, "scale": 0.0, "translate": 0.0}
    )[0]

    torch.testing.assert_close(item["rgb"], plain["rgb"].flip(-1))
    torch.testing.assert_close(item["ir"], plain["ir"].flip(-1))
    torch.testing.assert_close(item["targets"]["boxes"], torch.tensor([[1 - 0.1875, 0.5, 0.125, 0.25]]))


def test_augment_keeps_boxes_on_the_content(tmp_path):
    record = _write_marked_pair(tmp_path)
    dataset = MultispectralPersonDataset(
        [record], tmp_path, img_size=(64, 64), augment={"fliplr": 0.5, "scale": 0.25, "translate": 0.1}
    )
    torch.manual_seed(0)
    for _ in range(20):
        item = dataset[0]
        boxes = item["targets"]["boxes"]
        assert len(boxes) == len(item["targets"]["labels"])
        if len(boxes) == 0:
            continue
        cx, cy, w, h = (boxes[0] * 64).tolist()
        inside = item["ir"][0, round(cy - h / 2) + 1 : round(cy + h / 2) - 1, round(cx - w / 2) + 1 : round(cx + w / 2) - 1]
        assert inside.min() > 0.5  # the box still sits on the bright square
        assert item["rgb"].mean() == pytest.approx(item["ir"].mean() * 255 / 200, abs=0.02)  # same transform


def test_augment_drops_boxes_pushed_out_of_frame(tmp_path):
    record = _write_marked_pair(tmp_path)
    # The square (x 8..16) is under half visible once shifted left by more
    # than 12 px; shifts of up to 16 px make that 1 in 8 draws.
    dataset = MultispectralPersonDataset([record], tmp_path, img_size=(64, 64), augment={"translate": 0.25})
    torch.manual_seed(0)
    counts = {len(dataset[0]["targets"]["boxes"]) for _ in range(100)}
    assert counts == {0, 1}
