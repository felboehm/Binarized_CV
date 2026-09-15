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
