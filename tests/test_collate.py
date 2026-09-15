import torch

from binarized_cv.data.collate import detection_collate


def _item(id_: str, n_boxes: int) -> dict:
    return {
        "id": id_,
        "dataset": "test",
        "rgb": torch.zeros(3, 8, 8),
        "ir": torch.zeros(1, 8, 8),
        "targets": {"boxes": torch.zeros(n_boxes, 4), "labels": torch.zeros(n_boxes, dtype=torch.long)},
    }


def test_collate_stacks_images_and_keeps_variable_length_targets():
    batch = [_item("a", 2), _item("b", 0)]

    out = detection_collate(batch)

    assert out["images"]["rgb"].shape == (2, 3, 8, 8)
    assert out["images"]["ir"].shape == (2, 1, 8, 8)
    assert out["id"] == ["a", "b"]
    assert out["dataset"] == ["test", "test"]
    assert len(out["targets"]) == 2
    assert out["targets"][0]["boxes"].shape == (2, 4)
    assert out["targets"][1]["boxes"].shape == (0, 4)
