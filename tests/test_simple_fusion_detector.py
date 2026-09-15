import torch

from binarized_cv.models.registry import build_model


def _images(batch: int = 2, size: int = 32) -> dict[str, torch.Tensor]:
    return {"rgb": torch.randn(batch, 3, size, size), "ir": torch.randn(batch, 1, size, size)}


def _build():
    return build_model(
        "simple_fusion",
        num_classes=1,
        img_size=(32, 32),
        backbone_widths=(4, 8),
        fusion_channels=8,
    )


def test_train_mode_returns_loss_dict_and_backprops():
    model = _build()
    images = _images()
    targets = [
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]), "labels": torch.tensor([0])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)},
    ]

    losses = model(images, targets)
    total = sum(losses.values())
    total.backward()

    assert set(losses) == {"loss_coord", "loss_obj", "loss_cls"}
    assert torch.isfinite(total)


def test_eval_mode_returns_detections():
    model = _build()
    model.eval()
    images = _images(batch=1)

    with torch.no_grad():
        detections = model(images)

    assert len(detections) == 1
    assert set(detections[0].keys()) == {"boxes", "scores", "labels"}
