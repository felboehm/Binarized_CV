import torch
from torch import nn

from binarized_cv.models.registry import MODEL_REGISTRY, build_model


def _images(batch: int = 2, size: int = 64) -> dict[str, torch.Tensor]:
    return {"rgb": torch.randn(batch, 3, size, size), "ir": torch.randn(batch, 1, size, size)}


def _build():
    return build_model("ms_yolov8", num_classes=1, img_size=(64, 64), scale="n")


def test_registered_under_ms_yolov8_name():
    build_model("ms_yolov8", num_classes=1)  # triggers registration side effect
    assert "ms_yolov8" in MODEL_REGISTRY


def test_first_conv_takes_4_channel_fused_input():
    model = _build()
    assert model.model.model[0].conv.in_channels == 4


def test_upsample_layers_are_patched_to_bicubic():
    model = _build()
    upsamples = [m for m in model.model.modules() if isinstance(m, nn.Upsample)]
    assert upsamples, "expected at least one nn.Upsample in the neck"
    assert all(m.mode == "bicubic" for m in upsamples)


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

    assert set(losses) == {"loss_box", "loss_cls", "loss_dfl"}
    assert torch.isfinite(total)


def test_eval_mode_returns_detections_in_expected_format():
    model = _build()
    model.conf_thresh = 0.001  # keep low-confidence untrained predictions too
    model.eval()
    images = _images(batch=1)

    with torch.no_grad():
        detections = model(images)

    assert len(detections) == 1
    det = detections[0]
    assert set(det.keys()) == {"boxes", "scores", "labels"}
    assert det["boxes"].shape[-1] == 4


def test_eval_boxes_are_clamped_to_image_bounds():
    model = _build()
    model.conf_thresh = 0.001
    model.eval()
    images = _images(batch=1)

    with torch.no_grad():
        detections = model(images)

    boxes = detections[0]["boxes"]
    assert (boxes[:, [0, 2]] >= 0).all() and (boxes[:, [0, 2]] <= 64).all()
    assert (boxes[:, [1, 3]] >= 0).all() and (boxes[:, [1, 3]] <= 64).all()
