import torch

from binarized_cv.models.registry import MODEL_REGISTRY, build_model


def _images(batch: int = 2, size: int = 64) -> dict[str, torch.Tensor]:
    return {"rgb": torch.randn(batch, 3, size, size)}


def _build():
    return build_model("yolo26", num_classes=1, img_size=(64, 64), scale="n", pretrained=False)


def test_registered_under_yolo26_name():
    build_model("yolo26", num_classes=1)  # triggers registration side effect
    assert "yolo26" in MODEL_REGISTRY


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
    model.eval()
    images = _images(batch=1)

    with torch.no_grad():
        detections = model(images)

    assert len(detections) == 1
    det = detections[0]
    assert set(det.keys()) == {"boxes", "scores", "labels"}
    assert det["boxes"].shape[-1] == 4
    assert (det["scores"] > model.conf_thresh).all()


def test_eval_boxes_are_clamped_to_image_bounds():
    model = _build()
    model.conf_thresh = -1.0  # keep every candidate box, including untrained garbage predictions
    model.eval()
    images = _images(batch=1)

    with torch.no_grad():
        detections = model(images)

    boxes = detections[0]["boxes"]
    assert (boxes[:, [0, 2]] >= 0).all() and (boxes[:, [0, 2]] <= 64).all()
    assert (boxes[:, [1, 3]] >= 0).all() and (boxes[:, [1, 3]] <= 64).all()
