import torch

from binarized_cv.models.heads.anchor_free import AnchorFreeHead


def _targets():
    return [
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]), "labels": torch.tensor([0])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)},
    ]


def test_compute_loss_is_finite_and_backprops():
    head = AnchorFreeHead(in_channels=8, num_classes=1)
    features = torch.randn(2, 8, 4, 4, requires_grad=True)

    raw = head(features)
    losses = head.compute_loss(raw, _targets())
    total = sum(losses.values())
    total.backward()

    assert set(losses) == {"loss_coord", "loss_obj", "loss_cls"}
    assert all(torch.isfinite(v) for v in losses.values())
    assert features.grad is not None


def test_compute_loss_handles_batch_with_no_positives():
    head = AnchorFreeHead(in_channels=8, num_classes=1)
    raw = head(torch.randn(1, 8, 4, 4))
    empty_targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}]

    losses = head.compute_loss(raw, empty_targets)

    assert all(torch.isfinite(v) for v in losses.values())


def test_postprocess_returns_expected_format():
    head = AnchorFreeHead(in_channels=8, num_classes=1, conf_thresh=-1.0)
    raw = head(torch.randn(1, 8, 4, 4))

    detections = head.postprocess(raw, img_size=(128, 128))

    assert len(detections) == 1
    det = detections[0]
    assert set(det.keys()) == {"boxes", "scores", "labels"}
    assert det["boxes"].shape[0] > 0
    assert det["boxes"].shape[1] == 4
