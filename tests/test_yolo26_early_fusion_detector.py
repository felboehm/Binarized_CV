from __future__ import annotations

import pytest
import torch

from binarized_cv.models.registry import build_model


@pytest.fixture
def model():
    """Create a YOLO26 early-fusion detector for testing."""
    return build_model("yolo26_early_fusion", num_classes=1, img_size=(640, 640), scale="n")


@pytest.fixture
def batch():
    """Create a synthetic batch of images and targets."""
    return {
        "images": {
            "rgb": torch.randn(2, 3, 640, 640),
            "ir": torch.randn(2, 1, 640, 640),
        },
        "targets": [
            {
                "boxes": torch.tensor([[0.25, 0.25, 0.75, 0.75], [0.1, 0.1, 0.3, 0.3]]),  # normalized cxcywh
                "labels": torch.zeros(2),
            },
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.8, 0.8]]),
                "labels": torch.zeros(1),
            },
        ],
    }


def test_forward_inference(model, batch):
    """Test inference mode: images → detections."""
    model.eval()
    with torch.no_grad():
        detections = model(batch["images"])

    assert isinstance(detections, list)
    assert len(detections) == 2
    for det in detections:
        assert "boxes" in det and "scores" in det and "labels" in det
        assert det["boxes"].shape[1] == 4  # xyxy
        assert det["scores"].dim() == 1
        assert det["labels"].dtype == torch.int64


def test_forward_training(model, batch):
    """Test training mode: images + targets → loss dict."""
    model.train()
    loss_dict = model(batch["images"], batch["targets"])

    assert isinstance(loss_dict, dict)
    assert "loss_box" in loss_dict and "loss_cls" in loss_dict and "loss_dfl" in loss_dict
    for key, val in loss_dict.items():
        assert isinstance(val, torch.Tensor)
        assert val.dim() == 0  # scalar
        assert not torch.isnan(val) and not torch.isinf(val)


def test_gradient_flow(model, batch):
    """Test that gradients flow through the model during backprop."""
    model.train()
    loss_dict = model(batch["images"], batch["targets"])
    loss = sum(loss_dict.values())

    loss.backward()

    # Check that at least some parameters have gradients
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break

    assert has_gradients, "No gradients found in model parameters"


def test_modalities(model):
    """Test that model declares both RGB and IR modalities."""
    assert model.modalities == ("rgb", "ir")


def test_different_batch_sizes(model):
    """Test that model works with different batch sizes."""
    model.eval()
    with torch.no_grad():
        for batch_size in [1, 4, 8]:
            images = {
                "rgb": torch.randn(batch_size, 3, 640, 640),
                "ir": torch.randn(batch_size, 1, 640, 640),
            }
            detections = model(images)
            assert len(detections) == batch_size


def test_ir_resize_to_rgb_shape(model):
    """Test that IR is resized to match RGB spatial dimensions."""
    model.eval()
    with torch.no_grad():
        # IR with different spatial dimensions
        images = {
            "rgb": torch.randn(2, 3, 640, 640),
            "ir": torch.randn(2, 1, 320, 320),  # Different size
        }
        detections = model(images)
        assert isinstance(detections, list)
        assert len(detections) == 2
