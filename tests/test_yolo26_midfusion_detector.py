from __future__ import annotations

import pytest
import torch

from binarized_cv.models.registry import build_model


@pytest.fixture
def model():
    """Create a YOLO26 mid-fusion detector for testing."""
    return build_model("yolo26_midfusion", num_classes=1, img_size=(640, 640), scale="n")


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


def _one2one_scores(model, rgb, ir):
    preds = model._fused_forward(rgb, ir)
    preds = preds[1] if isinstance(preds, tuple) else preds
    return preds["one2one"]["scores"]


@pytest.mark.parametrize("fusion_layers", [(6,), (4, 6)])
def test_ir_input_changes_predictions(fusion_layers):
    """IR actually reaches the head. In train mode: at init, eval-mode
    BatchNorm (running stats 0/1) shrinks activations so far that neither
    input changes the output."""
    torch.manual_seed(0)
    model = build_model("yolo26_midfusion", img_size=(128, 128), fusion_layers=fusion_layers).train()
    rgb = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        a = _one2one_scores(model, rgb, torch.randn(2, 1, 128, 128))
        b = _one2one_scores(model, rgb, torch.randn(2, 1, 128, 128))
    assert not torch.allclose(a, b)


def test_ir_stream_stops_at_last_fusion_layer():
    model = build_model("yolo26_midfusion", fusion_layers=[4])
    assert len(model.ir_stream.model) == 5
    assert set(model.fusion) == {"4"}


@pytest.mark.parametrize("fusion_layers", [[], [11], [-1]])
def test_rejects_fusion_outside_backbone(fusion_layers):
    with pytest.raises(ValueError, match="fusion_layers"):
        build_model("yolo26_midfusion", fusion_layers=fusion_layers)


def test_binarization_covers_ir_stream_but_not_fusion():
    from binarized_cv.models.binarized import QuantConv2d

    model = build_model("yolo26_midfusion", binarization={"preset": "ib_guided"})
    # IR layer 6 is backbone_deep -> binary, like RGB layer 6.
    assert any(isinstance(m, QuantConv2d) and m.w_bits == 1 for m in model.ir_stream.model[6].modules())
    assert not any(isinstance(m, QuantConv2d) for m in model.fusion.modules())
