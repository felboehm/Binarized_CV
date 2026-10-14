import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

from binarized_cv.models.detectors import _ultralytics_common
from binarized_cv.models.registry import build_model


@pytest.fixture
def fake_coco(tmp_path, monkeypatch):
    """A stand-in for ultralytics' yolo26n.pt (80 classes), so no download."""
    torch.manual_seed(0)
    source = DetectionModel(cfg="yolo26n.yaml", ch=3, nc=80, verbose=False)
    path = tmp_path / "yolo26n.pt"
    torch.save({"model": source}, path)
    monkeypatch.setattr("ultralytics.utils.downloads.attempt_download_asset", lambda name: str(path))
    return source.state_dict()


def test_early_fusion_gets_rgb_weights_and_keeps_ir_channel(fake_coco):
    model = build_model("yolo26_early_fusion", pretrained=True)
    first = model.model.model[0].conv.weight
    assert torch.equal(first[:, :3], fake_coco["model.0.conv.weight"])
    assert torch.equal(model.model.model[6].cv1.conv.weight, fake_coco["model.6.cv1.conv.weight"])


def test_midfusion_ir_stream_first_conv_is_rgb_sum(fake_coco):
    model = build_model("yolo26_midfusion", pretrained=True)
    expected = fake_coco["model.0.conv.weight"].sum(dim=1, keepdim=True)
    assert torch.allclose(model.ir_stream.model[0].conv.weight, expected)
    assert torch.equal(model.ir_stream.model[6].cv1.conv.weight, fake_coco["model.6.cv1.conv.weight"])


def test_raises_when_checkpoint_does_not_fit(tmp_path, monkeypatch):
    path = tmp_path / "other.pt"
    torch.save({"model": torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3))}, path)
    monkeypatch.setattr("ultralytics.utils.downloads.attempt_download_asset", lambda name: str(path))
    with pytest.raises(RuntimeError, match="tensors match"):
        _ultralytics_common.load_pretrained(DetectionModel(cfg="yolo26n.yaml", ch=3, nc=1, verbose=False), "n")
