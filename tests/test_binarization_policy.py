import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

from binarized_cv.models.binarized import (
    BinarizationConfig,
    QuantConv2d,
    apply_binarization,
    format_region_stats,
    optimizer_param_groups,
    set_binarization_progress,
)
from binarized_cv.models.registry import build_model


@pytest.fixture(scope="module")
def fresh_yolo26():
    return lambda: DetectionModel(cfg="yolo26n.yaml", ch=3, nc=1, verbose=False)


def _conv_bits(model, name):
    module = model.get_submodule(name)
    return (module.w_bits, module.a_bits) if isinstance(module, QuantConv2d) else (None, None)


def test_ib_guided_region_map(fresh_yolo26):
    model = fresh_yolo26()
    stats = apply_binarization(model, {"preset": "ib_guided"})

    assert _conv_bits(model, "model.0.conv") == (None, None)  # input conv
    assert _conv_bits(model, "model.1.conv") == (None, None)  # stem
    assert _conv_bits(model, "model.4.cv1.conv") == (8, 8)  # backbone P3
    assert _conv_bits(model, "model.16.cv1.conv") == (8, 8)  # neck P3
    assert _conv_bits(model, "model.6.cv1.conv") == (1, 1)  # deep backbone
    assert _conv_bits(model, "model.9.cv1.conv") == (1, 1)  # SPPF
    assert _conv_bits(model, "model.13.cv1.conv") == (1, 1)  # neck P4
    assert _conv_bits(model, "model.10.cv1.conv") == (None, None)  # C2PSA
    # Neck attention inside a binary-region C3k2 still stays fp.
    assert _conv_bits(model, "model.22.m.0.1.attn.qkv.conv") == (None, None)
    assert _conv_bits(model, "model.22.cv1.conv") == (1, 1)
    assert _conv_bits(model, "model.23.one2one_cv2.0.0.conv") == (8, 8)
    assert _conv_bits(model, "model.23.one2one_cv2.0.2") == (8, 8)  # head_out
    assert _conv_bits(model, "model.23.cv2.0.0.conv") == (None, None)  # one2many

    assert stats["backbone_deep"].params > stats["stem"].params
    assert "deployed conv weights" in format_region_stats(stats)


def test_full_preset_spares_depthwise_and_final_projections(fresh_yolo26):
    model = fresh_yolo26()
    apply_binarization(model, {"preset": "full"})
    assert _conv_bits(model, "model.0.conv") == (None, None)
    assert _conv_bits(model, "model.1.conv") == (1, 1)
    assert _conv_bits(model, "model.23.one2one_cv3.0.0.0.conv") == (8, 8)  # depthwise fallback
    assert _conv_bits(model, "model.23.one2one_cv3.0.0.1.conv") == (1, 1)
    assert _conv_bits(model, "model.23.one2one_cv3.0.2") == (None, None)


def test_region_override_for_single_region_sweep(fresh_yolo26):
    model = fresh_yolo26()
    apply_binarization(model, {"preset": "fp32", "regions": {"p3": "binary", "stem": None}})
    quantized = {n for n, m in model.named_modules() if isinstance(m, QuantConv2d)}
    assert quantized and all(n.split(".")[1] in {"3", "4", "16", "17"} for n in quantized)


def test_every_conv_gets_a_region(fresh_yolo26):
    # Would raise on any unmapped layer, e.g. if the YOLO26 yaml changed.
    apply_binarization(fresh_yolo26(), {"preset": "fp32"})


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        BinarizationConfig.from_dict({"preset": "nope"}).region_precision()
    with pytest.raises(ValueError):
        BinarizationConfig.from_dict({"regions": {"not_a_region": "binary"}}).region_precision()
    with pytest.raises(ValueError):
        BinarizationConfig.from_dict({"typo_option": True})


def test_no_weight_decay_on_binary_latent_weights(fresh_yolo26):
    model = fresh_yolo26()
    apply_binarization(model, {"preset": "ib_guided"})
    decay, no_decay = optimizer_param_groups(model, 1e-4)
    binary_weight = model.get_submodule("model.6.cv1.conv").weight
    assert any(p is binary_weight for p in no_decay["params"])
    assert not any(p is binary_weight for p in decay["params"])
    assert no_decay["weight_decay"] == 0.0


def test_set_progress_updates_all_binary_layers(fresh_yolo26):
    model = fresh_yolo26()
    apply_binarization(model, {"preset": "full"})
    set_binarization_progress(model, 1.0)
    assert all(m.sharpness == pytest.approx(10.0) for m in model.modules() if isinstance(m, QuantConv2d))


@pytest.fixture
def batch():
    return {
        "images": {"rgb": torch.rand(2, 3, 128, 128), "ir": torch.rand(2, 1, 128, 128)},
        "targets": [
            {"boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]), "labels": torch.zeros(1)},
            {"boxes": torch.tensor([[0.2, 0.3, 0.1, 0.2]]), "labels": torch.zeros(1)},
        ],
    }


@pytest.mark.parametrize("name", ["yolo26", "yolo26_early_fusion"])
@pytest.mark.parametrize("preset", ["ib_guided", "full"])
def test_binarized_detector_trains_and_infers(name, preset, batch):
    model = build_model(name, img_size=(128, 128), binarization={"preset": preset})
    model.train()
    losses = model(batch["images"], batch["targets"])
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    binary = next(m for m in model.modules() if isinstance(m, QuantConv2d) and m.w_bits == 1)
    assert binary.weight.grad is not None and binary.weight.grad.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        detections = model(batch["images"])
    assert len(detections) == 2 and detections[0]["boxes"].shape[-1] == 4


def test_warm_start_from_fp32_checkpoint_then_round_trip(tmp_path, batch):
    fp32 = build_model("yolo26", img_size=(128, 128))
    path = tmp_path / "fp32.pt"
    torch.save(fp32.state_dict(), path)

    bnn = build_model("yolo26", img_size=(128, 128), binarization={"preset": "ib_guided"}, init_checkpoint=str(path))
    assert torch.equal(bnn.model.model[6].cv1.conv.weight, fp32.model.model[6].cv1.conv.weight)

    bnn.eval()
    with torch.no_grad():
        bnn(batch["images"])  # initialises activation thresholds/ranges
    bnn_path = tmp_path / "bnn.pt"
    torch.save(bnn.state_dict(), bnn_path)
    reloaded = build_model("yolo26", img_size=(128, 128), binarization={"preset": "ib_guided"})
    reloaded.load_state_dict(torch.load(bnn_path))
    assert reloaded.model.model[6].cv1.conv.initialized
