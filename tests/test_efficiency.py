import pytest
import torch
from torch import nn

from binarized_cv.eval.efficiency import (
    OpCounts,
    RaplMeter,
    count_ops,
    deployed_size,
    dummy_images,
    prepare_for_inference,
)
from binarized_cv.models.binarized import QuantConv2d
from binarized_cv.models.registry import build_model


class TwoConvs(nn.Module):
    modalities = ("rgb",)

    def __init__(self) -> None:
        super().__init__()
        self.fp = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.binary = QuantConv2d(8, 16, 3, padding=1, bias=False, w_bits=1, a_bits=1)

    def forward(self, images):
        return self.binary(self.fp(images["rgb"]))


def test_count_ops_per_precision_and_per_image():
    model = TwoConvs().eval()
    counts = count_ops(model, {"rgb": torch.rand(2, 3, 10, 10)})

    assert counts.macs == {"w32a32": 100 * 8 * 3 * 9, "w1a1": 100 * 16 * 8 * 9}
    assert counts.fp_macs == 100 * 8 * 3 * 9
    assert counts.bops == 100 * 16 * 8 * 9
    assert counts.ops == pytest.approx(counts.fp_macs + counts.bops / 64)


def test_kbit_bops_scale_with_both_bit_widths():
    counts = OpCounts({"w8a8": 10, "w4a2": 10, "attn_matmul": 5})

    assert counts.bops == 10 * 64 + 10 * 8
    assert counts.fp_macs == 5
    assert counts.ops == pytest.approx(5 + 10 + 10 * 8 / 64)


def test_deployed_size_packs_quantized_weights_only():
    model = TwoConvs()
    size = deployed_size(model)

    fp_bits = 8 * 3 * 9 * 32
    binary_bits = 16 * 8 * 9 * 1 + 16 * 32  # packed signs + per-channel scale
    threshold_bits = 8 * 32 + 2 * 32  # act_threshold + act_range buffer
    assert size["packed_bytes"] == (fp_bits + binary_bits + threshold_bits) / 8
    assert size["fp32_state_dict_bytes"] > size["packed_bytes"]


def test_prepare_for_inference_keeps_quantized_convs_and_drops_one2many():
    model = build_model("yolo26_early_fusion", binarization={"preset": "ib_guided"})
    quantized_before = sum(isinstance(m, QuantConv2d) for m in model.modules())
    images = dummy_images(model, 1, (64, 64), torch.device("cpu"))
    with torch.no_grad():
        reference = model.eval()(images)[0]["scores"]

    prepare_for_inference(model)

    detect = model.model.model[-1]
    assert detect.cv2 is None and detect.cv3 is None
    assert sum(isinstance(m, QuantConv2d) for m in model.modules()) == quantized_before
    assert not any(hasattr(m, "bn") and not isinstance(m.conv, QuantConv2d) for m in model.modules() if hasattr(m, "conv"))
    with torch.no_grad():
        torch.testing.assert_close(model(images)[0]["scores"], reference, rtol=1e-4, atol=1e-4)


def test_rapl_delta_handles_counter_wraparound(tmp_path):
    (tmp_path / "name").write_text("package-0\n")
    (tmp_path / "max_energy_range_uj").write_text("1000000\n")
    meter = RaplMeter(tmp_path)

    assert meter.delta(0.9, 0.1) == pytest.approx(0.2)
    assert meter.delta(0.1, 0.4) == pytest.approx(0.3)
