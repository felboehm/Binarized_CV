import pytest
import torch
from torch import nn

from binarized_cv.models.binarized import QuantConv2d
from binarized_cv.models.binarized.functional import binarize, ede_sharpness


def test_binarize_outputs_only_plus_minus_one_including_zero():
    x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 3.0])
    assert binarize(x).tolist() == [-1.0, -1.0, 1.0, 1.0, 1.0]


def test_binarize_surrogate_gradient_sharpens_with_t():
    x = torch.tensor([0.0, 2.0], requires_grad=True)
    binarize(x, t=0.1).sum().backward()
    soft = x.grad.clone()
    x.grad = None
    binarize(x, t=10.0).sum().backward()
    sharp = x.grad
    # Low t: ~identity STE everywhere. High t: large at 0, ~0 away from it.
    assert soft[0] == pytest.approx(1.0) and soft[1] > 0.9
    assert sharp[0] == pytest.approx(10.0) and sharp[1] < 1e-6


def test_stochastic_binarize_is_binary_and_tracks_sign_on_average():
    torch.manual_seed(0)
    x = torch.full((10000,), 0.5)
    out = binarize(x, t=1.0, stochastic=True)
    assert set(out.unique().tolist()) <= {-1.0, 1.0}
    expected_mean = torch.tanh(torch.tensor(0.5)).item()
    assert out.mean().item() == pytest.approx(expected_mean, abs=0.03)


def test_ede_sharpness_schedule_endpoints():
    assert ede_sharpness(0.0) == pytest.approx(0.1)
    assert ede_sharpness(1.0) == pytest.approx(10.0)
    assert ede_sharpness(2.0) == pytest.approx(10.0)


def _conv():
    torch.manual_seed(0)
    return nn.Conv2d(8, 16, 3, padding=1, bias=False)


def test_from_conv_copies_weights_and_keeps_state_dict_keys():
    conv = _conv()
    q = QuantConv2d.from_conv(conv, w_bits=1, a_bits=1)
    assert torch.equal(q.weight, conv.weight)
    assert set(conv.state_dict()) <= set(q.state_dict())


def test_full_precision_quant_conv_matches_conv():
    conv = _conv()
    q = QuantConv2d.from_conv(conv, w_bits=None, a_bits=None)
    x = torch.randn(2, 8, 10, 10)
    assert torch.allclose(q(x), conv(x))


def test_binary_weights_have_two_levels_per_output_channel():
    q = QuantConv2d.from_conv(_conv(), w_bits=1, a_bits=None)
    w = q.quantized_weight()
    for channel in w:
        assert channel.unique().numel() == 2
        assert channel.abs().unique().numel() == 1  # single real scale alpha


def test_binary_activation_threshold_initialised_from_first_batch():
    q = QuantConv2d.from_conv(_conv(), w_bits=1, a_bits=1)
    x = torch.rand(4, 8, 6, 6) + 0.5  # all positive, like post-SiLU inputs
    assert not q.initialized
    xq = q.quantized_input(x)
    assert q.initialized
    assert torch.allclose(q.act_threshold.flatten(), x.mean(dim=(0, 2, 3)))
    # A zero threshold would map everything to +1; the data-dependent one doesn't.
    assert set(xq.unique().tolist()) == {-1.0, 1.0}


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_kbit_quantization_level_counts(bits):
    q = QuantConv2d.from_conv(_conv(), w_bits=bits, a_bits=bits)
    w = q.quantized_weight()
    assert w[0].unique().numel() <= 2 ** bits - 1
    xq = q.quantized_input(torch.randn(2, 8, 16, 16))
    assert xq.unique().numel() <= 2**bits


def test_kbit_activation_range_frozen_in_eval():
    q = QuantConv2d.from_conv(_conv(), w_bits=8, a_bits=8)
    q.train()
    q(torch.randn(2, 8, 6, 6))
    frozen = q.act_range.clone()
    q.eval()
    q(torch.randn(2, 8, 6, 6) * 100)
    assert torch.equal(q.act_range, frozen)


@pytest.mark.parametrize("bits", [1, 4])
def test_gradients_reach_latent_weights_and_inputs(bits):
    q = QuantConv2d.from_conv(_conv(), w_bits=bits, a_bits=bits)
    x = torch.randn(2, 8, 6, 6, requires_grad=True)
    q(x).pow(2).sum().backward()
    assert q.weight.grad is not None and q.weight.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0
    if bits == 1:
        assert q.act_threshold.grad.abs().sum() > 0


def test_invalid_bits_rejected():
    with pytest.raises(ValueError):
        QuantConv2d(4, 4, 3, w_bits=0, a_bits=1)
