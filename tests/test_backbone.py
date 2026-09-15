import torch

from binarized_cv.models.backbones.simple_cnn import SimpleCNNBackbone


def test_output_shape_matches_stride():
    backbone = SimpleCNNBackbone(in_channels=3, widths=(4, 8))
    x = torch.randn(2, 3, 64, 64)

    out = backbone(x)

    assert out.shape == (2, 8, 16, 16)
    assert backbone.out_channels == 8
    assert backbone.stride == 4


def test_single_channel_input_for_ir():
    backbone = SimpleCNNBackbone(in_channels=1, widths=(4,))
    x = torch.randn(1, 1, 32, 32)

    out = backbone(x)

    assert out.shape == (1, 4, 16, 16)
