import torch

from binarized_cv.models.fusion.concat import ConcatFusion


def test_concat_fusion_projects_channels():
    fusion = ConcatFusion(in_channels_per_modality=[8, 4], out_channels=16)
    features = [torch.randn(2, 8, 5, 5), torch.randn(2, 4, 5, 5)]

    out = fusion(features)

    assert out.shape == (2, 16, 5, 5)
