import numpy as np
import pytest

from binarized_cv.data.alignment import CAP, Frame, choose_offsets, fit_affine, frame_cost, segment_transforms, warp_boxes
from binarized_cv.data.records import PairRecord

RIG = np.array([[0.64, 0.0, 120.0], [0.0, 0.9, 37.0], [0.0, 0.0, 1.0]])
OTHER = np.array([[0.64, 0.0, 130.0], [0.0, 0.9, 80.0], [0.0, 0.0, 1.0]])  # e.g. another rig: +10, +43 px


def _frames(T: np.ndarray, n: int, seed: int = 0, start: int = 0) -> list[Frame]:
    """Synthetic frames: random IR person boxes and their exact VIS image under T."""
    rng = np.random.default_rng(seed)
    record = PairRecord(id="x", dataset="test", split="train", rgb_image="", rgb_label="", ir_image="", ir_label="")
    frames = []
    for i in range(n):
        xy = rng.uniform(50, 550, size=(3, 2))
        ir = np.c_[xy, xy + rng.uniform(15, 40, size=(3, 2))]
        frames.append(Frame(start + i, record, warp_boxes(T, ir), ir))
    return frames


def test_frame_cost_zero_under_true_transform_and_cap_when_unmatched():
    f = _frames(RIG, 1)[0]
    assert frame_cost(RIG, f.vis, f.ir) == pytest.approx(0, abs=1e-6)
    assert frame_cost(RIG, f.vis, f.ir[:0]) is None
    assert frame_cost(RIG, f.vis, f.ir[:1]) == pytest.approx(2 * CAP / 3)  # 2 of 3 VIS boxes unmatched


def test_frame_cost_ignores_vis_boxes_outside_ir_footprint():
    f = _frames(RIG, 1)[0]
    outside = np.array([[5.0, 5.0, 20.0, 40.0]])  # left of the IR footprint (x >= 120)
    assert frame_cost(RIG, np.r_[f.vis, outside], f.ir) == pytest.approx(0, abs=1e-6)


def test_fit_affine_recovers_transform_from_identity():
    T, n = fit_affine(_frames(RIG, 40), np.eye(3))
    assert n >= 100
    np.testing.assert_allclose(T, RIG, atol=1e-3)


def test_segment_transforms_finds_second_block():
    frames = _frames(RIG, 150) + _frames(OTHER, 150, seed=1, start=150)
    transforms, labels = segment_transforms(frames, [RIG])
    assert len(transforms) == 2
    np.testing.assert_allclose(transforms[1], OTHER, atol=1e-2)
    assert labels == [0] * 150 + [1] * 150


def test_choose_offsets_piecewise_constant_and_fills_unlabeled():
    good0, good1 = {-1: 30.0, 0: 2.0, 1: 30.0}, {-1: 30.0, 0: 30.0, 1: 2.0}
    blank = {-1: 0.0, 0: 0.0, 1: 0.0}
    costs = [good0] * 5 + [blank] * 3 + [good1] * 5
    ks = choose_offsets(costs, penalty=4.0)
    assert ks[:5] == [0] * 5 and ks[-5:] == [1] * 5
    assert len(set(ks[5:8])) == 1  # one switch only, inside the unlabeled stretch


def test_choose_offsets_ignores_single_noisy_frame_and_missing_k():
    costs = [{0: 2.0, 1: 10.0}] * 4 + [{0: 9.0, 1: 2.0}] + [{0: 2.0, 1: 10.0}] * 4
    assert choose_offsets(costs, penalty=8.0) == [0] * 9
    assert choose_offsets([{0: 5.0, 1: 1.0}, {0: 5.0}], penalty=0.0) == [1, 0]
