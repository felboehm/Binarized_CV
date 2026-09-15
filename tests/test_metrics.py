import pytest
import torch

from binarized_cv.eval.metrics import average_precision


def test_perfect_predictions_give_ap_one():
    predictions = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}
    ]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]

    assert average_precision(predictions, targets, iou_threshold=0.5) == pytest.approx(1.0)


def test_no_predictions_give_ap_zero():
    predictions = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)}]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]

    assert average_precision(predictions, targets, iou_threshold=0.5) == 0.0


def test_no_ground_truth_gives_ap_zero():
    predictions = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}
    ]
    targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}]

    assert average_precision(predictions, targets, iou_threshold=0.5) == 0.0


def test_low_iou_prediction_is_a_false_positive():
    predictions = [
        {"boxes": torch.tensor([[50.0, 50.0, 60.0, 60.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}
    ]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]

    assert average_precision(predictions, targets, iou_threshold=0.5) == pytest.approx(0.0)
