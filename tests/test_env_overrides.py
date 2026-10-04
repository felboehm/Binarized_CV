from binarized_cv.eval import evaluate
from binarized_cv.train import train


def test_env_overrides_apply_and_explicit_overrides_win(monkeypatch):
    monkeypatch.setenv("BCV_OVERRIDES", "data.raw_root=/scratch/raw data.num_workers=14")
    cfg, _ = train.load_config(["data.num_workers=3"])
    assert cfg.data.raw_root == "/scratch/raw"
    assert cfg.data.num_workers == 3
    assert evaluate.load_config(["data.num_workers=3"]).data.raw_root == "/scratch/raw"


def test_no_env_overrides(monkeypatch):
    monkeypatch.delenv("BCV_OVERRIDES", raising=False)
    cfg, _ = train.load_config([])
    assert cfg.data.raw_root == "data/raw"
