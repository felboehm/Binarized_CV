from binarized_cv.data.manifest import build_manifest, load_manifest, save_manifest


def _make_trgb_pair(raw_root, image_id):
    dataset_root = raw_root / "trgb" / "trgb_dataset" / "train"
    for modality_dir in ("RGB_images_train", "IR_images_train"):
        d = dataset_root / modality_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{image_id}.jpg").write_bytes(b"")
        (d / f"{image_id}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def _make_wisard_pair(raw_root, frame_index):
    sample_root = raw_root / "wisard" / "WiSARD_Multi_Modal_Sample"
    vis_dirname = "flightA_VIS_0001"
    ir_dirname = "flightA_IR_0002"
    for dirname in (vis_dirname, ir_dirname):
        d = sample_root / dirname
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{dirname}_{frame_index}"
        (d / f"{stem}.jpeg").write_bytes(b"")
        (d / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def test_build_manifest_combines_both_datasets(tmp_path):
    _make_trgb_pair(tmp_path, "1001")
    _make_wisard_pair(tmp_path, "00000001")

    records = build_manifest(tmp_path)

    assert {r.dataset for r in records} == {"trgb", "wisard"}
    assert len(records) == 2


def test_save_and_load_manifest_roundtrip(tmp_path):
    _make_trgb_pair(tmp_path, "1001")
    _make_wisard_pair(tmp_path, "00000001")
    records = build_manifest(tmp_path)
    manifest_path = tmp_path / "processed" / "manifest.jsonl"

    save_manifest(records, manifest_path)
    loaded = load_manifest(manifest_path)

    assert loaded == records
