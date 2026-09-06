from binarized_cv.data.datasets.wisard import discover_wisard


def _make_frame(dir_path, dirname, frame_index):
    dir_path.mkdir(parents=True, exist_ok=True)
    stem = f"{dirname}_{frame_index}"
    (dir_path / f"{stem}.jpeg").write_bytes(b"")
    (dir_path / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def test_pairs_frames_across_differently_named_dirs(tmp_path):
    raw_root = tmp_path
    sample_root = raw_root / "wisard" / "WiSARD_Multi_Modal_Sample"
    vis_dirname = "210417_MtErie_Enterprise_VIS_0003"
    ir_dirname = "210417_MtErie_Enterprise_IR_0004"
    vis_dir = sample_root / vis_dirname
    ir_dir = sample_root / ir_dirname

    _make_frame(vis_dir, vis_dirname, "00000109")
    _make_frame(ir_dir, ir_dirname, "00000109")

    records = discover_wisard(raw_root)

    assert len(records) == 1
    record = records[0]
    assert record.id == "wisard_210417_MtErie_Enterprise_00000109"
    assert record.split == "unassigned"
    assert record.rgb_image == (
        "wisard/WiSARD_Multi_Modal_Sample/210417_MtErie_Enterprise_VIS_0003/"
        "210417_MtErie_Enterprise_VIS_0003_00000109.jpeg"
    )
    assert record.ir_image == (
        "wisard/WiSARD_Multi_Modal_Sample/210417_MtErie_Enterprise_IR_0004/"
        "210417_MtErie_Enterprise_IR_0004_00000109.jpeg"
    )


def test_skips_frames_present_in_only_one_modality(tmp_path):
    raw_root = tmp_path
    sample_root = raw_root / "wisard" / "WiSARD_Multi_Modal_Sample"
    vis_dirname = "210417_MtErie_Enterprise_VIS_0003"
    ir_dirname = "210417_MtErie_Enterprise_IR_0004"
    vis_dir = sample_root / vis_dirname
    ir_dir = sample_root / ir_dirname

    _make_frame(vis_dir, vis_dirname, "00000001")
    _make_frame(vis_dir, vis_dirname, "00000002")
    _make_frame(ir_dir, ir_dirname, "00000001")

    records = discover_wisard(raw_root)

    assert len(records) == 1
    assert records[0].id == "wisard_210417_MtErie_Enterprise_00000001"


def test_ignores_unmatched_flight_prefixes(tmp_path):
    raw_root = tmp_path
    sample_root = raw_root / "wisard" / "WiSARD_Multi_Modal_Sample"
    vis_dirname = "orphan_flight_VIS_0001"
    vis_dir = sample_root / vis_dirname

    _make_frame(vis_dir, vis_dirname, "00000001")

    records = discover_wisard(raw_root)

    assert records == []
