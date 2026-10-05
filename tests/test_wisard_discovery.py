from binarized_cv.data.datasets.wisard import discover_wisard


def _make_frame(dir_path, dirname, frame_index):
    dir_path.mkdir(parents=True, exist_ok=True)
    stem = f"{dirname}_{frame_index}"
    (dir_path / f"{stem}.jpeg").write_bytes(b"")
    (dir_path / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def test_pairs_frames_across_differently_named_dirs(tmp_path):
    raw_root = tmp_path
    sample_root = raw_root / "wisard"
    vis_dirname = "210417_MtErie_Enterprise_VIS_0003"
    ir_dirname = "210417_MtErie_Enterprise_IR_0004"
    vis_dir = sample_root / vis_dirname
    ir_dir = sample_root / ir_dirname

    _make_frame(vis_dir, vis_dirname, "00000109")
    _make_frame(ir_dir, ir_dirname, "00000109")

    records = discover_wisard(raw_root)

    assert len(records) == 1
    record = records[0]
    assert record.id == "wisard_210417_MtErie_Enterprise_VIS_0003_00000109"
    assert record.split == "val"  # MtErie is the val flight
    assert record.rgb_image == (
        "wisard/210417_MtErie_Enterprise_VIS_0003/"
        "210417_MtErie_Enterprise_VIS_0003_00000109.jpeg"
    )
    assert record.ir_image == (
        "wisard/210417_MtErie_Enterprise_IR_0004/"
        "210417_MtErie_Enterprise_IR_0004_00000109.jpeg"
    )


def test_skips_frames_present_in_only_one_modality(tmp_path):
    raw_root = tmp_path
    sample_root = raw_root / "wisard"
    vis_dirname = "210417_MtErie_Enterprise_VIS_0003"
    ir_dirname = "210417_MtErie_Enterprise_IR_0004"
    vis_dir = sample_root / vis_dirname
    ir_dir = sample_root / ir_dirname

    _make_frame(vis_dir, vis_dirname, "00000001")
    _make_frame(vis_dir, vis_dirname, "00000002")
    _make_frame(ir_dir, ir_dirname, "00000001")

    records = discover_wisard(raw_root)

    assert len(records) == 1
    assert records[0].id == "wisard_210417_MtErie_Enterprise_VIS_0003_00000001"


def test_ignores_unmatched_flight_prefixes(tmp_path):
    raw_root = tmp_path
    sample_root = raw_root / "wisard"
    vis_dirname = "orphan_flight_VIS_0001"
    vis_dir = sample_root / vis_dirname

    _make_frame(vis_dir, vis_dirname, "00000001")

    records = discover_wisard(raw_root)

    assert records == []


def test_splits_whole_flights(tmp_path):
    for prefix in ("210529_Carnation_Enterprise", "220109_Baker_Enterprise", "210924_FHL_Enterprise"):
        for modality in ("VIS_0001", "IR_0002"):
            dirname = f"{prefix}_{modality}"
            for frame in ("00000069", "00000070", "00000099"):  # train/val/test under the old frame-index split
                _make_frame(tmp_path / "wisard" / dirname, dirname, frame)

    splits = {}
    for record in discover_wisard(tmp_path):
        splits.setdefault(record.id.rsplit("_", 3)[0], set()).add(record.split)  # drop _VIS_<seq>_<frame>

    assert splits == {
        "wisard_210529_Carnation_Enterprise": {"zoom"},
        "wisard_220109_Baker_Enterprise": {"test"},
        "wisard_210924_FHL_Enterprise": {"train"},
    }


def test_pairs_frames_with_different_zero_padding(tmp_path):
    vis_dirname = "210417_MtErie_Enterprise_VIS_0005"
    ir_dirname = "210417_MtErie_Enterprise_IR_0006"
    _make_frame(tmp_path / "wisard" / vis_dirname, vis_dirname, "00000007")
    _make_frame(tmp_path / "wisard" / ir_dirname, ir_dirname, "00007")

    records = discover_wisard(tmp_path)

    assert len(records) == 1
    assert records[0].ir_image == f"wisard/{ir_dirname}/{ir_dirname}_00007.jpeg"


def test_ids_unique_across_sequences_of_one_flight(tmp_path):
    for vis, ir in (("VIS_0401", "IR_0402"), ("VIS_0405", "IR_0406")):
        for modality in (vis, ir):
            dirname = f"210924_FHL_Enterprise_{modality}"
            _make_frame(tmp_path / "wisard" / dirname, dirname, "00000000")

    ids = [r.id for r in discover_wisard(tmp_path)]

    assert sorted(ids) == [
        "wisard_210924_FHL_Enterprise_VIS_0401_00000000",
        "wisard_210924_FHL_Enterprise_VIS_0405_00000000",
    ]


def test_flags_sequences_unlabeled_in_one_modality(tmp_path):
    for dirname in ("210924_FHL_Enterprise_VIS_0403", "210924_FHL_Enterprise_IR_0404"):
        _make_frame(tmp_path / "wisard" / dirname, dirname, "00000000")

    (record,) = discover_wisard(tmp_path)

    assert (record.rgb_labeled, record.ir_labeled) == (False, True)
