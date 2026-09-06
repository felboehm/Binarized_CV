from binarized_cv.data.datasets.trgb import discover_trgb


def _make_pair(dir_path, image_id):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{image_id}.jpg").write_bytes(b"")
    (dir_path / f"{image_id}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def test_discovers_pairs_across_splits(tmp_path):
    raw_root = tmp_path
    dataset_root = raw_root / "trgb" / "trgb_dataset"

    _make_pair(dataset_root / "train" / "RGB_images_train", "1001")
    _make_pair(dataset_root / "train" / "IR_images_train", "1001")
    _make_pair(dataset_root / "val" / "RGB_images_val", "2001")
    _make_pair(dataset_root / "val" / "IR_images_val", "2001")

    records = discover_trgb(raw_root)

    assert {r.split for r in records} == {"train", "val"}
    train_record = next(r for r in records if r.split == "train")
    assert train_record.id == "trgb_train_1001"
    assert train_record.rgb_image == "trgb/trgb_dataset/train/RGB_images_train/1001.jpg"
    assert train_record.ir_image == "trgb/trgb_dataset/train/IR_images_train/1001.jpg"


def test_handles_mangled_test_folder_name(tmp_path):
    raw_root = tmp_path
    dataset_root = raw_root / "trgb" / "trgb_dataset"

    _make_pair(dataset_root / "test" / "RGB_images_test copy", "3001")
    _make_pair(dataset_root / "test" / "IR_images_test", "3001")

    records = discover_trgb(raw_root)

    assert len(records) == 1
    assert records[0].split == "test"


def test_ignores_hidden_and_appledouble_files(tmp_path):
    raw_root = tmp_path
    dataset_root = raw_root / "trgb" / "trgb_dataset"
    rgb_dir = dataset_root / "train" / "RGB_images_train"
    ir_dir = dataset_root / "train" / "IR_images_train"

    _make_pair(rgb_dir, "4001")
    _make_pair(ir_dir, "4001")
    (rgb_dir / "._4001.jpg").write_bytes(b"")
    (dataset_root / "train" / ".DS_Store").write_bytes(b"")

    records = discover_trgb(raw_root)

    assert len(records) == 1
    assert records[0].id == "trgb_train_4001"


def test_matches_modality_prefixed_ids_across_folders(tmp_path):
    raw_root = tmp_path
    dataset_root = raw_root / "trgb" / "trgb_dataset"
    rgb_dir = dataset_root / "train" / "RGB_images_train"
    ir_dir = dataset_root / "train" / "IR_images_train"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)
    (rgb_dir / "RGB_1505.jpg").write_bytes(b"")
    (rgb_dir / "RGB_1505.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (ir_dir / "IR_1505.jpg").write_bytes(b"")
    (ir_dir / "IR_1505.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    records = discover_trgb(raw_root)

    assert len(records) == 1
    record = records[0]
    assert record.id == "trgb_train_1505"
    assert record.rgb_image == "trgb/trgb_dataset/train/RGB_images_train/RGB_1505.jpg"
    assert record.ir_image == "trgb/trgb_dataset/train/IR_images_train/IR_1505.jpg"


def test_bare_and_prefixed_ids_both_counted_in_same_folder(tmp_path):
    raw_root = tmp_path
    dataset_root = raw_root / "trgb" / "trgb_dataset"
    rgb_dir = dataset_root / "train" / "RGB_images_train"
    ir_dir = dataset_root / "train" / "IR_images_train"

    _make_pair(rgb_dir, "8251066")
    _make_pair(ir_dir, "8251066")
    (rgb_dir / "RGB_1505.jpg").write_bytes(b"")
    (rgb_dir / "RGB_1505.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (ir_dir / "IR_1505.jpg").write_bytes(b"")
    (ir_dir / "IR_1505.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    records = discover_trgb(raw_root)

    assert {r.id for r in records} == {"trgb_train_8251066", "trgb_train_1505"}


def test_skips_unpaired_ids(tmp_path):
    raw_root = tmp_path
    dataset_root = raw_root / "trgb" / "trgb_dataset"
    rgb_dir = dataset_root / "train" / "RGB_images_train"
    ir_dir = dataset_root / "train" / "IR_images_train"

    _make_pair(rgb_dir, "5001")
    _make_pair(rgb_dir, "5002")
    _make_pair(ir_dir, "5001")

    records = discover_trgb(raw_root)

    assert len(records) == 1
    assert records[0].id == "trgb_train_5001"
