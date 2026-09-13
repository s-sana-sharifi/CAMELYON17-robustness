"""Tests for the CAMELYON17 local/HF dataset interface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset import Camelyon17Dataset
from src.data.splits import assign_official_split_ids, mask_for_split

REPO_ROOT = Path(__file__).resolve().parents[1]
WILDS_METADATA = REPO_ROOT / "data" / "camelyon17_v1.0" / "metadata.csv"

EXPECTED_SPLIT_SIZES = {
    "train": 302436,
    "id_val": 33560,
    "val": 34904,
    "test": 85054,
}

METADATA_FIELDS = {
    "image",
    "label",
    "center",
    "hospital",
    "patient",
    "slide",
    "node",
    "x_coord",
    "y_coord",
    "image_id",
}


class _FakeHFSplit:
    """Minimal stand-in for a loaded Hugging Face split (columns + row access)."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self._columns = {key: [row[key] for row in rows] for key in rows[0]}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._columns[key]
        return self._rows[key]


def _write_synthetic_metadata(root: Path) -> Path:
    data_dir = root / "camelyon17_v1.0"
    data_dir.mkdir()
    (data_dir / "patches").mkdir()
    rows = [
        # image_id, patient, node, x, y, tumor, slide, center, split
        (0, "004", 4, 10, 20, 1, 0, 0, 0),
        (1, "015", 2, 11, 21, 0, 5, 0, 1),
        (2, "020", 4, 12, 22, 1, 10, 1, 0),
        (3, "042", 3, 13, 23, 0, 23, 2, 0),
        (4, "060", 3, 14, 24, 1, 30, 3, 0),
        (5, "096", 0, 15, 25, 0, 48, 4, 1),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "image_id",
            "patient",
            "node",
            "x_coord",
            "y_coord",
            "tumor",
            "slide",
            "center",
            "split",
        ],
    ).set_index("image_id")
    df.to_csv(data_dir / "metadata.csv")
    return data_dir


def test_expected_split_sizes_from_local_metadata() -> None:
    if not WILDS_METADATA.is_file():
        pytest.skip("local CAMELYON17-WILDS metadata.csv is not present")
    df = pd.read_csv(WILDS_METADATA, index_col=0, dtype={"patient": "str"})
    split_ids = assign_official_split_ids(df["center"].to_numpy(), df["split"].to_numpy())
    sizes = {
        name: int(mask_for_split(split_ids, name).sum())
        for name in EXPECTED_SPLIT_SIZES
    }
    assert sizes == EXPECTED_SPLIT_SIZES


def test_metadata_fields_and_identifiers(tmp_path: Path) -> None:
    data_dir = _write_synthetic_metadata(tmp_path)
    ds = Camelyon17Dataset(root=tmp_path, split="train")
    assert len(ds) == 2
    assert list(ds._metadata["patient"]) == ["004", "060"]
    assert list(ds._metadata["slide"]) == [0, 30]
    assert list(ds._metadata["center"]) == [0, 3]
    assert list(ds._metadata.columns) == [
        "image_id",
        "patient",
        "node",
        "x_coord",
        "y_coord",
        "label",
        "slide",
        "center",
        "hospital",
    ]

    row = ds._metadata.iloc[0]
    assert int(row["label"]) == 1
    assert int(row["hospital"]) == int(row["center"]) == 0
    assert int(row["node"]) == 4
    assert int(row["x_coord"]) == 10
    assert int(row["y_coord"]) == 20
    assert int(row["image_id"]) == 0

    val_ds = Camelyon17Dataset(root=data_dir, split="validation")
    assert len(val_ds) == 1
    assert val_ds._metadata.iloc[0]["patient"] == "020"
    assert int(val_ds._metadata.iloc[0]["center"]) == 1

    test_ds = Camelyon17Dataset(root=data_dir, split="test")
    assert test_ds._metadata.iloc[0]["patient"] == "042"
    assert int(test_ds._metadata.iloc[0]["slide"]) == 23

    id_val_ds = Camelyon17Dataset(root=data_dir, split="id_val")
    assert list(id_val_ds._metadata["patient"]) == ["015", "096"]
    assert list(id_val_ds._metadata["slide"]) == [5, 48]


def test_missing_local_patch_raises_file_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_synthetic_metadata(tmp_path)
    ds = Camelyon17Dataset(root=tmp_path, split="train")

    def _fail_download(*_args, **_kwargs):
        raise AssertionError("dataset loader must not download missing patches")

    monkeypatch.setattr("urllib.request.urlopen", _fail_download)
    monkeypatch.setattr("urllib.request.urlretrieve", _fail_download)

    with pytest.raises(FileNotFoundError, match="not downloaded automatically"):
        ds[0]


def test_hf_dataset_preserves_metadata_and_identifiers() -> None:
    hf_val = _FakeHFSplit(
        [
            {
                "image": None,
                "label": 0,
                "center": 0,
                "image_id": 10,
                "patient": 15,
                "node": 2,
                "x_coord": 100,
                "y_coord": 200,
                "slide": 5,
            },
            {
                "image": None,
                "label": 1,
                "center": 1,
                "image_id": 11,
                "patient": 20,
                "node": 4,
                "x_coord": 101,
                "y_coord": 201,
                "slide": 10,
            },
            {
                "image": None,
                "label": 0,
                "center": 3,
                "image_id": 12,
                "patient": 60,
                "node": 3,
                "x_coord": 102,
                "y_coord": 202,
                "slide": 30,
            },
        ]
    )
    id_val = Camelyon17Dataset(hf_dataset=hf_val, split="id_val", hf_split="validation")
    assert len(id_val) == 2
    assert list(id_val._metadata["patient"]) == ["015", "060"]
    assert list(id_val._metadata["slide"]) == [5, 30]
    assert list(id_val._metadata["center"]) == [0, 3]
    assert set(id_val._metadata.columns) == METADATA_FIELDS - {"image"}

    val = Camelyon17Dataset(hf_dataset=hf_val, split="val", hf_split="validation")
    assert len(val) == 1
    assert val._metadata.iloc[0]["patient"] == "020"
    assert int(val._metadata.iloc[0]["center"]) == 1
    assert int(val._metadata.iloc[0]["label"]) == 1
