"""Tests for official WILDS CAMELYON17 split mapping."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.splits import (
    SPLIT_TO_ID,
    TEST_CENTER,
    VAL_CENTER,
    assign_official_split_ids,
    canonicalize_split,
    official_split_ids_from_hf,
)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("train", "train"),
        ("id_val", "id_val"),
        ("val", "val"),
        ("test", "test"),
        ("validation", "val"),
        ("ood_val", "val"),
        ("id_validation", "id_val"),
    ],
)
def test_canonicalize_split_names_and_aliases(alias: str, canonical: str) -> None:
    assert canonicalize_split(alias) == canonical


def test_canonicalize_split_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        canonicalize_split("mixed-to-test")


def test_official_center_mapping() -> None:
    centers = np.array([0, 3, 4, VAL_CENTER, TEST_CENTER, 0, 3])
    raw_splits = np.array([0, 0, 0, 0, 0, 1, 1])

    split_ids = assign_official_split_ids(centers, raw_splits)

    assert split_ids[0] == SPLIT_TO_ID["train"]
    assert split_ids[1] == SPLIT_TO_ID["train"]
    assert split_ids[2] == SPLIT_TO_ID["train"]
    assert split_ids[3] == SPLIT_TO_ID["val"]
    assert split_ids[4] == SPLIT_TO_ID["test"]
    assert split_ids[5] == SPLIT_TO_ID["id_val"]
    assert split_ids[6] == SPLIT_TO_ID["id_val"]

    source = split_ids[np.isin(centers, [0, 3, 4])]
    assert set(source.tolist()) <= {SPLIT_TO_ID["train"], SPLIT_TO_ID["id_val"]}
    assert np.all(split_ids[centers == VAL_CENTER] == SPLIT_TO_ID["val"])
    assert np.all(split_ids[centers == TEST_CENTER] == SPLIT_TO_ID["test"])


def test_hf_split_mapping() -> None:
    train_centers = np.array([0, 3, 4])
    assert np.all(
        official_split_ids_from_hf("train", train_centers) == SPLIT_TO_ID["train"]
    )

    test_centers = np.array([2, 2])
    assert np.all(official_split_ids_from_hf("test", test_centers) == SPLIT_TO_ID["test"])

    validation_centers = np.array([0, 3, 4, 1])
    validation_ids = official_split_ids_from_hf("validation", validation_centers)
    assert validation_ids.tolist() == [
        SPLIT_TO_ID["id_val"],
        SPLIT_TO_ID["id_val"],
        SPLIT_TO_ID["id_val"],
        SPLIT_TO_ID["val"],
    ]
