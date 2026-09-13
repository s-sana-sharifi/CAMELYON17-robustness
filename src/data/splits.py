"""Official WILDS CAMELYON17 split semantics.

WILDS stores a raw ``split`` column (0 = train, 1 = in-distribution holdout)
and then remaps hospitals 1 and 2 to OOD validation and test. Hugging Face
``Camelyon17-WILDS`` keeps the same hospital IDs; its ``validation`` split is
the union of WILDS ``id_val`` and ``val``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray

VAL_CENTER = 1
TEST_CENTER = 2

SPLIT_TO_ID: Mapping[str, int] = {
    "train": 0,
    "id_val": 1,
    "test": 2,
    "val": 3,
}
ID_TO_SPLIT: Mapping[int, str] = {idx: name for name, idx in SPLIT_TO_ID.items()}

SPLIT_ALIASES: Mapping[str, str] = {
    "validation": "val",
    "ood_val": "val",
    "id_validation": "id_val",
}

CENTER_NAMES: Mapping[int, str] = {
    0: "train-center1",
    1: "validation-center",
    2: "test-center",
    3: "train-center2",
    4: "train-center3",
}
LABEL_NAMES: Mapping[int, str] = {0: "non-tumor", 1: "tumor"}

# HF split name that contains each official WILDS split.
HF_SPLIT_FOR_OFFICIAL: Mapping[str, str] = {
    "train": "train",
    "id_val": "validation",
    "val": "validation",
    "test": "test",
}


def canonicalize_split(split: str) -> str:
    """Return the official WILDS split name (``train`` / ``id_val`` / ``val`` / ``test``)."""
    name = SPLIT_ALIASES.get(split, split)
    if name not in SPLIT_TO_ID:
        allowed = ", ".join(sorted(SPLIT_TO_ID) + sorted(SPLIT_ALIASES))
        raise ValueError(f"Unknown split {split!r}. Expected one of: {allowed}")
    return name


def assign_official_split_ids(
    centers: NDArray[np.integer],
    raw_splits: NDArray[np.integer],
) -> NDArray[np.integer]:
    """Apply the official WILDS center remapping to the metadata ``split`` column."""
    split_ids = np.asarray(raw_splits, dtype=np.int64).copy()
    centers = np.asarray(centers)
    split_ids[centers == VAL_CENTER] = SPLIT_TO_ID["val"]
    split_ids[centers == TEST_CENTER] = SPLIT_TO_ID["test"]
    return split_ids


def official_split_ids_from_hf(
    hf_split: str,
    centers: NDArray[np.integer],
) -> NDArray[np.integer]:
    """Map a Hugging Face CAMELYON17-WILDS split onto official WILDS split IDs."""
    centers = np.asarray(centers)
    if hf_split == "train":
        return np.full(len(centers), SPLIT_TO_ID["train"], dtype=np.int64)
    if hf_split == "test":
        return np.full(len(centers), SPLIT_TO_ID["test"], dtype=np.int64)
    if hf_split == "validation":
        split_ids = np.full(len(centers), SPLIT_TO_ID["id_val"], dtype=np.int64)
        split_ids[centers == VAL_CENTER] = SPLIT_TO_ID["val"]
        return split_ids
    raise ValueError(
        f"Unknown Hugging Face split {hf_split!r}. Expected 'train', 'validation', or 'test'."
    )


def mask_for_split(split_ids: NDArray[np.integer], split: str) -> NDArray[np.bool_]:
    """Boolean mask of rows that belong to an official WILDS split."""
    split_name = canonicalize_split(split)
    return np.asarray(split_ids) == SPLIT_TO_ID[split_name]
