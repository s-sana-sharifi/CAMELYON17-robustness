"""CAMELYON17-WILDS data loading (official splits only)."""

from src.data.dataset import Camelyon17Dataset
from src.data.splits import (
    CENTER_NAMES,
    LABEL_NAMES,
    SPLIT_TO_ID,
    TEST_CENTER,
    VAL_CENTER,
    assign_official_split_ids,
    canonicalize_split,
)

__all__ = [
    "CENTER_NAMES",
    "LABEL_NAMES",
    "SPLIT_TO_ID",
    "TEST_CENTER",
    "VAL_CENTER",
    "Camelyon17Dataset",
    "assign_official_split_ids",
    "canonicalize_split",
]
