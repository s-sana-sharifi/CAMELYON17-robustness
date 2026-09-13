"""CAMELYON17-WILDS dataset interface (no download, official splits only)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.splits import (
    HF_SPLIT_FOR_OFFICIAL,
    assign_official_split_ids,
    canonicalize_split,
    mask_for_split,
    official_split_ids_from_hf,
)

_METADATA_COLUMNS = (
    "image_id",
    "patient",
    "node",
    "x_coord",
    "y_coord",
    "label",
    "slide",
    "center",
    "hospital",
)


class Camelyon17Dataset(Dataset):
    """96x96 CAMELYON17 patches with official WILDS labels and metadata.

    Load from a local WILDS directory (``metadata.csv`` + ``patches/``) or from
    an already-materialized Hugging Face ``Camelyon17-WILDS`` dataset. Data is
    never downloaded.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        *,
        transform: Callable[[Image.Image], Any] | None = None,
        hf_dataset: Any | None = None,
        hf_split: str | None = None,
    ) -> None:
        self.split = canonicalize_split(split)
        self.transform = transform
        self._hf_dataset = None
        self._hf_indices: Sequence[int] | None = None
        self._data_dir: Path | None = None

        if hf_dataset is not None:
            self._init_from_hf(hf_dataset, hf_split)
        elif root is not None:
            self._init_from_wilds_root(Path(root))
        else:
            raise ValueError("Provide `root` (local WILDS files) or `hf_dataset` (already loaded).")

    @property
    def data_dir(self) -> Path | None:
        """Directory containing ``metadata.csv`` and ``patches/``, if using local files."""
        return self._data_dir

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._metadata.iloc[index]
        image = self._load_image(index)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": int(row["label"]),
            "center": int(row["center"]),
            "hospital": int(row["hospital"]),
            "patient": str(row["patient"]),
            "slide": int(row["slide"]),
            "node": int(row["node"]),
            "x_coord": int(row["x_coord"]),
            "y_coord": int(row["y_coord"]),
            "image_id": int(row["image_id"]),
        }

    def patch_path(self, index: int) -> Path:
        """Filesystem path for a local WILDS patch (not used for Hugging Face rows)."""
        if self._data_dir is None:
            raise RuntimeError("patch_path() is only available for local WILDS files.")
        row = self._metadata.iloc[index]
        patient = row["patient"]
        node = int(row["node"])
        return self._data_dir / (
            f"patches/patient_{patient}_node_{node}/"
            f"patch_patient_{patient}_node_{node}_x_{int(row['x_coord'])}_y_{int(row['y_coord'])}.png"
        )

    def _init_from_wilds_root(self, root: Path) -> None:
        data_dir = _resolve_wilds_dir(root)
        metadata_path = data_dir / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"CAMELYON17 metadata not found at {metadata_path}. "
                "Place the official WILDS files locally; this loader does not download data."
            )
        df = pd.read_csv(metadata_path, index_col=0, dtype={"patient": "str"})
        split_ids = assign_official_split_ids(df["center"].to_numpy(), df["split"].to_numpy())
        keep = mask_for_split(split_ids, self.split)
        subset = df.loc[keep].copy()
        subset["label"] = subset["tumor"].astype("int64")
        subset["hospital"] = subset["center"].astype("int64")
        subset["image_id"] = subset.index.astype("int64")
        self._metadata = subset.loc[:, _METADATA_COLUMNS].reset_index(drop=True)
        self._data_dir = data_dir

    def _init_from_hf(self, hf_dataset: Any, hf_split: str | None) -> None:
        dataset, inferred_hf_split = _unwrap_hf_dataset(
            hf_dataset, hf_split or HF_SPLIT_FOR_OFFICIAL[self.split]
        )
        centers = np.asarray(dataset["center"])
        split_ids = official_split_ids_from_hf(inferred_hf_split, centers)
        keep = mask_for_split(split_ids, self.split)
        indices = np.flatnonzero(keep).tolist()
        if not indices:
            raise ValueError(
                f"No Hugging Face rows remain for official split {self.split!r} "
                f"inside HF split {inferred_hf_split!r}."
            )

        def col(name: str) -> np.ndarray:
            return np.asarray(dataset[name])[indices]

        records = {
            "image_id": col("image_id"),
            "patient": [str(v).zfill(3) for v in col("patient")],
            "node": col("node"),
            "x_coord": col("x_coord"),
            "y_coord": col("y_coord"),
            "label": col("label"),
            "slide": col("slide"),
            "center": centers[indices],
            "hospital": centers[indices],
        }
        self._metadata = pd.DataFrame(records)
        self._hf_dataset = dataset
        self._hf_indices = indices

    def _load_image(self, index: int) -> Image.Image:
        if self._hf_dataset is not None and self._hf_indices is not None:
            row = self._hf_dataset[self._hf_indices[index]]
            image = row["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            return image.convert("RGB")
        path = self.patch_path(index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing patch {path}. Data is not downloaded automatically.")
        return Image.open(path).convert("RGB")


def _resolve_wilds_dir(root: Path) -> Path:
    root = root.expanduser().resolve()
    if (root / "metadata.csv").is_file():
        return root
    versioned = root / "camelyon17_v1.0"
    if (versioned / "metadata.csv").is_file():
        return versioned
    return versioned if (root / "camelyon17_v1.0").is_dir() else root


def _unwrap_hf_dataset(hf_dataset: Any, hf_split: str) -> tuple[Any, str]:
    if hasattr(hf_dataset, "keys") and hf_split in getattr(hf_dataset, "keys")():
        return hf_dataset[hf_split], hf_split
    return hf_dataset, hf_split
