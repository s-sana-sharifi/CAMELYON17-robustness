"""CAMELYON17-WILDS dataset interface."""

from __future__ import annotations

import tarfile
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
    """CAMELYON17 patches with official WILDS labels and metadata.

    Data can be loaded from:
    - a local WILDS directory containing ``metadata.csv`` and ``patches/``,
    - a local CAMELYON17 ``.tar`` archive,
    - an already-materialized Hugging Face dataset.

    No data is downloaded automatically.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        *,
        transform: Callable[[Image.Image], Any] | None = None,
        hf_dataset: Any | None = None,
        hf_split: str | None = None,
        tar_path: str | Path | None = None,
    ) -> None:
        self.split = canonicalize_split(split)
        self.transform = transform

        self._hf_dataset = None
        self._hf_indices: Sequence[int] | None = None

        self._data_dir: Path | None = None

        self._tar_path: Path | None = None
        self._tar: tarfile.TarFile | None = None
        self._tar_member_prefix = "camelyon17_v1.0/patches"

        provided_sources = sum(
            source is not None
            for source in (root, hf_dataset, tar_path)
        )

        if provided_sources != 1:
            raise ValueError(
                "Provide exactly one data source: `root`, `hf_dataset`, "
                "or `tar_path`."
            )

        if hf_dataset is not None:
            self._init_from_hf(hf_dataset, hf_split)
        elif tar_path is not None:
            self._init_from_tar(Path(tar_path))
        else:
            self._init_from_wilds_root(Path(root))

    @property
    def data_dir(self) -> Path | None:
        """Directory containing local WILDS files, if applicable."""
        return self._data_dir

    @property
    def tar_path(self) -> Path | None:
        """Path to the CAMELYON17 tar archive, if applicable."""
        return self._tar_path

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
        """Return filesystem path for a local WILDS patch."""
        if self._data_dir is None:
            raise RuntimeError(
                "patch_path() is only available for local WILDS files."
            )

        row = self._metadata.iloc[index]
        patient = row["patient"]
        node = int(row["node"])

        return self._data_dir / (
            f"patches/patient_{patient}_node_{node}/"
            f"patch_patient_{patient}_node_{node}_"
            f"x_{int(row['x_coord'])}_y_{int(row['y_coord'])}.png"
        )

    def _init_from_wilds_root(self, root: Path) -> None:
        data_dir = _resolve_wilds_dir(root)
        metadata_path = data_dir / "metadata.csv"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"CAMELYON17 metadata not found at {metadata_path}."
            )

        df = pd.read_csv(
            metadata_path,
            index_col=0,
            dtype={"patient": "str"},
        )

        self._set_metadata_from_wilds_dataframe(df)
        self._data_dir = data_dir

    def _init_from_tar(self, tar_path: Path) -> None:
        tar_path = tar_path.expanduser().resolve()

        if not tar_path.is_file():
            raise FileNotFoundError(
                f"CAMELYON17 tar archive not found at {tar_path}."
            )

        try:
            with tarfile.open(tar_path, mode="r") as tar:
                metadata_member = tar.getmember(
                    "camelyon17_v1.0/metadata.csv"
                )

                metadata_file = tar.extractfile(metadata_member)
                if metadata_file is None:
                    raise FileNotFoundError(
                        "Could not read metadata.csv from the CAMELYON17 "
                        "tar archive."
                    )

                df = pd.read_csv(
                    metadata_file,
                    dtype={"patient": "str"},
                )
        except tarfile.TarError as exc:
            raise RuntimeError(
                f"Could not read CAMELYON17 tar archive: {tar_path}"
            ) from exc

        self._set_metadata_from_wilds_dataframe(df)
        self._tar_path = tar_path

    def _set_metadata_from_wilds_dataframe(
        self,
        df: pd.DataFrame,
    ) -> None:
        required_columns = {
            "patient",
            "node",
            "x_coord",
            "y_coord",
            "tumor",
            "slide",
            "center",
            "split",
        }

        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                "CAMELYON17 metadata is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        split_ids = assign_official_split_ids(
            df["center"].to_numpy(),
            df["split"].to_numpy(),
        )

        keep = mask_for_split(split_ids, self.split)
        subset = df.loc[keep].copy()

        subset["label"] = subset["tumor"].astype("int64")
        subset["hospital"] = subset["center"].astype("int64")
        subset["image_id"] = subset.index.astype("int64")

        self._metadata = subset.loc[
            :,
            _METADATA_COLUMNS,
        ].reset_index(drop=True)

    def _init_from_hf(
        self,
        hf_dataset: Any,
        hf_split: str | None,
    ) -> None:
        dataset, inferred_hf_split = _unwrap_hf_dataset(
            hf_dataset,
            hf_split or HF_SPLIT_FOR_OFFICIAL[self.split],
        )

        centers = np.asarray(dataset["center"])
        split_ids = official_split_ids_from_hf(
            inferred_hf_split,
            centers,
        )

        keep = mask_for_split(split_ids, self.split)
        indices = np.flatnonzero(keep).tolist()

        if not indices:
            raise ValueError(
                f"No Hugging Face rows remain for official split "
                f"{self.split!r} inside HF split "
                f"{inferred_hf_split!r}."
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
        if (
            self._hf_dataset is not None
            and self._hf_indices is not None
        ):
            row = self._hf_dataset[self._hf_indices[index]]
            image = row["image"]

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            return image.convert("RGB")

        if self._tar_path is not None:
            return self._load_image_from_tar(index)

        path = self.patch_path(index)

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing patch {path}. "
                "Data is not downloaded automatically."
            )

        return Image.open(path).convert("RGB")

    def _load_image_from_tar(self, index: int) -> Image.Image:
        tar = self._get_tar()

        row = self._metadata.iloc[index]

        member_name = (
            f"{self._tar_member_prefix}/"
            f"patient_{row['patient']}_node_{int(row['node'])}/"
            f"patch_patient_{row['patient']}_node_{int(row['node'])}_"
            f"x_{int(row['x_coord'])}_y_{int(row['y_coord'])}.png"
        )

        try:
            member = tar.getmember(member_name)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Patch not found in CAMELYON17 archive: {member_name}"
            ) from exc

        file_object = tar.extractfile(member)

        if file_object is None:
            raise RuntimeError(
                f"Could not extract patch from archive: {member_name}"
            )

        with file_object:
            image = Image.open(file_object)
            image.load()

        return image.convert("RGB")

    def _get_tar(self) -> tarfile.TarFile:
        if self._tar_path is None:
            raise RuntimeError("Tar archive is not configured.")

        if self._tar is None:
            try:
                self._tar = tarfile.open(
                    self._tar_path,
                    mode="r",
                )
            except tarfile.TarError as exc:
                raise RuntimeError(
                    f"Could not open CAMELYON17 tar archive: "
                    f"{self._tar_path}"
                ) from exc

        return self._tar

    def close(self) -> None:
        """Close an open tar archive handle."""
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude open tar handles when a Dataset is pickled."""
        state = self.__dict__.copy()
        state["_tar"] = None
        return state


def _resolve_wilds_dir(root: Path) -> Path:
    root = root.expanduser().resolve()

    if (root / "metadata.csv").is_file():
        return root

    versioned = root / "camelyon17_v1.0"

    if (versioned / "metadata.csv").is_file():
        return versioned

    return versioned if (root / "camelyon17_v1.0").is_dir() else root


def _unwrap_hf_dataset(
    hf_dataset: Any,
    hf_split: str,
) -> tuple[Any, str]:
    if hasattr(hf_dataset, "keys") and hf_split in getattr(
        hf_dataset,
        "keys",
    )():
        return hf_dataset[hf_split], hf_split

    return hf_dataset, hf_split
