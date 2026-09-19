from __future__ import annotations

import numpy as np
import pandas as pd


def assign_preferred_tumor(
    metadata: pd.DataFrame,
    center_col: str = "center",
) -> pd.Series:
    """
    Assign the predefined preferred tumor label to each center.

    The mapping is:
        center 0 -> tumor 0
        center 1 -> tumor 1
        center 2 -> tumor 0
        center 3 -> tumor 1
        center 4 -> tumor 0

    This mapping is an experimental convention and has no biological
    interpretation.
    """
    if center_col not in metadata.columns:
        raise KeyError(
            f"Center column '{center_col}' not found."
        )

    centers = metadata[center_col].astype(int)

    if not centers.isin([0, 1, 2, 3, 4]).all():
        raise ValueError(
            "CAMELYON17 center values must be in {0, 1, 2, 3, 4}."
        )

    return centers % 2


def compute_alignment(
    metadata: pd.DataFrame,
    center_col: str = "center",
    tumor_col: str = "tumor",
) -> float:
    """
    Compute the proportion of samples whose tumor label matches
    the predefined preferred label for their center.
    """
    if center_col not in metadata.columns:
        raise KeyError(
            f"Center column '{center_col}' not found."
        )

    if tumor_col not in metadata.columns:
        raise KeyError(
            f"Tumor column '{tumor_col}' not found."
        )

    preferred = assign_preferred_tumor(
        metadata,
        center_col=center_col,
    )

    tumor = metadata[tumor_col].astype(int)

    return float((tumor == preferred).mean())


def sample_with_center_tumor_alignment(
    metadata: pd.DataFrame,
    rho: float,
    n_samples_per_center: int,
    seed: int,
    center_col: str = "center",
    tumor_col: str = "tumor",
) -> pd.DataFrame:
    """
    Construct a controlled center-tumor distribution.

    For each center, exactly `n_samples_per_center` samples are selected.
    A proportion `rho` of samples follow the center's preferred tumor
    label, while the remainder have the opposite tumor label.

    Sampling is without replacement and deterministic for a fixed seed.

    The returned dataset has equal numbers of samples from each center.
    """
    if not 0.5 <= rho <= 1.0:
        raise ValueError(
            "rho must be in the interval [0.5, 1.0]."
        )

    if n_samples_per_center <= 0:
        raise ValueError(
            "n_samples_per_center must be positive."
        )

    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")

    required_columns = {center_col, tumor_col}

    missing = required_columns - set(metadata.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    rng = np.random.default_rng(seed)

    sampled_parts = []

    for center in sorted(metadata[center_col].unique()):

        center_metadata = metadata[
            metadata[center_col] == center
        ]

        preferred_tumor = int(center) % 2
        opposite_tumor = 1 - preferred_tumor

        preferred_pool = center_metadata[
            center_metadata[tumor_col] == preferred_tumor
        ]

        opposite_pool = center_metadata[
            center_metadata[tumor_col] == opposite_tumor
        ]

        n_preferred = int(
            round(n_samples_per_center * rho)
        )

        n_opposite = (
            n_samples_per_center - n_preferred
        )

        if len(preferred_pool) < n_preferred:
            raise ValueError(
                f"Center {center} does not contain enough "
                f"preferred-label samples. "
                f"Required={n_preferred}, "
                f"available={len(preferred_pool)}."
            )

        if len(opposite_pool) < n_opposite:
            raise ValueError(
                f"Center {center} does not contain enough "
                f"opposite-label samples. "
                f"Required={n_opposite}, "
                f"available={len(opposite_pool)}."
            )

        preferred_indices = rng.choice(
            preferred_pool.index.to_numpy(),
            size=n_preferred,
            replace=False,
        )

        opposite_indices = rng.choice(
            opposite_pool.index.to_numpy(),
            size=n_opposite,
            replace=False,
        )

        selected_indices = np.concatenate(
            [
                preferred_indices,
                opposite_indices,
            ]
        )

        sampled_parts.append(
            metadata.loc[selected_indices]
        )

    sampled = pd.concat(
        sampled_parts,
        axis=0,
    )

    sampled = sampled.sample(
        frac=1.0,
        random_state=seed,
    )

    return sampled.reset_index(drop=True)
