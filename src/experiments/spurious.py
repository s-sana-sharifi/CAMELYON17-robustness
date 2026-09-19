from __future__ import annotations

import numpy as np
import pandas as pd


PREFERRED_TUMOR_BY_CENTER = {
    0: 0,
    1: 1,
    2: 0,
    3: 1,
    4: None,
}


def assign_preferred_tumor(
    metadata: pd.DataFrame,
    center_col: str = "center",
) -> pd.Series:
    """
    Assign the experimental preferred tumor label to each center.

    Centers 0-3 have a preferred tumor label.
    Center 4 is neutral and receives NaN.

    Mapping:
        center 0 -> tumor 0
        center 1 -> tumor 1
        center 2 -> tumor 0
        center 3 -> tumor 1
        center 4 -> neutral

    This mapping is an experimental convention and has no
    biological interpretation.
    """
    if center_col not in metadata.columns:
        raise KeyError(
            f"Center column '{center_col}' not found."
        )

    centers = metadata[center_col].astype(int)

    if not centers.isin(PREFERRED_TUMOR_BY_CENTER).all():
        raise ValueError(
            "CAMELYON17 center values must be in {0, 1, 2, 3, 4}."
        )

    return centers.map(PREFERRED_TUMOR_BY_CENTER)


def compute_alignment(
    metadata: pd.DataFrame,
    center_col: str = "center",
    tumor_col: str = "tumor",
) -> float:
    """
    Compute alignment among non-neutral centers.

    Neutral center 4 is excluded from this metric because it has
    no preferred tumor label.
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

    non_neutral = preferred.notna()

    if not non_neutral.any():
        raise ValueError(
            "No non-neutral centers are present."
        )

    return float(
        (tumor[non_neutral] == preferred[non_neutral])
        .mean()
    )


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

    Centers 0-3 have a preferred tumor label. For each of these
    centers, exactly `rho` of sampled examples follow the preferred
    label.

    Center 4 is neutral and is sampled with equal numbers of tumor 0
    and tumor 1.

    The same number of samples is selected from every center.

    Sampling is without replacement and deterministic for a fixed seed.

    Because the preferred centers are balanced between tumor 0 and
    tumor 1, and the neutral center is sampled 50/50, the final
    dataset preserves a 50/50 marginal tumor distribution.
    """
    if not 0.5 <= rho <= 1.0:
        raise ValueError(
            "rho must be in the interval [0.5, 1.0]."
        )

    if n_samples_per_center <= 0:
        raise ValueError(
            "n_samples_per_center must be positive."
        )

    if n_samples_per_center % 2 != 0:
        raise ValueError(
            "n_samples_per_center must be even."
        )

    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")

    required_columns = {center_col, tumor_col}
    missing = required_columns - set(metadata.columns)

    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}"
        )

    centers = set(metadata[center_col].astype(int).unique())

    if centers != set(PREFERRED_TUMOR_BY_CENTER):
        raise ValueError(
            "Metadata must contain exactly centers {0, 1, 2, 3, 4}."
        )

    rng = np.random.default_rng(seed)
    sampled_parts = []

    for center in sorted(PREFERRED_TUMOR_BY_CENTER):

        center_metadata = metadata[
            metadata[center_col] == center
        ]

        preferred_tumor = PREFERRED_TUMOR_BY_CENTER[center]

        if preferred_tumor is None:
            # Neutral center: exactly 50/50 tumor distribution.
            n_tumor_0 = n_samples_per_center // 2
            n_tumor_1 = n_samples_per_center // 2

            tumor_0_pool = center_metadata[
                center_metadata[tumor_col] == 0
            ]

            tumor_1_pool = center_metadata[
                center_metadata[tumor_col] == 1
            ]

            if len(tumor_0_pool) < n_tumor_0:
                raise ValueError(
                    f"Center {center} does not contain enough "
                    "tumor-0 samples."
                )

            if len(tumor_1_pool) < n_tumor_1:
                raise ValueError(
                    f"Center {center} does not contain enough "
                    "tumor-1 samples."
                )

            tumor_0_indices = rng.choice(
                tumor_0_pool.index.to_numpy(),
                size=n_tumor_0,
                replace=False,
            )

            tumor_1_indices = rng.choice(
                tumor_1_pool.index.to_numpy(),
                size=n_tumor_1,
                replace=False,
            )

            selected_indices = np.concatenate(
                [tumor_0_indices, tumor_1_indices]
            )

        else:
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
                    "preferred-tumor samples."
                )

            if len(opposite_pool) < n_opposite:
                raise ValueError(
                    f"Center {center} does not contain enough "
                    "opposite-tumor samples."
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
