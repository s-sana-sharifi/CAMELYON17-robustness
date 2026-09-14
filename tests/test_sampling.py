```python
from __future__ import annotations

import pandas as pd
import pytest

from src.data.sampling import sample_patches_by_patient


def make_metadata(patients: dict[int, int]) -> pd.DataFrame:
    """Create synthetic metadata for testing patient-level sampling."""
    rows = []

    for patient, n_patches in patients.items():
        for patch_id in range(n_patches):
            rows.append(
                {
                    "patient": patient,
                    "patch_id": f"{patient}_{patch_id}",
                }
            )

    return pd.DataFrame(rows)


def test_samples_exact_number_per_patient() -> None:
    """Each patient should contribute exactly n_samples patches."""
    metadata = make_metadata(
        {
            1: 10,
            2: 20,
            3: 30,
        }
    )

    sampled = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    counts = sampled["patient"].value_counts().sort_index()

    expected = pd.Series(
        {
            1: 5,
            2: 5,
            3: 5,
        },
    )
    expected.index.name = "patient"

    pd.testing.assert_series_equal(
        counts,
        expected,
    )


def test_sampling_is_without_replacement() -> None:
    """Sampling should not select the same row more than once."""
    metadata = make_metadata(
        {
            1: 10,
            2: 10,
        }
    )

    sampled = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    assert len(sampled) == 10
    assert len(sampled.index.unique()) == len(sampled)


def test_sampling_is_reproducible() -> None:
    """The same seed should produce the same sample."""
    metadata = make_metadata(
        {
            1: 10,
            2: 20,
            3: 30,
        }
    )

    sampled_1 = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    sampled_2 = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    pd.testing.assert_frame_equal(
        sampled_1,
        sampled_2,
    )


def test_different_seeds_can_produce_different_samples() -> None:
    """Different seeds should be able to produce different samples."""
    metadata = make_metadata(
        {
            1: 20,
            2: 20,
            3: 20,
        }
    )

    sampled_1 = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    sampled_2 = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=123,
    )

    assert not sampled_1.equals(sampled_2)


def test_sampling_preserves_patient_identity() -> None:
    """Every sampled row should belong to a known patient."""
    metadata = make_metadata(
        {
            1: 10,
            2: 20,
            3: 30,
        }
    )

    sampled = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    assert set(sampled["patient"]) == {1, 2, 3}


def test_sampling_without_replacement() -> None:
    """The sampler should return unique patch rows."""
    metadata = make_metadata(
        {
            1: 10,
            2: 20,
        }
    )

    sampled = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    assert sampled["patch_id"].is_unique


def test_raises_when_patient_has_too_few_patches() -> None:
    """Sampling should fail if a patient has fewer than n_samples patches."""
    metadata = make_metadata(
        {
            1: 10,
            2: 3,
            3: 20,
        }
    )

    with pytest.raises(ValueError):
        sample_patches_by_patient(
            metadata,
            n_samples=5,
            seed=42,
        )


def test_raises_for_invalid_n_samples() -> None:
    """n_samples must be positive."""
    metadata = make_metadata(
        {
            1: 10,
            2: 20,
        }
    )

    with pytest.raises(ValueError):
        sample_patches_by_patient(
            metadata,
            n_samples=0,
            seed=42,
        )


def test_raises_for_missing_patient_column() -> None:
    """The sampler should require the configured patient column."""
    metadata = pd.DataFrame(
        {
            "patch_id": ["a", "b", "c"],
        }
    )

    with pytest.raises(KeyError):
        sample_patches_by_patient(
            metadata,
            n_samples=2,
            seed=42,
        )


def test_seed_must_be_integer() -> None:
    """The seed must be an integer."""
    metadata = make_metadata(
        {
            1: 10,
            2: 20,
        }
    )

    with pytest.raises(TypeError):
        sample_patches_by_patient(
            metadata,
            n_samples=5,
            seed="42",
        )
```

