import numpy as np
import pandas as pd
import pytest

from src.data.sampling import sample_patches_by_patient


def make_metadata(patients: dict[int, int]) -> pd.DataFrame:
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


def test_samples_exact_number_per_patient():
    metadata = make_metadata({1: 10, 2: 20, 3: 30})

    sampled = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    counts = sampled["patient"].value_counts().sort_index()

    expected = pd.Series({1: 5, 2: 5, 3: 5})
    expected.index.name = "patient"

    pd.testing.assert_series_equal(
        counts,
        expected,
        check_names=False,
    )


def test_samples_without_replacement():
    metadata = make_metadata({1: 10, 2: 10})

    sampled = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    assert sampled.index.is_unique


def test_sampling_is_reproducible():
    metadata = make_metadata({1: 10, 2: 10, 3: 10})

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

    pd.testing.assert_frame_equal(sampled_1, sampled_2)


def test_different_seeds_can_produce_different_samples():
    metadata = make_metadata({1: 20, 2: 20})

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

    assert not sampled_1.index.equals(sampled_2.index)


def test_patient_identity_is_preserved():
    metadata = make_metadata({1: 10, 2: 10, 3: 10})

    sampled = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    assert set(sampled["patient"]) == {1, 2, 3}


def test_sampled_index_is_unique():
    metadata = make_metadata({1: 10, 2: 10, 3: 10})

    sampled = sample_patches_by_patient(
        metadata,
        n_samples=5,
        seed=42,
    )

    assert sampled.index.is_unique


def test_insufficient_patches_raises_value_error():
    metadata = make_metadata({1: 10, 2: 3})

    with pytest.raises(ValueError):
        sample_patches_by_patient(
            metadata,
            n_samples=5,
            seed=42,
        )


def test_invalid_n_samples_raises_value_error():
    metadata = make_metadata({1: 10})

    with pytest.raises(ValueError):
        sample_patches_by_patient(
            metadata,
            n_samples=0,
            seed=42,
        )


def test_missing_patient_column_raises_key_error():
    metadata = pd.DataFrame(
        {
            "patch_id": ["a", "b", "c"],
        }
    )

    with pytest.raises(KeyError):
        sample_patches_by_patient(
            metadata,
            n_samples=1,
            seed=42,
        )


def test_seed_must_be_integer():
    metadata = make_metadata({1: 10})

    with pytest.raises(TypeError):
        sample_patches_by_patient(
            metadata,
            n_samples=5,
            seed="42",
        )
```
