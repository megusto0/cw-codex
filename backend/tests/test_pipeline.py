from __future__ import annotations

import numpy as np

from app.memory import ContinuousHopfieldMemory
from app.pipeline import build_runtime_bundle


def test_feature_vector_shape_stability(fitted_encoder):
    encoder, matrix = fitted_encoder
    assert matrix.shape == (4, len(encoder.feature_names))
    expected_dimension = 19 + 19 + 19 + 11 + 6 + len(encoder.patient_ids) + 5
    assert matrix.shape[1] == expected_dimension


def test_missing_heart_rate_handling(fitted_encoder, synthetic_windows):
    encoder, matrix = fitted_encoder
    hr_start, hr_end = encoder.block_slices["heart_rate_context"]
    missing_index = next(index for index, window in enumerate(synthetic_windows) if window["window_id"] == "missing-hr")
    hr_block = matrix[missing_index, hr_start:hr_end]
    assert np.isfinite(hr_block).all()
    assert hr_block[-1] == 1.0


def test_scaler_fit_on_train_only(fitted_encoder, synthetic_windows):
    encoder, matrix = fitted_encoder
    meal_start, _ = encoder.block_slices["meal_context"]
    test_index = next(index for index, window in enumerate(synthetic_windows) if window["window_id"] == "test-c")
    transformed_carbs = matrix[test_index, meal_start]
    expected = (100.0 - 15.0) / 5.0
    assert np.isclose(transformed_carbs, expected)


def test_retrieval_returns_exactly_k_items_and_finite_energy():
    x_memory = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    metadata = [
        {"window_id": "a", "label": "controlled_response", "patient_id": "559"},
        {"window_id": "b", "label": "postprandial_spike", "patient_id": "563"},
        {"window_id": "c", "label": "late_low", "patient_id": "570"},
    ]
    memory = ContinuousHopfieldMemory().fit(x_memory, metadata)
    result = memory.retrieve(np.array([0.8, 0.2, 0.0]), k=2, beta=8.0, steps=3)
    assert len(result["top_k"]) == 2
    assert np.isfinite(memory.energy(np.array([0.8, 0.2, 0.0])))


def test_no_retrieval_leakage_or_self_retrieval_in_held_out_bundle():
    bundle = build_runtime_bundle(force=False)
    train_ids = {item["window_id"] for item in bundle["memory_model"].metadata}
    test_ids = {window["window_id"] for window in bundle["windows"] if window["split"] == "test"}
    assert train_ids.isdisjoint(test_ids)
