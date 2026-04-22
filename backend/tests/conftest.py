from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.pipeline import FeatureEncoder


def make_window(
    window_id: str,
    *,
    patient_id: str = "559",
    split: str = "train",
    label: str = "controlled_response",
    carbs: float = 30.0,
    bolus: float = 3.0,
    baseline_glucose: float = 100.0,
    trend_30m: float = 5.0,
    trend_90m: float = 10.0,
    heart_rate_missing: float = 0.0,
) -> dict[str, Any]:
    premeal_values = np.linspace(baseline_glucose - 10, baseline_glucose, 19).tolist()
    premeal_delta = [value - baseline_glucose for value in premeal_values]
    return {
        "window_id": window_id,
        "patient_id": patient_id,
        "meal_time": "2022-01-01T12:00:00",
        "meal_segment": "lunch",
        "meal_type": "Lunch",
        "carbs": carbs,
        "bolus": bolus,
        "has_bolus": float(bolus > 0),
        "carbs_per_unit": carbs / bolus if bolus else carbs,
        "active_basal": 0.8,
        "baseline_glucose": baseline_glucose,
        "trend_30m": trend_30m,
        "trend_90m": trend_90m,
        "premeal_mean": float(np.mean(premeal_values)),
        "premeal_std": float(np.std(premeal_values)),
        "premeal_cv": float(np.std(premeal_values) / np.mean(premeal_values)),
        "hr_mean": 78.0 if not heart_rate_missing else 0.0,
        "hr_std": 4.0 if not heart_rate_missing else 0.0,
        "hr_min": 70.0 if not heart_rate_missing else 0.0,
        "hr_max": 86.0 if not heart_rate_missing else 0.0,
        "heart_rate_missing": heart_rate_missing,
        "premeal_values": premeal_values,
        "premeal_delta": premeal_delta,
        "premeal_missingness": [0.0] * 19,
        "full_curve_minutes": list(range(-90, 181, 5)),
        "full_curve_values": premeal_values + [baseline_glucose + 5.0] * 36,
        "full_curve_missingness": [0.0] * 55,
        "response_peak": baseline_glucose + 5.0,
        "response_nadir": baseline_glucose - 3.0,
        "rise_from_baseline": 5.0,
        "post_range": 8.0,
        "post_cv": 0.05,
        "post_tir": 0.95,
        "label": label,
        "label_display": label.replace("_", " "),
        "label_reason": "synthetic",
        "pre_coverage": 1.0,
        "post_coverage": 1.0,
        "split": split,
        "usable_for_memory": True,
        "exclusion_reason": None,
    }


@pytest.fixture()
def synthetic_windows() -> list[dict[str, Any]]:
    return [
        make_window("train-a", split="train", carbs=10.0, bolus=1.0, patient_id="559"),
        make_window("train-b", split="train", carbs=20.0, bolus=2.0, patient_id="563"),
        make_window("test-c", split="test", carbs=100.0, bolus=10.0, patient_id="570"),
        make_window("missing-hr", split="test", carbs=40.0, bolus=4.0, patient_id="575", heart_rate_missing=1.0),
    ]


@pytest.fixture()
def fitted_encoder(synthetic_windows: list[dict[str, Any]]) -> tuple[FeatureEncoder, np.ndarray]:
    return FeatureEncoder.fit(synthetic_windows)
