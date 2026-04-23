from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timezone
import json
from pathlib import Path
import pickle
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from .config import Settings, get_settings
from .memory import ContinuousHopfieldMemory, l2_normalize, stable_softmax
from .reporting import generate_coursework_report


TS_FORMAT = "%d-%m-%Y %H:%M:%S"
SPECIAL_LABELS = {"pre_existing_hyper", "pre_existing_low"}
PROTOTYPE_LABELS = [
    "controlled_response",
    "postprandial_spike",
    "late_low",
    "unstable_response",
]
DISPLAY_LABELS = {
    "controlled_response": "Контролируемый отклик",
    "postprandial_spike": "Постпрандиальный пик",
    "late_low": "Позднее снижение",
    "unstable_response": "Нестабильный отклик",
    "pre_existing_hyper": "Исходная гипергликемия",
    "pre_existing_low": "Исходная гипогликемия",
    "ambiguous": "Неоднозначный отклик",
    "custom_query": "Пользовательский запрос",
}
MEAL_SEGMENT_LABELS = {
    "breakfast": "Завтрак",
    "lunch": "Обед",
    "dinner": "Ужин",
    "overnight": "Ночной интервал",
}
SPLIT_LABELS = {
    "train": "Память",
    "val": "Валидационный срез",
    "test": "Отложенная проверка",
    "excluded": "Исключено",
    "query": "Запрос",
}
FEATURE_BLOCK_LABELS = {
    "premeal_cgm": "Предпищевой профиль CGM",
    "premeal_delta": "Отклонение от базовой гликемии",
    "missingness": "Индикаторы пропусков",
    "meal_context": "Контекст приема пищи",
    "time_context": "Временной контекст",
    "patient_context": "Контекст пациента",
    "heart_rate_context": "Контекст частоты сердечных сокращений",
}
EXCLUSION_REASON_LABELS = {
    "invalid_carbs": "некорректная запись углеводов",
    "overlap_previous_meal": "перекрытие с предыдущим приемом пищи",
    "overlap_next_meal": "перекрытие со следующим приемом пищи",
    "missing_baseline": "отсутствует базовая точка CGM перед приемом пищи",
    "insufficient_premeal_cgm": "недостаточное покрытие CGM до приема пищи",
    "insufficient_postmeal_cgm": "недостаточное покрытие CGM после приема пищи",
    "missing_premeal_signal": "отсутствует предпищевой сигнал",
    "pre_existing_state": "окно исключено из-за исходного состояния до приема пищи",
}
BASELINE_METHOD_LABELS = {
    "hopfield": "Ассоциативная память Хопфилда",
    "cosine_knn": "Косинусное kNN-извлечение",
    "nearest_prototype": "Ближайший прототип",
    "patient_majority": "Пациент-специфический модальный класс",
    "logistic_regression": "Логистическая регрессия",
}
PROTOTYPE_INTERPRETATIONS = {
    "controlled_response": "Прототип описывает окна с умеренным постпрандиальным откликом и сохранением гликемии в целевом диапазоне.",
    "postprandial_spike": "Прототип описывает случаи с выраженным подъемом гликемии после приема пищи при сходном предпищевом контексте.",
    "late_low": "Прототип объединяет траектории, в которых после начального ответа формируется позднее снижение гликемии.",
    "unstable_response": "Прототип фиксирует редкие окна с высокой вариабельностью и слабой устойчивостью профиля.",
}


@dataclass
class PatientStreams:
    patient_id: str
    glucose: pd.DataFrame
    meals: pd.DataFrame
    bolus: pd.DataFrame
    basal: pd.DataFrame
    temp_basal: pd.DataFrame
    heart_rate: pd.DataFrame


@dataclass
class FeatureEncoder:
    patient_ids: list[str]
    feature_names: list[str]
    block_slices: dict[str, tuple[int, int]]
    scale_mask: np.ndarray
    mean_: np.ndarray
    scale_: np.ndarray
    default_premeal_delta_curve: list[float]
    defaults: dict[str, float]
    premeal_length: int
    full_minutes: list[int]

    @staticmethod
    def meal_segments() -> list[str]:
        return ["breakfast", "lunch", "dinner", "overnight"]

    @classmethod
    def fit(cls, windows: Sequence[dict[str, Any]]) -> tuple["FeatureEncoder", np.ndarray]:
        patient_ids = sorted({str(window["patient_id"]) for window in windows})
        premeal_length = len(windows[0]["premeal_values"])

        feature_names: list[str] = []
        block_slices: dict[str, tuple[int, int]] = {}
        scale_mask: list[bool] = []
        cursor = 0

        def register_block(name: str, names: Sequence[str], scalable: Sequence[bool]) -> None:
            nonlocal cursor
            block_slices[name] = (cursor, cursor + len(names))
            feature_names.extend(names)
            scale_mask.extend(scalable)
            cursor += len(names)

        register_block(
            "premeal_cgm",
            [f"premeal_cgm_{index:02d}" for index in range(premeal_length)],
            [True] * premeal_length,
        )
        register_block(
            "premeal_delta",
            [f"premeal_delta_{index:02d}" for index in range(premeal_length)],
            [True] * premeal_length,
        )
        register_block(
            "missingness",
            [f"premeal_missing_{index:02d}" for index in range(premeal_length)],
            [False] * premeal_length,
        )
        register_block(
            "meal_context",
            [
                "carbs",
                "bolus",
                "has_bolus",
                "carbs_per_unit",
                "baseline_glucose",
                "trend_30m",
                "trend_90m",
                "premeal_mean",
                "premeal_std",
                "premeal_cv",
                "active_basal",
            ],
            [True, True, False, True, True, True, True, True, True, True, True],
        )
        register_block(
            "time_context",
            ["time_sin", "time_cos", "segment_breakfast", "segment_lunch", "segment_dinner", "segment_overnight"],
            [False, False, False, False, False, False],
        )
        register_block(
            "patient_context",
            [f"patient_{patient_id}" for patient_id in patient_ids],
            [False] * len(patient_ids),
        )
        register_block(
            "heart_rate_context",
            ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_missing"],
            [True, True, True, True, False],
        )

        encoder = cls(
            patient_ids=patient_ids,
            feature_names=feature_names,
            block_slices=block_slices,
            scale_mask=np.asarray(scale_mask, dtype=bool),
            mean_=np.zeros(cursor, dtype=float),
            scale_=np.ones(cursor, dtype=float),
            default_premeal_delta_curve=[],
            defaults={},
            premeal_length=premeal_length,
            full_minutes=list(range(-90, 181, 5)),
        )
        raw_matrix = np.vstack([encoder._raw_vector(window) for window in windows]).astype(float)
        train_mask = np.array([window["split"] == "train" for window in windows], dtype=bool)
        train_matrix = raw_matrix[train_mask]

        if train_matrix.size == 0:
            raise RuntimeError("Отсутствуют train-окна для обучения кодировщика признаков.")

        encoder.mean_[encoder.scale_mask] = np.mean(train_matrix[:, encoder.scale_mask], axis=0)
        encoder.scale_[encoder.scale_mask] = np.std(train_matrix[:, encoder.scale_mask], axis=0)
        encoder.scale_ = np.where(encoder.scale_ < 1e-6, 1.0, encoder.scale_)

        train_windows = [window for window in windows if window["split"] == "train"]
        encoder.default_premeal_delta_curve = (
            np.median(np.asarray([window["premeal_delta"] for window in train_windows], dtype=float), axis=0).tolist()
        )
        encoder.defaults = {
            "carbs": float(np.median([window["carbs"] for window in train_windows])),
            "bolus": float(np.median([window["bolus"] for window in train_windows])),
            "baseline_glucose": float(np.median([window["baseline_glucose"] for window in train_windows])),
            "trend_30m": float(np.median([window["trend_30m"] for window in train_windows])),
            "trend_90m": float(np.median([window["trend_90m"] for window in train_windows])),
            "premeal_mean": float(np.median([window["premeal_mean"] for window in train_windows])),
            "premeal_std": float(np.median([window["premeal_std"] for window in train_windows])),
            "premeal_cv": float(np.median([window["premeal_cv"] for window in train_windows])),
            "active_basal": float(np.median([window["active_basal"] for window in train_windows])),
            "hr_mean": float(np.median([window["hr_mean"] for window in train_windows])),
            "hr_std": float(np.median([window["hr_std"] for window in train_windows])),
            "hr_min": float(np.median([window["hr_min"] for window in train_windows])),
            "hr_max": float(np.median([window["hr_max"] for window in train_windows])),
        }
        return encoder, encoder.transform_raw_matrix(raw_matrix)

    def transform_raw_matrix(self, raw_matrix: np.ndarray) -> np.ndarray:
        transformed = raw_matrix.copy()
        transformed[:, self.scale_mask] = (transformed[:, self.scale_mask] - self.mean_[self.scale_mask]) / self.scale_[
            self.scale_mask
        ]
        return transformed

    def transform_window(self, window: dict[str, Any]) -> np.ndarray:
        raw_vector = self._raw_vector(window).reshape(1, -1)
        return self.transform_raw_matrix(raw_vector)[0]

    def build_custom_window(self, payload: dict[str, Any]) -> dict[str, Any]:
        baseline = float(payload.get("baseline_glucose", self.defaults["baseline_glucose"]))
        premeal_values = payload.get("premeal_cgm")
        if premeal_values is None:
            premeal_values = [baseline + delta for delta in self.default_premeal_delta_curve]
        else:
            premeal_values = np.asarray(premeal_values, dtype=float)
            if premeal_values.size != self.premeal_length:
                source_grid = np.linspace(0.0, 1.0, num=premeal_values.size)
                target_grid = np.linspace(0.0, 1.0, num=self.premeal_length)
                premeal_values = np.interp(target_grid, source_grid, premeal_values)
            premeal_values = premeal_values.tolist()

        premeal_missingness = payload.get("premeal_missingness")
        if premeal_missingness is None:
            premeal_missingness = [0.0] * self.premeal_length
        elif len(premeal_missingness) != self.premeal_length:
            premeal_missingness = [1.0] * self.premeal_length

        meal_hour = int(payload.get("meal_hour", 12))
        meal_segment = classify_meal_segment(meal_hour)
        heart_rate_values = {
            "hr_mean": float(payload.get("hr_mean", self.defaults["hr_mean"])),
            "hr_std": float(payload.get("hr_std", self.defaults["hr_std"])),
            "hr_min": float(payload.get("hr_min", self.defaults["hr_min"])),
            "hr_max": float(payload.get("hr_max", self.defaults["hr_max"])),
            "heart_rate_missing": bool(payload.get("heart_rate_missing", False)),
        }
        bolus = float(payload.get("bolus", self.defaults["bolus"]))
        carbs = float(payload.get("carbs", self.defaults["carbs"]))
        active_basal = float(payload.get("active_basal", self.defaults["active_basal"]))

        window = {
            "window_id": "custom-query",
            "patient_id": str(payload.get("patient_id", "custom")),
            "meal_time": payload.get("meal_time", "custom"),
            "meal_segment": meal_segment,
            "meal_type": payload.get("meal_type", "Пользовательское окно"),
            "carbs": carbs,
            "bolus": bolus,
            "has_bolus": float(bolus > 0.0),
            "carbs_per_unit": float(carbs / bolus) if bolus > 0 else float(carbs),
            "active_basal": active_basal,
            "baseline_glucose": baseline,
            "trend_30m": float(payload.get("trend_30m", self.defaults["trend_30m"])),
            "trend_90m": float(payload.get("trend_90m", self.defaults["trend_90m"])),
            "premeal_mean": float(np.mean(premeal_values)),
            "premeal_std": float(np.std(premeal_values)),
            "premeal_cv": float(np.std(premeal_values) / max(np.mean(premeal_values), 1.0)),
            "premeal_values": [float(value) for value in premeal_values],
            "premeal_delta": [float(value - baseline) for value in premeal_values],
            "premeal_missingness": [float(value) for value in premeal_missingness],
            "full_curve_minutes": self.full_minutes,
            "full_curve_values": [float(value) for value in premeal_values]
            + [float(baseline)] * (len(self.full_minutes) - len(premeal_values)),
            "full_curve_missingness": [float(value) for value in premeal_missingness]
            + [1.0] * (len(self.full_minutes) - len(premeal_values)),
            "label": "custom_query",
            "label_reason": "Пользовательский исследовательский запрос без клинической интерпретации.",
            "response_peak": baseline,
            "response_nadir": baseline,
            "rise_from_baseline": 0.0,
            "post_range": 0.0,
            "post_cv": 0.0,
            "post_tir": 1.0,
            "split": "query",
            "usable_for_memory": False,
            **heart_rate_values,
        }
        return window

    def block_similarity(self, query_vector: np.ndarray, candidate_vector: np.ndarray) -> dict[str, float]:
        similarities: dict[str, float] = {}
        for name, (start, end) in self.block_slices.items():
            query_block = query_vector[start:end]
            candidate_block = candidate_vector[start:end]
            numerator = float(np.dot(query_block, candidate_block))
            denominator = float(np.linalg.norm(query_block) * np.linalg.norm(candidate_block))
            if denominator == 0.0:
                similarities[name] = 1.0
            else:
                similarities[name] = max(0.0, min(1.0, (numerator / denominator + 1.0) / 2.0))
        return similarities

    def _patient_vector(self, patient_id: str) -> list[float]:
        return [1.0 if patient_id == candidate else 0.0 for candidate in self.patient_ids]

    def _segment_vector(self, meal_segment: str) -> list[float]:
        return [1.0 if meal_segment == segment else 0.0 for segment in self.meal_segments()]

    def _raw_vector(self, window: dict[str, Any]) -> np.ndarray:
        values: list[float] = []
        values.extend(float(value) for value in window["premeal_values"])
        values.extend(float(value) for value in window["premeal_delta"])
        values.extend(float(value) for value in window["premeal_missingness"])
        values.extend(
            [
                float(window["carbs"]),
                float(window["bolus"]),
                float(window["has_bolus"]),
                float(window["carbs_per_unit"]),
                float(window["baseline_glucose"]),
                float(window["trend_30m"]),
                float(window["trend_90m"]),
                float(window["premeal_mean"]),
                float(window["premeal_std"]),
                float(window["premeal_cv"]),
                float(window["active_basal"]),
            ]
        )
        hour = extract_meal_hour(window.get("meal_time"))
        radians = 2.0 * np.pi * (hour / 24.0)
        values.extend([float(np.sin(radians)), float(np.cos(radians))])
        values.extend(self._segment_vector(window["meal_segment"]))
        values.extend(self._patient_vector(str(window["patient_id"])))
        values.extend(
            [
                float(window["hr_mean"]),
                float(window["hr_std"]),
                float(window["hr_min"]),
                float(window["hr_max"]),
                float(window["heart_rate_missing"]),
            ]
        )
        return np.asarray(values, dtype=float)


def safe_float(value: Any) -> float | None:
    if value in (None, "", " ", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.to_datetime(value, format=TS_FORMAT)
    except (TypeError, ValueError):
        return None


def classify_meal_segment(hour: int) -> str:
    if 5 <= hour < 10:
        return "breakfast"
    if 10 <= hour < 15:
        return "lunch"
    if 15 <= hour < 21:
        return "dinner"
    return "overnight"


def extract_meal_hour(value: Any) -> int:
    if isinstance(value, pd.Timestamp):
        return int(value.hour)
    if isinstance(value, str) and value != "custom":
        try:
            return int(pd.Timestamp(value).hour)
        except ValueError:
            return 12
    return 12


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " "))


def display_meal_segment(segment: str) -> str:
    return MEAL_SEGMENT_LABELS.get(segment, segment)


def display_split(split: str) -> str:
    return SPLIT_LABELS.get(split, split)


def display_feature_block(block: str) -> str:
    return FEATURE_BLOCK_LABELS.get(block, block.replace("_", " "))


def display_exclusion_reason(reason: str | None) -> str:
    if not reason:
        return "не указана"
    return EXCLUSION_REASON_LABELS.get(reason, reason.replace("_", " "))


def display_baseline_method(name: str) -> str:
    return BASELINE_METHOD_LABELS.get(name, name.replace("_", " "))


def format_share(value: float) -> str:
    return f"{value * 100:.1f}%"


def classify_uncertainty(gap: float, entropy: float) -> str:
    if gap < 0.03 or entropy > 5.0:
        return "высокая неоднозначность"
    if gap < 0.08 or entropy > 4.5:
        return "умеренная неоднозначность"
    return "устойчивое извлечение"


def prototype_meaning(label: str) -> str:
    return PROTOTYPE_INTERPRETATIONS.get(label, "Прототип отражает усредненный профиль соответствующего класса окон.")


def build_case_summary(record: dict[str, Any]) -> str:
    predicted = display_label(record["predicted_label"])
    observed = display_label(record["label"])
    uncertainty = classify_uncertainty(record["top_weight_gap"], record["attention_entropy"])
    patient_scope = (
        "с преобладанием внутрипациентских совпадений"
        if record["same_patient_rate"] >= 0.5
        else "с заметной межпациентской компонентой"
    )
    if record["top1_correct"]:
        return (
            f"Запрос отнесен к классу «{predicted}» уже на первой позиции; {uncertainty}, "
            f"а ближайшие случаи согласованы {patient_scope}."
        )
    return (
        f"Запрос с истинной меткой «{observed}» смещен к классу «{predicted}»; "
        f"{uncertainty}, что указывает на перекрытие паттернов {patient_scope}."
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def parse_ohio_directory(data_dir: Path) -> dict[str, PatientStreams]:
    if not data_dir.exists():
        raise FileNotFoundError(f"OhioT1DM directory not found: {data_dir}")

    grouped_files: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(data_dir.glob("*.xml")):
        grouped_files[path.name.split("-")[0]].append(path)

    streams_by_patient: dict[str, PatientStreams] = {}
    for patient_id, paths in grouped_files.items():
        glucose_frames: list[pd.DataFrame] = []
        meal_frames: list[pd.DataFrame] = []
        bolus_frames: list[pd.DataFrame] = []
        basal_frames: list[pd.DataFrame] = []
        temp_basal_frames: list[pd.DataFrame] = []
        heart_rate_frames: list[pd.DataFrame] = []

        for path in paths:
            root = ET.parse(path).getroot()
            glucose_frames.append(_parse_simple_stream(root.find("glucose_level"), value_keys=("value",)))
            meal_frames.append(_parse_simple_stream(root.find("meal"), value_keys=("carbs",), text_keys=("type",)))
            bolus_frames.append(_parse_interval_stream(root.find("bolus"), value_keys=("dose", "bwz_carb_input"), text_keys=("type",)))
            basal_frames.append(_parse_simple_stream(root.find("basal"), value_keys=("value",)))
            temp_basal_frames.append(_parse_interval_stream(root.find("temp_basal"), value_keys=("value",)))
            heart_rate_frames.append(_parse_simple_stream(root.find("basis_heart_rate"), value_keys=("value",)))

        streams_by_patient[patient_id] = PatientStreams(
            patient_id=patient_id,
            glucose=_combine_frames(glucose_frames, dedupe_columns=["timestamp"]),
            meals=_combine_frames(meal_frames, dedupe_columns=["timestamp", "carbs"]),
            bolus=_combine_frames(bolus_frames, dedupe_columns=["timestamp", "dose"]),
            basal=_combine_frames(basal_frames, dedupe_columns=["timestamp", "value"]),
            temp_basal=_combine_frames(temp_basal_frames, dedupe_columns=["timestamp", "end_timestamp", "value"]),
            heart_rate=_combine_frames(heart_rate_frames, dedupe_columns=["timestamp", "value"]),
        )
    return streams_by_patient


def _parse_simple_stream(node: ET.Element | None, value_keys: Sequence[str], text_keys: Sequence[str] = ()) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    columns = ["timestamp", *value_keys, *text_keys]
    if node is None:
        return pd.DataFrame(columns=columns)
    for event in node.findall("event"):
        timestamp = parse_timestamp(event.attrib.get("ts"))
        if timestamp is None:
            continue
        record: dict[str, Any] = {"timestamp": timestamp}
        for key in value_keys:
            record[key] = safe_float(event.attrib.get(key))
        for key in text_keys:
            record[key] = (event.attrib.get(key) or "").strip()
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def _parse_interval_stream(node: ET.Element | None, value_keys: Sequence[str], text_keys: Sequence[str] = ()) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    columns = ["timestamp", "end_timestamp", *value_keys, *text_keys]
    if node is None:
        return pd.DataFrame(columns=columns)
    for event in node.findall("event"):
        start = parse_timestamp(event.attrib.get("ts_begin"))
        end = parse_timestamp(event.attrib.get("ts_end"))
        if start is None:
            continue
        record: dict[str, Any] = {"timestamp": start, "end_timestamp": end}
        for key in value_keys:
            record[key] = safe_float(event.attrib.get(key))
        for key in text_keys:
            record[key] = (event.attrib.get(key) or "").strip()
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def _combine_frames(frames: Sequence[pd.DataFrame], dedupe_columns: Sequence[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame(columns=frames[0].columns)
    combined = pd.concat(usable, ignore_index=True)
    combined = combined.drop_duplicates(subset=list(dedupe_columns))
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def align_signal_to_grid(
    timestamps: Sequence[pd.Timestamp],
    values: Sequence[float],
    origin: pd.Timestamp,
    grid_minutes: np.ndarray,
    max_gap_minutes: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(timestamps) == 0:
        nan_values = np.full(len(grid_minutes), np.nan, dtype=float)
        return nan_values, np.ones(len(grid_minutes), dtype=float)

    relative = np.asarray([(timestamp - origin).total_seconds() / 60.0 for timestamp in timestamps], dtype=float)
    signal = np.asarray(values, dtype=float)
    order = np.argsort(relative)
    relative = relative[order]
    signal = signal[order]

    aligned = np.full(len(grid_minutes), np.nan, dtype=float)
    missingness = np.ones(len(grid_minutes), dtype=float)

    for index, point in enumerate(grid_minutes):
        position = int(np.searchsorted(relative, point))
        prev_idx = position - 1
        next_idx = position
        prev_time = relative[prev_idx] if prev_idx >= 0 else None
        next_time = relative[next_idx] if next_idx < len(relative) else None

        if prev_time is not None and abs(prev_time - point) <= 2.5:
            aligned[index] = signal[prev_idx]
            missingness[index] = 0.0
            continue
        if next_time is not None and abs(next_time - point) <= 2.5:
            aligned[index] = signal[next_idx]
            missingness[index] = 0.0
            continue

        if prev_time is not None and next_time is not None and (next_time - prev_time) <= max_gap_minutes:
            fraction = (point - prev_time) / max(next_time - prev_time, 1e-6)
            aligned[index] = signal[prev_idx] + fraction * (signal[next_idx] - signal[prev_idx])
            missingness[index] = 1.0
            continue

        if prev_time is not None and abs(point - prev_time) <= max_gap_minutes / 2.0:
            aligned[index] = signal[prev_idx]
            missingness[index] = 1.0
            continue
        if next_time is not None and abs(next_time - point) <= max_gap_minutes / 2.0:
            aligned[index] = signal[next_idx]
            missingness[index] = 1.0

    return aligned, missingness


def label_window(metrics: dict[str, float]) -> tuple[str, str]:
    baseline = metrics["baseline_glucose"]
    rise = metrics["rise_from_baseline"]
    peak = metrics["peak_15_180"]
    nadir = metrics["nadir_45_180"]
    post_range = metrics["post_range"]
    post_cv = metrics["post_cv"]
    post_tir = metrics["post_tir"]

    if baseline < 70:
        return "pre_existing_low", "Перед приемом пищи базовая гликемия уже была ниже 70 мг/дл."
    if baseline > 180:
        return "pre_existing_hyper", "Перед приемом пищи базовая гликемия уже превышала 180 мг/дл."
    if nadir < 70 and baseline >= 80:
        return "late_low", "В интервале после приема пищи зарегистрировано позднее снижение ниже 70 мг/дл."
    if baseline < 180 and (rise >= 60 or (peak > 180 and rise >= 40)):
        return "postprandial_spike", "После приема пищи наблюдался выраженный подъем гликемии относительно исходного уровня."
    if post_range >= 100 or post_cv >= 0.36:
        return "unstable_response", "Постпрандиальное окно характеризуется широким диапазоном или высокой вариабельностью."
    if 70 <= baseline <= 180 and peak <= 180 and nadir >= 70 and rise < 40 and post_tir >= 0.7:
        return "controlled_response", "Постпрандиальный отклик в основном оставался в целевом диапазоне без крупной экскурсии."
    return "ambiguous", "Окно не соответствует ни одному из основных ретроспективных паттернов с достаточной определенностью."


def extract_windows(streams_by_patient: dict[str, PatientStreams], settings: Settings) -> tuple[list[dict[str, Any]], dict[str, int]]:
    full_minutes = np.arange(-settings.premeal_minutes, settings.postmeal_minutes + settings.grid_minutes, settings.grid_minutes)
    pre_mask = full_minutes <= 0
    post_mask = full_minutes >= 0
    windows: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()

    for patient_id, streams in streams_by_patient.items():
        if streams.meals.empty or streams.glucose.empty:
            continue
        meal_times = streams.meals["timestamp"].to_list()
        glucose_times = streams.glucose["timestamp"].to_list()
        glucose_values = streams.glucose["value"].ffill().bfill().to_list()

        for meal_index, meal_row in streams.meals.iterrows():
            meal_time = meal_row["timestamp"]
            carbs = safe_float(meal_row.get("carbs"))
            exclusion_reason: str | None = None
            if carbs is None or carbs <= 0:
                exclusion_reason = "invalid_carbs"
            elif meal_index > 0 and (meal_time - meal_times[meal_index - 1]).total_seconds() < settings.premeal_minutes * 60:
                exclusion_reason = "overlap_previous_meal"
            elif meal_index < len(meal_times) - 1 and (meal_times[meal_index + 1] - meal_time).total_seconds() < settings.postmeal_minutes * 60:
                exclusion_reason = "overlap_next_meal"

            full_curve_raw, full_missingness = align_signal_to_grid(glucose_times, glucose_values, meal_time, full_minutes)
            pre_curve_raw = full_curve_raw[pre_mask]
            post_curve_raw = full_curve_raw[post_mask]
            pre_missingness = full_missingness[pre_mask]
            full_curve_for_metrics = full_curve_raw.copy()

            pre_coverage = float(np.mean(~np.isnan(pre_curve_raw)))
            post_coverage = float(np.mean(~np.isnan(post_curve_raw)))
            baseline_glucose = float(pre_curve_raw[-1]) if not np.isnan(pre_curve_raw[-1]) else np.nan

            if exclusion_reason is None:
                if np.isnan(baseline_glucose):
                    exclusion_reason = "missing_baseline"
                elif pre_coverage < 0.75:
                    exclusion_reason = "insufficient_premeal_cgm"
                elif post_coverage < 0.70:
                    exclusion_reason = "insufficient_postmeal_cgm"

            pre_fill_value = float(np.nanmedian(pre_curve_raw)) if np.sum(~np.isnan(pre_curve_raw)) else np.nan
            if np.isnan(pre_fill_value):
                exclusion_reason = exclusion_reason or "missing_premeal_signal"
                pre_fill_value = 0.0
            pre_curve = np.where(np.isnan(pre_curve_raw), pre_fill_value, pre_curve_raw)
            baseline_glucose = float(pre_curve[-1])
            premeal_delta = pre_curve - baseline_glucose

            if np.sum(~np.isnan(post_curve_raw)) == 0:
                post_fill_value = baseline_glucose
            else:
                post_fill_value = float(np.nanmedian(post_curve_raw))
            full_curve_for_metrics = np.where(np.isnan(full_curve_for_metrics), post_fill_value, full_curve_for_metrics)
            post_curve = full_curve_for_metrics[post_mask]
            peak_15_180 = float(np.max(post_curve[3:])) if len(post_curve) > 3 else float(np.max(post_curve))
            nadir_45_180 = float(np.min(post_curve[9:])) if len(post_curve) > 9 else float(np.min(post_curve))
            rise_from_baseline = float(peak_15_180 - baseline_glucose)
            post_range = float(np.max(post_curve) - np.min(post_curve))
            post_mean = float(np.mean(post_curve))
            post_cv = float(np.std(post_curve) / max(post_mean, 1.0))
            post_tir = float(np.mean((post_curve >= 70) & (post_curve <= 180)))
            metrics = {
                "baseline_glucose": baseline_glucose,
                "peak_15_180": peak_15_180,
                "nadir_45_180": nadir_45_180,
                "rise_from_baseline": rise_from_baseline,
                "post_range": post_range,
                "post_cv": post_cv,
                "post_tir": post_tir,
            }
            label, label_reason = label_window(metrics)

            bolus_window_start = meal_time - pd.Timedelta(minutes=30)
            bolus_window_end = meal_time + pd.Timedelta(minutes=30)
            bolus_rows = streams.bolus[
                (streams.bolus["timestamp"] >= bolus_window_start) & (streams.bolus["timestamp"] <= bolus_window_end)
            ]
            bolus = float(bolus_rows["dose"].fillna(0.0).sum()) if not bolus_rows.empty else 0.0
            active_basal = resolve_active_basal(streams, meal_time)

            hr_window_start = meal_time - pd.Timedelta(minutes=settings.premeal_minutes)
            hr_rows = streams.heart_rate[
                (streams.heart_rate["timestamp"] >= hr_window_start) & (streams.heart_rate["timestamp"] <= meal_time)
            ]
            if hr_rows.empty:
                hr_mean = hr_std = hr_min = hr_max = np.nan
                heart_rate_missing = True
            else:
                hr_values = hr_rows["value"].dropna().to_numpy(dtype=float)
                hr_mean = float(np.mean(hr_values))
                hr_std = float(np.std(hr_values))
                hr_min = float(np.min(hr_values))
                hr_max = float(np.max(hr_values))
                heart_rate_missing = False

            hour = int(meal_time.hour)
            meal_segment = classify_meal_segment(hour)
            has_bolus = float(bolus > 0.0)
            carbs_per_unit = float(carbs / bolus) if bolus > 0 else float(carbs)
            premeal_mean = float(np.mean(pre_curve))
            premeal_std = float(np.std(pre_curve))
            premeal_cv = float(premeal_std / max(premeal_mean, 1.0))
            trend_30m = float(pre_curve[-1] - pre_curve[-7]) if len(pre_curve) >= 7 else 0.0
            trend_90m = float(pre_curve[-1] - pre_curve[0])
            usable_for_memory = exclusion_reason is None and label not in SPECIAL_LABELS

            if exclusion_reason is None and label in SPECIAL_LABELS:
                exclusion_reason = "pre_existing_state"

            if exclusion_reason:
                exclusion_counts[exclusion_reason] += 1

            window = {
                "window_id": f"{patient_id}-{meal_time.strftime('%Y%m%d%H%M%S')}-{meal_index}",
                "patient_id": patient_id,
                "meal_time": meal_time,
                "meal_type": meal_row.get("type") or "Прием пищи",
                "meal_segment": meal_segment,
                "meal_segment_display": display_meal_segment(meal_segment),
                "carbs": float(carbs or 0.0),
                "bolus": bolus,
                "has_bolus": has_bolus,
                "carbs_per_unit": carbs_per_unit,
                "active_basal": float(active_basal),
                "baseline_glucose": baseline_glucose,
                "trend_30m": trend_30m,
                "trend_90m": trend_90m,
                "premeal_mean": premeal_mean,
                "premeal_std": premeal_std,
                "premeal_cv": premeal_cv,
                "hr_mean": float(hr_mean) if not np.isnan(hr_mean) else 0.0,
                "hr_std": float(hr_std) if not np.isnan(hr_std) else 0.0,
                "hr_min": float(hr_min) if not np.isnan(hr_min) else 0.0,
                "hr_max": float(hr_max) if not np.isnan(hr_max) else 0.0,
                "heart_rate_missing": float(heart_rate_missing),
                "premeal_values": pre_curve.tolist(),
                "premeal_delta": premeal_delta.tolist(),
                "premeal_missingness": pre_missingness.tolist(),
                "full_curve_minutes": full_minutes.tolist(),
                "full_curve_values": full_curve_raw.tolist(),
                "full_curve_missingness": full_missingness.tolist(),
                "response_peak": peak_15_180,
                "response_nadir": nadir_45_180,
                "rise_from_baseline": rise_from_baseline,
                "post_range": post_range,
                "post_cv": post_cv,
                "post_tir": post_tir,
                "label": label,
                "label_display": display_label(label),
                "label_reason": label_reason,
                "pre_coverage": pre_coverage,
                "post_coverage": post_coverage,
                "split": "excluded",
                "split_display": display_split("excluded"),
                "usable_for_memory": usable_for_memory,
                "exclusion_reason": exclusion_reason,
                "exclusion_reason_display": display_exclusion_reason(exclusion_reason),
            }
            windows.append(window)
    return windows, dict(exclusion_counts)


def resolve_active_basal(streams: PatientStreams, meal_time: pd.Timestamp) -> float:
    basal_rows = streams.basal[streams.basal["timestamp"] <= meal_time]
    basal_value = float(basal_rows.iloc[-1]["value"]) if not basal_rows.empty else 0.0
    active_temp = streams.temp_basal[
        (streams.temp_basal["timestamp"] <= meal_time)
        & (streams.temp_basal["end_timestamp"].notna())
        & (streams.temp_basal["end_timestamp"] >= meal_time)
    ]
    if not active_temp.empty and active_temp.iloc[-1]["value"] is not None:
        return float(active_temp.iloc[-1]["value"])
    return basal_value


def assign_splits(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [window for window in windows if window["usable_for_memory"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in eligible:
        grouped[str(window["patient_id"])].append(window)

    for patient_windows in grouped.values():
        patient_windows.sort(key=lambda item: item["meal_time"])
        count = len(patient_windows)
        train_end = max(1, int(count * 0.70))
        val_end = max(train_end + 1, int(count * 0.85))
        val_end = min(val_end, count - 1) if count > 2 else count
        for index, window in enumerate(patient_windows):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"
            window["split"] = split
            window["split_display"] = display_split(split)
    return windows


def build_prototypes(
    train_windows: Sequence[dict[str, Any]],
    feature_matrix: np.ndarray,
    encoder: FeatureEncoder,
    index_by_window_id: dict[str, int],
) -> dict[str, dict[str, Any]]:
    prototypes: dict[str, dict[str, Any]] = {}
    train_matrix = feature_matrix[[index_by_window_id[window["window_id"]] for window in train_windows]]
    train_ids = [window["window_id"] for window in train_windows]
    train_labels = [window["label"] for window in train_windows]
    normalized_train = l2_normalize(train_matrix)

    for label in PROTOTYPE_LABELS:
        members = [window for window in train_windows if window["label"] == label]
        if not members:
            continue
        member_indices = [index_by_window_id[window["window_id"]] for window in members]
        member_matrix = feature_matrix[member_indices]
        centroid = np.mean(member_matrix, axis=0)
        centroid = l2_normalize(centroid.reshape(1, -1))[0]
        similarities = normalized_train @ centroid
        nearest_train_indices = np.argsort(similarities)[::-1][: min(15, len(train_windows))]
        neighbor_labels = [train_labels[index] for index in nearest_train_indices]
        purity = float(neighbor_labels.count(label) / max(len(neighbor_labels), 1))

        representative_members = np.argsort(l2_normalize(member_matrix) @ centroid)[::-1][: min(3, len(members))]
        representative_windows = [members[index]["window_id"] for index in representative_members]
        mean_curve = np.nanmean(np.asarray([window["full_curve_values"] for window in members], dtype=float), axis=0)

        prototypes[label] = {
            "label": label,
            "label_display": display_label(label),
            "support_size": len(members),
            "support_fraction": float(len(members) / max(len(train_windows), 1)),
            "purity": purity,
            "vector": centroid.tolist(),
            "mean_curve_minutes": members[0]["full_curve_minutes"],
            "mean_curve_values": mean_curve.tolist(),
            "representative_window_ids": representative_windows,
            "interpretation_text": prototype_meaning(label),
            "typical_context": {
                "carbs": float(np.median([window["carbs"] for window in members])),
                "bolus": float(np.median([window["bolus"] for window in members])),
                "baseline_glucose": float(np.median([window["baseline_glucose"] for window in members])),
                "trend_30m": float(np.median([window["trend_30m"] for window in members])),
                "trend_90m": float(np.median([window["trend_90m"] for window in members])),
                "meal_segment_mode": Counter([window["meal_segment"] for window in members]).most_common(1)[0][0],
                "meal_segment_mode_display": display_meal_segment(
                    Counter([window["meal_segment"] for window in members]).most_common(1)[0][0]
                ),
            },
        }
    return prototypes


def aggregate_label_weights(labels: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for label, weight in zip(labels, weights):
        totals[str(label)] += float(weight)
    return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        },
    }


def evaluate_models(
    windows: Sequence[dict[str, Any]],
    feature_matrix: np.ndarray,
    encoder: FeatureEncoder,
    index_by_window_id: dict[str, int],
    memory_model: ContinuousHopfieldMemory,
    prototypes: dict[str, dict[str, Any]],
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_windows = [window for window in windows if window["split"] == "train"]
    test_windows = [window for window in windows if window["split"] == "test"]
    train_labels = [window["label"] for window in train_windows]
    train_matrix = feature_matrix[[index_by_window_id[window["window_id"]] for window in train_windows]]
    train_norm = l2_normalize(train_matrix)
    prototype_labels = list(prototypes.keys())
    prototype_matrix = (
        l2_normalize(np.asarray([prototypes[label]["vector"] for label in prototype_labels], dtype=float))
        if prototype_labels
        else np.empty((0, feature_matrix.shape[1]))
    )

    patient_majority = {
        patient_id: Counter(window["label"] for window in patient_windows).most_common(1)[0][0]
        for patient_id, patient_windows in group_by_key(train_windows, "patient_id").items()
    }

    logistic = LogisticRegression(max_iter=600)
    logistic_labels: list[str] = []
    if len(set(train_labels)) > 1:
        logistic.fit(train_matrix, train_labels)
        logistic_labels = logistic.classes_.tolist()

    hopfield_records: list[dict[str, Any]] = []
    cosine_predictions: list[str] = []
    hopfield_predictions: list[str] = []
    prototype_predictions: list[str] = []
    patient_majority_predictions: list[str] = []
    logistic_predictions: list[str] = []
    y_true: list[str] = []

    for query_window in test_windows:
        query_vector = feature_matrix[index_by_window_id[query_window["window_id"]]]
        retrieval = memory_model.retrieve(
            query_vector,
            k=settings.retrieval_top_k,
            beta=settings.hopfield_beta,
            steps=settings.recall_steps,
        )
        top_ids = [item["window_id"] for item in retrieval["top_k"]]
        top_labels = [item["label"] for item in retrieval["top_k"]]
        top_patients = [item["patient_id"] for item in retrieval["top_k"]]
        label_weights = aggregate_label_weights(train_labels, retrieval["weights"])
        predicted_label = next(iter(label_weights)) if label_weights else top_labels[0]

        query_norm = l2_normalize(query_vector.reshape(1, -1))[0]
        cosine_similarities = train_norm @ query_norm
        cosine_indices = np.argsort(cosine_similarities)[::-1][: settings.retrieval_top_k]
        cosine_top_labels = [train_windows[index]["label"] for index in cosine_indices]
        cosine_predictions.append(cosine_top_labels[0])

        if prototype_labels:
            prototype_similarities = prototype_matrix @ query_norm
            prototype_label = prototype_labels[int(np.argmax(prototype_similarities))]
            prototype_distribution = stable_softmax(prototype_similarities).tolist()
            sorted_prototype = np.sort(np.asarray(prototype_distribution, dtype=float))
            prototype_gap = float(sorted_prototype[-1] - sorted_prototype[-2]) if len(sorted_prototype) > 1 else float(sorted_prototype[-1])
        else:
            prototype_label = "ambiguous"
            prototype_distribution = []
            prototype_gap = 0.0

        patient_majority_predictions.append(patient_majority.get(query_window["patient_id"], Counter(train_labels).most_common(1)[0][0]))
        prototype_predictions.append(prototype_label)
        hopfield_predictions.append(predicted_label)
        y_true.append(query_window["label"])

        if logistic_labels:
            logistic_predictions.append(str(logistic.predict(query_vector.reshape(1, -1))[0]))

        rank = next((index + 1 for index, label in enumerate(top_labels) if label == query_window["label"]), None)
        same_patient_rate = float(np.mean([patient == query_window["patient_id"] for patient in top_patients])) if top_patients else 0.0
        entropy = float(-np.sum(np.clip(retrieval["weights"], 1e-12, 1.0) * np.log(np.clip(retrieval["weights"], 1e-12, 1.0))))
        sorted_weights = np.sort(np.asarray(retrieval["weights"], dtype=float))
        gap = float(sorted_weights[-1] - sorted_weights[-2]) if len(sorted_weights) > 1 else float(sorted_weights[-1])
        top_similarity = float(retrieval["top_k"][0]["similarity"]) if retrieval["top_k"] else 0.0
        second_similarity = float(retrieval["top_k"][1]["similarity"]) if len(retrieval["top_k"]) > 1 else 0.0
        top_weight = float(retrieval["top_k"][0]["weight"]) if retrieval["top_k"] else 0.0
        second_weight = float(retrieval["top_k"][1]["weight"]) if len(retrieval["top_k"]) > 1 else 0.0
        case_record = {
            "window_id": query_window["window_id"],
            "patient_id": query_window["patient_id"],
            "label": query_window["label"],
            "top_ids": top_ids,
            "top_labels": top_labels,
            "top_patients": top_patients,
            "top1_correct": top_labels[0] == query_window["label"],
            "top3_hit": query_window["label"] in top_labels[:3],
            "top5_hit": query_window["label"] in top_labels[:5],
            "mrr": 1.0 / rank if rank else 0.0,
            "label_purity_top5": float(np.mean([label == query_window["label"] for label in top_labels])) if top_labels else 0.0,
            "same_patient_rate": same_patient_rate,
            "cross_patient_top1": bool(top_patients and top_patients[0] != query_window["patient_id"]),
            "energy_before": memory_model.energy(query_vector, beta=settings.hopfield_beta),
            "energy_after": float(retrieval["trajectory"][-1]["energy"]) if retrieval["trajectory"] else memory_model.energy(query_vector),
            "energy_drop": memory_model.energy(query_vector, beta=settings.hopfield_beta)
            - (float(retrieval["trajectory"][-1]["energy"]) if retrieval["trajectory"] else memory_model.energy(query_vector)),
            "attention_entropy": entropy,
            "top_weight_gap": gap,
            "top1_similarity": top_similarity,
            "top2_similarity": second_similarity,
            "top1_weight": top_weight,
            "top2_weight": second_weight,
            "predicted_label": predicted_label,
            "prototype_label": prototype_label,
            "prototype_top_gap": prototype_gap,
            "prototype_distribution": dict(zip(prototype_labels, prototype_distribution)),
        }
        case_record["summary_text"] = build_case_summary(case_record)

        hopfield_records.append(case_record)

    prototype_purity = float(np.mean([prototype["purity"] for prototype in prototypes.values()])) if prototypes else 0.0
    baseline_comparison = {
        "hopfield": classification_metrics(y_true, hopfield_predictions, sorted(set(y_true))),
        "cosine_knn": classification_metrics(y_true, cosine_predictions, sorted(set(y_true))),
        "nearest_prototype": classification_metrics(y_true, prototype_predictions, sorted(set(y_true))),
        "patient_majority": classification_metrics(y_true, patient_majority_predictions, sorted(set(y_true))),
    }
    if logistic_predictions:
        baseline_comparison["logistic_regression"] = classification_metrics(y_true, logistic_predictions, sorted(set(y_true)))

    noise_robustness = evaluate_noise_robustness(test_windows, feature_matrix, index_by_window_id, memory_model, settings)
    per_patient = []
    for patient_id, records in group_by_records(hopfield_records, "patient_id").items():
        per_patient.append(
            {
                "patient_id": patient_id,
                "top1_accuracy": float(np.mean([record["top1_correct"] for record in records])),
                "top3_hit_rate": float(np.mean([record["top3_hit"] for record in records])),
                "mrr": float(np.mean([record["mrr"] for record in records])),
                "same_patient_rate": float(np.mean([record["same_patient_rate"] for record in records])),
                "count": len(records),
            }
        )

    successes = sorted(
        [record for record in hopfield_records if record["top1_correct"]],
        key=lambda item: item["top_weight_gap"],
        reverse=True,
    )
    failures = sorted(
        [record for record in hopfield_records if not record["top1_correct"]],
        key=lambda item: (item["top_weight_gap"], -item["attention_entropy"]),
    )
    ambiguous_cases = sorted(hopfield_records, key=lambda item: (item["top_weight_gap"], -item["attention_entropy"]))
    prototype_confusions = sorted(
        [record for record in hopfield_records if record["prototype_label"] != record["label"]],
        key=lambda item: (item["prototype_top_gap"], item["top_weight_gap"]),
    )

    retrieval_metrics = {
        "top1_accuracy": float(np.mean([record["top1_correct"] for record in hopfield_records])),
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in hopfield_records])),
        "top5_hit_rate": float(np.mean([record["top5_hit"] for record in hopfield_records])),
        "mean_reciprocal_rank": float(np.mean([record["mrr"] for record in hopfield_records])),
        "label_purity_top5": float(np.mean([record["label_purity_top5"] for record in hopfield_records])),
        "prototype_purity": prototype_purity,
        "nearest_neighbor_consistency": float(np.mean([record["top1_correct"] for record in hopfield_records])),
    }

    diagnostics = {
        "average_energy_before": float(np.mean([record["energy_before"] for record in hopfield_records])),
        "average_energy_after": float(np.mean([record["energy_after"] for record in hopfield_records])),
        "average_energy_drop": float(np.mean([record["energy_drop"] for record in hopfield_records])),
        "average_top_weight_gap": float(np.mean([record["top_weight_gap"] for record in hopfield_records])),
        "average_attention_entropy": float(np.mean([record["attention_entropy"] for record in hopfield_records])),
        "same_patient_top1_rate": float(np.mean([not record["cross_patient_top1"] for record in hopfield_records])),
        "cross_patient_top1_rate": float(np.mean([record["cross_patient_top1"] for record in hopfield_records])),
    }

    evaluation = {
        "retrieval_metrics": retrieval_metrics,
        "diagnostics": diagnostics,
        "baselines": baseline_comparison,
        "baseline_note": (
            "Классификационные базовые методы сохранены только как вторичный ориентир. "
            "Основной объект анализа в проекте — качество извлечения похожих случаев, а не подбор максимума по единственной метрике."
        ),
        "per_patient": per_patient,
        "noise_robustness": noise_robustness,
        "noise_note": (
            "Гауссов шум моделирует малые возмущения признакового вектора, а маскирование признаков имитирует потерю части контекста. "
            "Снижение top-k метрик при росте искажения интерпретируется как деградация устойчивости извлечения."
        ),
        "same_vs_cross_analysis": {
            "same_patient_top1_rate": diagnostics["same_patient_top1_rate"],
            "cross_patient_top1_rate": diagnostics["cross_patient_top1_rate"],
            "interpretation": (
                "Преобладание внутрипациентских совпадений ожидаемо, поскольку векторы содержат пациент-специфический контекст."
            ),
            "cross_patient_meaning": (
                "Межпациентские совпадения остаются содержательными, когда предпищевой профиль и контекст приема пищи указывают на сходную форму отклика."
            ),
            "limitation": (
                "Слишком высокая доля внутрипациентских совпадений ограничивает переносимость retrieval-подхода между пациентами."
            ),
        },
        "qualitative_examples": {
            "successes": successes[:5],
            "failures": failures[:5],
        },
        "failure_analysis": {
            "ambiguous_cases": ambiguous_cases[:5],
            "prototype_confusions": prototype_confusions[:5],
            "interpretation": (
                "Неудачные извлечения чаще возникают там, где gap между первыми весами мал, энтропия распределения внимания повышена, "
                "а прототипы частично перекрываются по признаковым блокам."
            ),
        },
        "limitations": [
            "В исследовании доступны только шесть пациентов OhioT1DM, поэтому межпациентская переносимость ограничена.",
            "Используемые метки являются детерминированными ретроспективными категориями, а не клинической истиной.",
            "Эксперимент проводится в ретроспективной постановке и не подтверждает применимость в реальном времени.",
            "Часть окон не содержит полных данных по частоте сердечных сокращений, поэтому носимый контекст неполон.",
            "Классификационные метрики остаются умеренными и не должны трактоваться как доказательство сильной предиктивной модели.",
            "Система не является клинической рекомендацией и не должна использоваться для выбора терапии или дозирования.",
        ],
    }
    chart_data = {
        "noise_robustness": noise_robustness,
        "per_patient": per_patient,
        "baseline_comparison": {
            name: {
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
            for name, metrics in baseline_comparison.items()
        },
    }
    return evaluation, chart_data


def evaluate_noise_robustness(
    test_windows: Sequence[dict[str, Any]],
    feature_matrix: np.ndarray,
    index_by_window_id: dict[str, int],
    memory_model: ContinuousHopfieldMemory,
    settings: Settings,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(7)
    results: list[dict[str, Any]] = []
    noise_levels = [0.0, 0.03, 0.06, 0.1]
    mask_levels = [0.0, 0.1, 0.2]

    for sigma in noise_levels:
        top1_scores = []
        top3_scores = []
        for window in test_windows:
            vector = feature_matrix[index_by_window_id[window["window_id"]]]
            noisy = l2_normalize((vector + rng.normal(0.0, sigma, size=vector.shape)).reshape(1, -1))[0]
            retrieval = memory_model.retrieve(noisy, k=settings.retrieval_top_k, beta=settings.hopfield_beta, steps=settings.recall_steps)
            labels = [item["label"] for item in retrieval["top_k"]]
            top1_scores.append(labels[0] == window["label"])
            top3_scores.append(window["label"] in labels[:3])
        results.append(
            {
                "mode": "gaussian_noise",
                "mode_display": "Гауссов шум",
                "level": sigma,
                "top1_accuracy": float(np.mean(top1_scores)),
                "top3_hit_rate": float(np.mean(top3_scores)),
            }
        )

    for ratio in mask_levels:
        top1_scores = []
        top3_scores = []
        for window in test_windows:
            vector = feature_matrix[index_by_window_id[window["window_id"]]].copy()
            if ratio > 0:
                mask_count = max(1, int(vector.size * ratio))
                indices = rng.choice(vector.size, size=mask_count, replace=False)
                vector[indices] = 0.0
            retrieval = memory_model.retrieve(vector, k=settings.retrieval_top_k, beta=settings.hopfield_beta, steps=settings.recall_steps)
            labels = [item["label"] for item in retrieval["top_k"]]
            top1_scores.append(labels[0] == window["label"])
            top3_scores.append(window["label"] in labels[:3])
        results.append(
            {
                "mode": "feature_mask",
                "mode_display": "Маскирование признаков",
                "level": ratio,
                "top1_accuracy": float(np.mean(top1_scores)),
                "top3_hit_rate": float(np.mean(top3_scores)),
            }
        )
    return results


def group_by_key(records: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    return grouped


def group_by_records(records: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    return group_by_key(records, key)


def make_dashboard(
    streams_by_patient: dict[str, PatientStreams],
    windows: Sequence[dict[str, Any]],
    encoder: FeatureEncoder,
    evaluation: dict[str, Any],
    exclusions: dict[str, int],
    settings: Settings,
) -> dict[str, Any]:
    usable_windows = [window for window in windows if window["usable_for_memory"]]
    label_distribution = Counter(window["label"] for window in usable_windows)
    split_distribution = Counter(window["split"] for window in usable_windows)
    retrieval_metrics = evaluation["retrieval_metrics"]
    return {
        "title": settings.project_name,
        "subtitle": "Ретроспективная исследовательская демонстрация ассоциативной памяти Хопфилда для извлечения похожих постпрандиальных случаев из OhioT1DM.",
        "disclaimer": "Интерфейс предназначен только для исследовательского анализа. Он не является клинической системой, не формирует рекомендации по терапии и не используется для расчета доз.",
        "patients_count": len(streams_by_patient),
        "total_meal_windows": len(windows),
        "usable_meal_windows": len(usable_windows),
        "memory_size": split_distribution.get("train", 0),
        "feature_dimension": len(encoder.feature_names),
        "headline_metrics": retrieval_metrics,
        "headline_summary": (
            f"На отложенных окнах память Хопфилда достигла {format_share(retrieval_metrics['top3_hit_rate'])} попаданий в top-3 "
            f"и {format_share(retrieval_metrics['top1_accuracy'])} совпадений метки в top-1; "
            "результат демонстрирует содержательность retrieval-подхода, но не подтверждает клиническую пригодность."
        ),
        "scientific_summary": [
            f"{len(streams_by_patient)} пациентов, {len(windows)} выделенных окон приема пищи.",
            f"{len(usable_windows)} окон пригодны для анализа памяти, {split_distribution.get('train', 0)} используются как элементы памяти.",
            f"{len(encoder.feature_names)} признаков описывают предпищевой CGM, контекст приема пищи, время, пациента и ЧСС.",
            f"Top-1 извлечение: {format_share(retrieval_metrics['top1_accuracy'])}; top-3 извлечение: {format_share(retrieval_metrics['top3_hit_rate'])}; MRR: {retrieval_metrics['mean_reciprocal_rank']:.3f}.",
        ],
        "interpretation_note": (
            "Что показывает этот блок: сводка фиксирует масштаб выборки, размер памяти и ключевые retrieval-метрики, "
            "которые следует интерпретировать как качество поиска похожих случаев, а не как клинический прогноз."
        ),
        "label_distribution": {label: label_distribution.get(label, 0) for label in sorted(label_distribution)},
        "split_distribution": dict(split_distribution),
        "exclusion_reasons": exclusions,
        "limitations_panel": evaluation["limitations"],
        "reused_visual_ideas": [
            "Левая навигационная колонка с спокойным ритмом карточек.",
            "Сдержанные контейнеры графиков с аккуратными границами и читаемыми отступами.",
            "Иерархия метрик и сравнительных таблиц в стиле исследовательской панели.",
        ],
        "not_reused_from_glucoscope": [
            "Не используются backend-модули и маршруты Glucoscope.",
            "Не используются код обработки приемов пищи, аналитика, RL-логика или XML-парсинг из Glucoscope.",
            "Не используется язык терапевтических рекомендаций и логика дозирования.",
        ],
    }


def _legacy_make_about_payload_unused(encoder: FeatureEncoder, settings: Settings) -> dict[str, Any]:
    return {
        "title": settings.project_name,
        "plain_language": "Проект кодирует постпрандиальные окна как фиксированные векторы признаков и извлекает похожие исторические случаи с помощью непрерывной ассоциативной памяти Хопфилда.",
        "why_memory_based": [
            "На малой ретроспективной выборке retrieval-подход позволяет опираться на конкретные исторические прецеденты, а не имитировать сильную клиническую предикцию.",
            "Процедура recall в памяти Хопфилда возвращает веса, энтропию и энергетическую траекторию, поэтому процесс извлечения остается наблюдаемым и интерпретируемым.",
        ],
        "why_retrieval_justified": [
            "При шести пациентах и ограниченном числе окон переусложнение модели повышает риск нестабильной оценки и переобучения.",
            "Похожие случаи и прототипы дают наблюдаемую связь между входным контекстом и исторической памятью системы.",
            "Retrieval-подход естественно поддерживает анализ сильных и неудачных случаев, а также исследование межпациентского сходства.",
        ],
        "why_not_classifier_zoo": [
            "Цель проекта состоит не в бесконечном подборе классификаторов, а в демонстрации ассоциативной памяти как исследовательского инструмента.",
            "Классификационные базовые методы сохранены только как вторичный ориентир и не определяют основную ценность интерфейса.",
        ],
        "vector_construction": {
            "feature_blocks": [
                "Значения CGM на фиксированной сетке от -90 до 0 минут перед приемом пищи.",
                "Отклонения CGM от базовой гликемии на той же сетке.",
                "Индикаторы пропусков и интерполяции для последовательности CGM.",
                "Контекст приема пищи: углеводы, болюс, базовая гликемия, тренды, вариабельность и активный базал.",
                "Временной контекст: циклическое кодирование времени суток и сегмент приема пищи.",
                "Контекст пациента: one-hot код идентификатора пациента.",
                "Контекст ЧСС: среднее, вариабельность, экстремумы и флаг отсутствия данных.",
            ],
            "feature_dimension": len(encoder.feature_names),
        },
        "hopfield_equations": {
            "similarity": "s_i = x_i · q",
            "weights": "w = softmax(beta * s)",
            "recall": "q_next = sum_i w_i x_i",
            "energy": "E(q) = -logsumexp(beta * Xq)/beta + 0.5 ||q||^2",
        },
        "safety_statements": [
            "Система не является клинической рекомендацией.",
            "Извлеченное сходство не эквивалентно терапевтическому совету.",
            "Интерфейс не предназначен для назначения инсулина или изменения лечения.",
        ],
        "limitations": [
            "Ретроспективные метки являются исследовательскими категориями, а не клинической истиной.",
            "Система извлекает запомненные паттерны и не предлагает изменение терапии.",
            "Выборка из шести пациентов не поддерживает сильные внешние обобщения.",
        ],
    }


def _legacy_generate_report_markdown_unused(
    dashboard: dict[str, Any],
    evaluation: dict[str, Any],
    prototypes: dict[str, dict[str, Any]],
    windows: Sequence[dict[str, Any]],
) -> str:
    retrieval_metrics = evaluation["retrieval_metrics"]
    diagnostics = evaluation["diagnostics"]
    failures = evaluation["qualitative_examples"]["failures"]
    successes = evaluation["qualitative_examples"]["successes"]
    failure_analysis = evaluation.get("failure_analysis", {})
    same_vs_cross = evaluation.get("same_vs_cross_analysis", {})

    lines = [
        f"# {dashboard['title']}",
        "",
        "## 1. Цель проекта",
        "Проект направлен на исследование того, может ли современная ассоциативная память Хопфилда использоваться как интерпретируемая система извлечения похожих постпрандиальных случаев на малой ретроспективной выборке OhioT1DM.",
        "",
        "## 2. Исходные данные",
        f"- Количество пациентов: {dashboard['patients_count']}",
        f"- Всего выделено окон приема пищи: {dashboard['total_meal_windows']}",
        f"- Пригодных для анализа окон: {dashboard['usable_meal_windows']}",
        f"- Размер памяти (train-окна): {dashboard['memory_size']}",
        f"- Размерность признакового пространства: {dashboard['feature_dimension']}",
        "",
        "## 3. Формирование окон приема пищи",
        "Каждое окно включает предпищевой контекст CGM на интервале от -90 до 0 минут и постпрандиальный ответ на интервале от 0 до +180 минут. Окна с перекрывающимися приемами пищи, отсутствующей базовой точкой или недостаточным покрытием CGM исключаются с явной фиксацией причины.",
        "",
        "## 4. Кодирование признаков",
        "Вектор признаков объединяет предпищевую форму CGM, отклонение от базовой гликемии, индикаторы пропусков, контекст приема пищи, временной контекст, идентификатор пациента и статистики ЧСС при наличии соответствующих данных.",
        "",
        "## 5. Ассоциативная память Хопфилда",
        "Векторы train-окна формируют непрерывную матрицу памяти Хопфилда. Для отложенного запроса выполняется итеративный recall с весами внимания, после чего анализируются top-k совпадения, траектория энергии и концентрация весов.",
        "",
        "## 6. Прототипы и их интерпретация",
    ]
    for label, prototype in prototypes.items():
        lines.append(
            f"- {display_label(label)}: поддержка {prototype['support_size']} окон ({format_share(prototype['support_fraction'])}), "
            f"чистота {prototype['purity']:.2f}, типичные углеводы {prototype['typical_context']['carbs']:.1f} г, "
            f"доминирующий сегмент {display_meal_segment(prototype['typical_context']['meal_segment_mode'])}. {prototype['interpretation_text']}"
        )

    lines.extend(
        [
            "",
            "## 7. Базовые методы сравнения",
            "В качестве вторичного ориентира использованы косинусное kNN-извлечение, ближайший прототип, пациент-специфический модальный класс и логистическая регрессия. Эти методы не образуют основную цель проекта и служат только для калибровки умеренности результатов.",
            "",
            "### Почему проект не является задачей бесконечного подбора классификаторов",
            "Ценность работы состоит в интерпретируемом retrieval-подходе: система показывает конкретные исторические случаи, прототипы и диагностические признаки неопределенности. Добавление большого числа классификаторов увеличило бы объем сравнений, но не улучшило бы объяснимость и защитимость проекта на малой выборке.",
            "",
            "## 8. Экспериментальные результаты",
            "Отложенная проверка строится хронологически по каждому пациенту. Основной акцент сделан на качестве извлечения, структуре прототипов, устойчивости к искажению входа и честной фиксации ограничений.",
            "",
            f"- Top-1 совпадение метки: {retrieval_metrics['top1_accuracy']:.3f}",
            f"- Попадание в top-3: {retrieval_metrics['top3_hit_rate']:.3f}",
            f"- Попадание в top-5: {retrieval_metrics['top5_hit_rate']:.3f}",
            f"- Средний reciprocal rank (MRR): {retrieval_metrics['mean_reciprocal_rank']:.3f}",
            f"- Среднее снижение энергии после recall-процедуры: {diagnostics['average_energy_drop']:.3f}",
            f"- Средняя энтропия распределения внимания: {diagnostics['average_attention_entropy']:.3f}",
            f"- Доля внутрипациентских top-1 совпадений: {same_vs_cross.get('same_patient_top1_rate', 0.0):.3f}",
            f"- Доля межпациентских top-1 совпадений: {same_vs_cross.get('cross_patient_top1_rate', 0.0):.3f}",
            "",
            "Краткая интерпретация: retrieval-подход демонстрирует умеренное, но содержательное качество поиска похожих случаев. Основная исследовательская ценность состоит в наблюдаемости памяти и возможности анализировать причины совпадений, а не в максимизации единственного классификационного числа.",
            "",
            "## 9. Анализ успешных и неудачных случаев",
            "Примеры корректных и некорректных извлечений используются для объяснения поведения модели на уровне отдельных окон.",
            "",
            "### Успешные случаи",
        ]
    )
    if successes:
        for success in successes[:3]:
            lines.append(
                f"- {success['window_id']}: истинная метка «{display_label(success['label'])}», "
                f"извлечение корректно на позиции 1, разрыв весов {success['top_weight_gap']:.3f}. {success['summary_text']}"
            )
    else:
        lines.append("- Явно выраженные успешные случаи в текущем отложенном срезе не выделены.")

    lines.extend(["", "### Неудачные и неоднозначные случаи"])
    if failures:
        for failure in failures[:3]:
            lines.append(
                f"- {failure['window_id']}: истинная метка «{display_label(failure['label'])}», top-1 случай относится к классу "
                f"«{display_label(failure['top_labels'][0])}», разрыв весов {failure['top_weight_gap']:.3f}. {failure['summary_text']}"
            )
    else:
        lines.append("- В текущем отложенном срезе не зарегистрированы ошибки top-1 извлечения.")

    lines.extend(["", "### Прототипные конфликты"])
    for example in failure_analysis.get("prototype_confusions", [])[:3]:
        lines.append(
            f"- {example['window_id']}: истинный класс «{display_label(example['label'])}», ближайший прототип "
            f"«{display_label(example['prototype_label'])}». Это указывает на частичное перекрытие классов в пространстве признаков."
        )

    lines.extend(
        [
            "",
            "## 10. Методологические ограничения",
            "Ограничения не скрываются и должны учитываться при защите проекта:",
        ]
    )
    for limitation in evaluation["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend(
        [
            "",
            "### Почему retrieval-подход методологически оправдан на малой выборке",
            "При ограниченном числе пациентов retrieval обеспечивает проверяемую связь между запросом и историческими окнами памяти, не требуя завышенных заявлений о генерализации. Такой подход лучше согласуется с учебной целью проекта: показать, как ассоциативная память структурирует сходство случаев и где именно эта структура начинает ошибаться.",
            "",
            "## 11. Заключение",
            "Проект демонстрирует, что ассоциативная память Хопфилда может быть успешно использована как ретроспективный инструмент извлечения похожих постпрандиальных случаев. Несмотря на умеренные классификационные показатели, система предоставляет защищаемую исследовательскую ценность за счет интерпретируемых top-k совпадений, прототипов, диагностик неопределенности и явного анализа неудачных случаев. При этом проект не является клинической системой и не должен использоваться как источник терапевтических рекомендаций.",
        ]
    )
    return "\n".join(lines)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _format_case_summary(example: dict[str, Any], model_label: str) -> str:
    predicted_labels = example.get("top_labels") or [example.get("predicted_label", "ambiguous")]
    predicted_label = display_label(predicted_labels[0])
    top_gap = float(example.get("top1_similarity", 0.0)) - float(example.get("top2_similarity", 0.0))
    return (
        f"- {model_label}: {example['window_id']}, истинная метка «{display_label(example['label'])}», "
        f"top-1 класс «{predicted_label}», разрыв top-1/top-2 {top_gap:.3f}, "
        f"доля внутрипациентских совпадений в top-5 {float(example.get('same_patient_rate', 0.0)):.2f}. "
        f"{example['summary_text']}"
    )


def make_about_payload(encoder: FeatureEncoder, settings: Settings) -> dict[str, Any]:
    return {
        "title": settings.project_name,
        "model_note": (
            "Интерфейс поддерживает два retrieval-движка: память Хопфилда по умолчанию и "
            "Siamese retrieval-модель как единственный нейросетевой сравнительный вариант. "
            "Оба режима используются только в ретроспективной исследовательской демонстрации."
        ),
        "plain_language": (
            "Проект кодирует постпрандиальные окна как фиксированные представления и извлекает "
            "похожие исторические случаи либо через непрерывную ассоциативную память Хопфилда, "
            "либо через Siamese-энкодер метрического пространства."
        ),
        "why_memory_based": [
            "На малой ретроспективной выборке retrieval-подход позволяет опираться на конкретные исторические прецеденты, а не имитировать сильную клиническую предикцию.",
            "Память Хопфилда возвращает веса, энтропию и энергетическую траекторию recall-процедуры, поэтому механизм извлечения остается наблюдаемым.",
            "Siamese-модель добавляет нейросетевое пространство эмбеддингов, но сохраняет ту же задачу: поиск похожих случаев, а не соревнование классификаторов.",
        ],
        "why_retrieval_justified": [
            "При шести пациентах и ограниченном числе окон переусложнение модели повышает риск нестабильной оценки и переобучения.",
            "Похожие случаи и прототипы дают наблюдаемую связь между входным контекстом и исторической памятью системы.",
            "Retrieval-подход естественно поддерживает анализ сильных и неудачных случаев, исследование межпациентского сходства и диагностику неопределенности.",
        ],
        "why_not_classifier_zoo": [
            "Цель проекта состоит не в бесконечном подборе классификаторов, а в сравнении двух интерпретируемых retrieval-представлений на одной и той же задаче.",
            "Классификационные базовые методы сохранены только как вторичный ориентир и не определяют основную ценность интерфейса.",
            "Siamese-модель включена как единственный нейросетевой retrieval-сравнитель, чтобы показать альтернативное метрическое пространство без перегрузки проекта.",
        ],
        "vector_construction": {
            "feature_blocks": [
                "Значения CGM на фиксированной сетке от -90 до 0 минут перед приемом пищи.",
                "Отклонения CGM от базовой гликемии на той же сетке.",
                "Индикаторы пропусков и интерполяции для последовательности CGM.",
                "Контекст приема пищи: углеводы, болюс, базовая гликемия, тренды, вариабельность и активный базал.",
                "Временной контекст: циклическое кодирование времени суток и сегмент приема пищи.",
                "Контекст пациента: one-hot код идентификатора пациента.",
                "Контекст ЧСС: среднее, вариабельность, экстремумы и флаг отсутствия данных.",
            ],
            "feature_dimension": len(encoder.feature_names),
        },
        "hopfield_equations": {
            "similarity": "s_i = x_i · q",
            "weights": "w = softmax(beta * s)",
            "recall": "q_next = sum_i w_i x_i",
            "energy": "E(q) = -logsumexp(beta * Xq)/beta + 0.5 ||q||^2",
        },
        "safety_statements": [
            "Система не является клинической рекомендацией.",
            "Извлеченное сходство не эквивалентно терапевтическому совету.",
            "Интерфейс не предназначен для назначения инсулина или изменения лечения.",
            "Сходство между случаями не должно интерпретироваться как совет по терапии.",
        ],
        "limitations": [
            "Ретроспективные метки являются исследовательскими категориями, а не клинической истиной.",
            "Система извлекает запомненные паттерны и не предлагает изменение терапии.",
            "Выборка из шести пациентов не поддерживает сильные внешние обобщения.",
            "Контекст ЧСС доступен не для всех окон и остается неполным.",
            "Даже улучшение retrieval-метрик у Siamese-модели не снимает ограничений малой выборки и retrospective-only постановки.",
        ],
    }


def generate_report_markdown(
    dashboard: dict[str, Any],
    evaluation: dict[str, Any],
    prototypes: dict[str, dict[str, Any]],
    windows: Sequence[dict[str, Any]],
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()
    _ = windows
    retrieval_metrics = evaluation["retrieval_metrics"]
    diagnostics = evaluation["diagnostics"]
    failures = evaluation["qualitative_examples"]["failures"]
    successes = evaluation["qualitative_examples"]["successes"]
    failure_analysis = evaluation.get("failure_analysis", {})
    same_vs_cross = evaluation.get("same_vs_cross_analysis", {})

    siamese_evaluation = _load_optional_json(settings.siamese_metrics_path) or {}
    siamese_config = _load_optional_json(settings.siamese_config_path) or {}
    siamese_prototypes = _load_optional_json(settings.siamese_prototypes_path) or {}
    siamese_retrieval = siamese_evaluation.get("retrieval_metrics", {})
    siamese_diagnostics = siamese_evaluation.get("diagnostics", {})
    siamese_examples = siamese_evaluation.get("qualitative_examples", {})
    siamese_failure_analysis = siamese_evaluation.get("failure_analysis", {})
    siamese_same_vs_cross = siamese_evaluation.get("same_vs_cross_analysis", {})
    siamese_available = bool(siamese_evaluation)

    baseline_labels = {
        "cosine_knn": "косинусное kNN-извлечение",
        "nearest_prototype": "ближайший прототип",
        "patient_majority": "пациент-специфический модальный класс",
        "logistic_regression": "логистическая регрессия",
    }
    baseline_summary = ", ".join(
        baseline_labels[key] for key in baseline_labels if key in evaluation.get("baselines", {})
    )

    lines = [
        f"# {settings.project_name}",
        "",
        "## 1. Цель проекта",
        (
            "Проект посвящен retrieval-first анализу постпрандиальных окон OhioT1DM. Базовым "
            "движком остается ассоциативная память Хопфилда, а в качестве единственного "
            "нейросетевого сравнения добавлена Siamese retrieval-модель с временным энкодером "
            "и общим пространством эмбеддингов. Цель состоит не в расширении набора "
            "классификаторов, а в сопоставлении двух способов извлечения похожих случаев, "
            "интерпретации прототипов и анализа неопределенности."
        ),
        "",
        "## 2. Исходные данные",
        f"- Количество пациентов: {dashboard['patients_count']}",
        f"- Всего выделено окон приема пищи: {dashboard['total_meal_windows']}",
        f"- Пригодных для анализа окон: {dashboard['usable_meal_windows']}",
        f"- Размер памяти train-окон: {dashboard['memory_size']}",
        f"- Размерность исходного признакового пространства: {dashboard['feature_dimension']}",
        "",
        "## 3. Формирование окон приема пищи",
        (
            "Каждое окно содержит предпищевой контекст CGM на интервале от -90 до 0 минут и "
            "постпрандиальный отклик на интервале от 0 до +180 минут. Окна с перекрывающимися "
            "приемами пищи, отсутствующей базовой точкой или недостаточным покрытием CGM "
            "исключаются с явной фиксацией причины. Разделение на train/val/test выполняется "
            "хронологически по пациентам, чтобы минимизировать утечку информации."
        ),
        "",
        "## 4. Кодирование признаков",
        (
            "Общее представление окна объединяет предпищевую форму CGM, отклонение от базовой "
            "гликемии, маску пропусков, контекст приема пищи, временной контекст, идентификатор "
            "пациента и статистики ЧСС при наличии соответствующих данных. Это представление "
            "используется как для памяти Хопфилда, так и для формирования входов Siamese-модели."
        ),
        "",
        "## 5. Ассоциативная память Хопфилда",
        (
            "Векторы train-окон формируют непрерывную матрицу памяти Хопфилда. Для отложенного "
            "запроса выполняется итеративный recall с весами внимания, после чего анализируются "
            "top-k совпадения, траектория энергии, концентрация весов и сходство по блокам "
            "признаков. Этот режим остается моделью по умолчанию и основной точкой интерпретации."
        ),
        "",
        "## 6. Сиамская retrieval-модель",
        (
            "Siamese-модель реализована как компактный temporal encoder: 1D-CNN по предпищевой "
            "последовательности CGM с каналами исходного сигнала, отклонения от базового уровня "
            "и маски пропусков, а также MLP по табличному контексту окна. После слияния ветвей "
            "получается нормализованный эмбеддинг, а retrieval выполняется по косинусному "
            "сходству в пространстве эмбеддингов."
        ),
    ]
    if siamese_available:
        lines.extend(
            [
                f"- Размерность пространства эмбеддингов: {int(siamese_config.get('embedding_dim', settings.siamese_embedding_dim))}",
                (
                    "Модель обучается в offline-режиме на train-памяти с supervised contrastive "
                    "loss; метки используются только для организации метрического пространства, а "
                    "не для переопределения проекта в классификационный бенчмарк."
                ),
            ]
        )
    else:
        lines.append(
            "- Артефакты Siamese-модели отсутствуют; при пересборке проекта этот раздел автоматически заполняется после offline-обучения."
        )

    lines.extend(
        [
            "",
            "## 7. Прототипы и их интерпретация",
            (
                "В проекте используются два типа прототипов. Для Хопфилда это центры в исходном "
                "признаковом пространстве, связанные с усредненным профилем окна. Для "
                "Siamese-модели это средние точки в пространстве представлений, описывающие "
                "регион эмбеддингов, а не буквальный средний набор исходных признаков."
            ),
        ]
    )
    all_prototype_labels = list(dict.fromkeys([*prototypes.keys(), *siamese_prototypes.keys()]))
    for label in all_prototype_labels:
        hopfield_proto = prototypes.get(label)
        siamese_proto = siamese_prototypes.get(label)
        fragments: list[str] = []
        if hopfield_proto:
            fragments.append(
                f"Хопфилд-прототип: поддержка {hopfield_proto['support_size']} окон ({format_share(hopfield_proto['support_fraction'])}), "
                f"чистота {hopfield_proto['purity']:.2f}, типичные углеводы {hopfield_proto['typical_context']['carbs']:.1f} г, "
                f"доминирующий сегмент {display_meal_segment(hopfield_proto['typical_context']['meal_segment_mode'])}."
            )
        if siamese_proto:
            fragments.append(
                f"Siamese-прототип: поддержка {siamese_proto['support_size']} окон ({format_share(siamese_proto['support_fraction'])}), "
                f"чистота {siamese_proto['purity']:.2f} в пространстве эмбеддингов."
            )
        interpretation_text = ""
        if hopfield_proto:
            interpretation_text = hopfield_proto["interpretation_text"]
        elif siamese_proto:
            interpretation_text = siamese_proto.get("interpretation_text", "")
        lines.append(f"- {display_label(label)}. {' '.join(fragments)} {interpretation_text}".strip())

    lines.extend(
        [
            "",
            "## 8. Базовые методы сравнения",
            (
                "Вторичными ориентирами сохранены только простые методы сравнения: "
                f"{baseline_summary}. Эти методы используются для калибровки умеренности "
                "результатов и не определяют основную структуру интерфейса."
            ),
            "",
            "### Почему проект не является задачей бесконечного подбора классификаторов",
            (
                "Главная ценность работы состоит в наблюдаемом процессе извлечения похожих "
                "случаев, прототипов и диагностик неопределенности. Добавление большого числа "
                "классификаторов увеличило бы объем сравнений, но ухудшило бы методологическую "
                "чистоту и объяснимость на малой ретроспективной выборке."
            ),
            "",
            "### Почему retrieval-подход методологически оправдан на малой выборке",
            (
                "При ограниченном числе пациентов retrieval обеспечивает проверяемую связь между "
                "запросом и историческими окнами памяти. Такой подход позволяет обсуждать сильные, "
                "слабые и неоднозначные случаи без завышенных заявлений о генерализации или "
                "клинической пригодности."
            ),
            "",
            "## 9. Экспериментальные результаты",
            "Сравнение моделей носит исследовательский характер и не предназначено для клинической интерпретации.",
            "",
            f"- Хопфилд: top-1 {retrieval_metrics['top1_accuracy']:.3f}, top-3 {retrieval_metrics['top3_hit_rate']:.3f}, top-5 {retrieval_metrics['top5_hit_rate']:.3f}, MRR {retrieval_metrics['mean_reciprocal_rank']:.3f}.",
            f"- Хопфилд: среднее снижение энергии {diagnostics['average_energy_drop']:.3f}, средняя энтропия {diagnostics['average_attention_entropy']:.3f}, внутрипациентские top-1 {same_vs_cross.get('same_patient_top1_rate', 0.0):.3f}, межпациентские top-1 {same_vs_cross.get('cross_patient_top1_rate', 0.0):.3f}.",
        ]
    )
    if siamese_available:
        lines.extend(
            [
                f"- Siamese: top-1 {siamese_retrieval.get('top1_accuracy', 0.0):.3f}, top-3 {siamese_retrieval.get('top3_hit_rate', 0.0):.3f}, top-5 {siamese_retrieval.get('top5_hit_rate', 0.0):.3f}, MRR {siamese_retrieval.get('mean_reciprocal_rank', 0.0):.3f}.",
                f"- Siamese: средний gap top-1/top-2 {siamese_diagnostics.get('average_similarity_gap', 0.0):.3f}, средняя энтропия {siamese_diagnostics.get('average_attention_entropy', 0.0):.3f}, внутрипациентские top-1 {siamese_same_vs_cross.get('same_patient_top1_rate', 0.0):.3f}, межпациентские top-1 {siamese_same_vs_cross.get('cross_patient_top1_rate', 0.0):.3f}.",
                (
                    "На текущем отложенном срезе Siamese-модель дает более высокие "
                    "retrieval-метрики, чем Hopfield, однако это преимущество следует трактовать "
                    "как сравнение двух представлений на одной и той же малой выборке, а не как "
                    "доказательство универсального превосходства."
                ),
            ]
        )
    else:
        lines.append("- Siamese-метрики будут добавлены в отчёт после offline-обучения и пересборки артефактов.")
    lines.extend(
        [
            (
                "Общая интерпретация: обе модели пригодны для научно-исследовательской демонстрации "
                "similar-case retrieval. Хопфилд подчеркивает recall-динамику и энергию, а "
                "Siamese-модель показывает альтернативное метрическое пространство без перехода к "
                "classifier-zoo постановке."
            ),
            "",
            "## 10. Анализ успешных и неудачных случаев",
            "Ниже приведены характерные примеры, показывающие как устойчивые совпадения, так и пограничные или ошибочные retrieval-ситуации.",
            "",
            "### Успешные случаи",
        ]
    )
    if successes:
        lines.append(_format_case_summary(successes[0], "Хопфилд"))
    else:
        lines.append("- Хопфилд: явно выраженные успешные случаи в текущем срезе не выделены.")
    siamese_successes = siamese_examples.get("successes", [])
    if siamese_successes:
        lines.append(_format_case_summary(siamese_successes[0], "Сиамская retrieval-модель"))
    elif siamese_available:
        lines.append("- Сиамская retrieval-модель: явно выраженные успешные случаи в текущем срезе не выделены.")

    lines.extend(["", "### Неудачные и неоднозначные случаи"])
    if failures:
        lines.append(_format_case_summary(failures[0], "Хопфилд"))
    else:
        lines.append("- Хопфилд: в текущем срезе не зарегистрированы ошибки top-1 извлечения.")
    siamese_failures = siamese_examples.get("failures", [])
    if siamese_failures:
        lines.append(_format_case_summary(siamese_failures[0], "Сиамская retrieval-модель"))
    elif siamese_available:
        lines.append("- Сиамская retrieval-модель: явные ошибки top-1 извлечения в текущем срезе не выделены.")

    lines.extend(["", "### Прототипные конфликты"])
    hopfield_confusions = failure_analysis.get("prototype_confusions", [])
    siamese_confusions = siamese_failure_analysis.get("prototype_confusions", [])
    if hopfield_confusions:
        example = hopfield_confusions[0]
        lines.append(
            f"- Хопфилд: {example['window_id']}, истинный класс «{display_label(example['label'])}», "
            f"ближайший прототип «{display_label(example['prototype_label'])}». Это указывает на частичное перекрытие классов в исходном признаковом пространстве."
        )
    if siamese_confusions:
        example = siamese_confusions[0]
        lines.append(
            f"- Сиамская retrieval-модель: {example['window_id']}, истинный класс «{display_label(example['label'])}», "
            f"ближайший прототип «{display_label(example['prototype_label'])}». Конфликт показывает неоднозначную позицию окна в пространстве эмбеддингов."
        )
    if not hopfield_confusions and not siamese_confusions:
        lines.append("- Явно выраженные прототипные конфликты в текущем срезе не выделены.")

    lines.extend(
        [
            "",
            "## 11. Методологические ограничения",
            "Ограничения не скрываются и должны учитываться при интерпретации результатов и защите проекта:",
        ]
    )
    for limitation in evaluation["limitations"]:
        lines.append(f"- {limitation}")
    if siamese_available:
        lines.append(
            "- Siamese-энкодер обучается на той же малой train-памяти, поэтому его преимущества чувствительны к split, случайной инициализации и объему доступных примеров."
        )

    lines.extend(
        [
            "",
            "## 12. Заключение",
            (
                "Проект демонстрирует retrieval-first подход к анализу постпрандиальных окон: "
                "память Хопфилда обеспечивает наблюдаемый recall с энергетической диагностикой, а "
                "Siamese-модель добавляет нейросетевое метрическое пространство для того же класса "
                "задач. Основной вклад состоит в интерпретируемом извлечении похожих случаев, "
                "анализе прототипов и честной фиксации ограничений. Проект не является "
                "клинической системой и не должен использоваться как источник терапевтических "
                "рекомендаций."
            ),
        ]
    )
    return "\n".join(lines)


def refresh_report_documents(bundle: dict[str, Any], settings: Settings) -> str:
    report_markdown = generate_coursework_report(bundle, settings)
    settings.latest_report_path.write_text(report_markdown, encoding="utf-8")
    settings.coursework_report_path.write_text(report_markdown, encoding="utf-8")
    return report_markdown


def build_runtime_bundle(force: bool = False, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.coursework_report_path.parent.mkdir(parents=True, exist_ok=True)

    if settings.runtime_bundle_path.exists() and not force:
        with settings.runtime_bundle_path.open("rb") as handle:
            return pickle.load(handle)

    streams_by_patient = parse_ohio_directory(settings.data_dir)
    windows, exclusions = extract_windows(streams_by_patient, settings)
    windows = assign_splits(windows)
    memory_windows = [window for window in windows if window["usable_for_memory"]]
    if not memory_windows:
        raise RuntimeError("Из OhioT1DM не удалось извлечь пригодные окна приема пищи.")

    encoder, feature_matrix = FeatureEncoder.fit(memory_windows)
    index_by_window_id = {window["window_id"]: index for index, window in enumerate(memory_windows)}
    for window in windows:
        window["memory_index"] = index_by_window_id.get(window["window_id"], None)

    train_windows = [window for window in memory_windows if window["split"] == "train"]
    train_indices = [index_by_window_id[window["window_id"]] for window in train_windows]
    memory_model = ContinuousHopfieldMemory().fit(
        feature_matrix[train_indices],
        [{"window_id": window["window_id"], "label": window["label"], "patient_id": window["patient_id"]} for window in train_windows],
    )
    prototypes = build_prototypes(train_windows, feature_matrix, encoder, index_by_window_id)
    evaluation, chart_data = evaluate_models(windows, feature_matrix, encoder, index_by_window_id, memory_model, prototypes, settings)
    dashboard = make_dashboard(streams_by_patient, windows, encoder, evaluation, exclusions, settings)
    about = make_about_payload(encoder, settings)

    bundle = {
        "generated_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "settings": settings.as_public_dict(),
        "dashboard": dashboard,
        "windows": [json_ready(window) for window in windows],
        "memory_window_ids": [window["window_id"] for window in memory_windows],
        "index_by_window_id": index_by_window_id,
        "feature_matrix": feature_matrix,
        "encoder": encoder,
        "memory_model": memory_model,
        "prototypes": json_ready(prototypes),
        "evaluation": json_ready(evaluation),
        "chart_data": json_ready(chart_data),
        "about": about,
    }

    summary_df = pd.DataFrame(
        [
            {
                key: json_ready(value)
                for key, value in window.items()
                if key
                not in {"premeal_values", "premeal_delta", "premeal_missingness", "full_curve_minutes", "full_curve_values", "full_curve_missingness"}
            }
            for window in windows
        ]
    )
    summary_df.to_csv(settings.windows_dataset_path, index=False)
    settings.windows_json_path.write_text(json.dumps(bundle["windows"], indent=2), encoding="utf-8")
    np.save(settings.feature_matrix_path, feature_matrix)
    settings.feature_metadata_path.write_text(
        json.dumps(
            {
                "feature_names": encoder.feature_names,
                "block_slices": encoder.block_slices,
                "patient_ids": encoder.patient_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    memory_model.save(settings.models_dir / "hopfield_memory")
    settings.latest_metrics_path.write_text(
        json.dumps({"dashboard": dashboard, "evaluation": bundle["evaluation"]}, indent=2),
        encoding="utf-8",
    )
    settings.chart_data_path.write_text(json.dumps(bundle["chart_data"], indent=2), encoding="utf-8")
    refresh_report_documents(bundle, settings)
    with settings.runtime_bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return bundle
