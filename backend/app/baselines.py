from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from .config import Settings
from .memory import l2_normalize
from .pipeline import classification_metrics, display_label


def compute_noise_stability(points: list[dict[str, Any]]) -> float | None:
    if not points:
        return None
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        by_mode.setdefault(str(point["mode"]), []).append(point)

    retentions: list[float] = []
    for mode_points in by_mode.values():
        ordered = sorted(mode_points, key=lambda item: float(item["level"]))
        clean = ordered[0]
        clean_score = (float(clean["top1_accuracy"]) + float(clean["top3_hit_rate"])) / 2.0
        if clean_score <= 0:
            continue
        for point in ordered[1:]:
            score = (float(point["top1_accuracy"]) + float(point["top3_hit_rate"])) / 2.0
            retentions.append(score / clean_score)
    if not retentions:
        return None
    return float(np.mean(retentions))


def _dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    n = left.size
    m = right.size
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(left[i - 1]) - float(right[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def _soft_dtw_distance(left: np.ndarray, right: np.ndarray, gamma: float = 0.2) -> float:
    n = left.size
    m = right.size
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    def softmin(values: tuple[float, float, float]) -> float:
        scaled = np.asarray(values, dtype=float) / gamma
        minimum = np.min(scaled)
        return float(-gamma * (np.log(np.exp(-(scaled - minimum)).sum()) + minimum))

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (float(left[i - 1]) - float(right[j - 1])) ** 2
            dp[i, j] = cost + softmin((dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]))
    return float(dp[n, m])


def _sequence_noise(points: np.ndarray, mode: str, level: float, rng: np.random.Generator) -> np.ndarray:
    noisy = points.copy()
    if mode == "gaussian_noise" and level > 0:
        noisy += rng.normal(0.0, level, size=noisy.shape)
    elif mode == "feature_mask" and level > 0:
        mask = rng.random(noisy.shape) < level
        noisy[mask] = 0.0
    return noisy


def _topk_metrics(true_label: str, top_labels: list[str]) -> dict[str, float | bool]:
    rank = next((index + 1 for index, label in enumerate(top_labels) if label == true_label), None)
    return {
        "top1_correct": bool(top_labels and top_labels[0] == true_label),
        "top3_hit": true_label in top_labels[:3],
        "top5_hit": true_label in top_labels[:5],
        "mrr": 1.0 / rank if rank else 0.0,
        "label_purity_top5": float(np.mean([label == true_label for label in top_labels])) if top_labels else 0.0,
    }


@dataclass
class BaselineInputs:
    train_windows: list[dict[str, Any]]
    test_windows: list[dict[str, Any]]
    train_matrix: np.ndarray
    test_matrix: np.ndarray
    train_sequences: np.ndarray
    test_sequences: np.ndarray
    prototype_labels: list[str]
    prototype_vectors: np.ndarray


def _baseline_inputs(bundle: dict[str, Any]) -> BaselineInputs:
    windows = bundle["windows"]
    index_by_window_id = bundle["index_by_window_id"]
    feature_matrix = np.asarray(bundle["feature_matrix"], dtype=float)
    train_windows = [window for window in windows if window["split"] == "train" and window["usable_for_memory"]]
    test_windows = [window for window in windows if window["split"] == "test" and window["usable_for_memory"]]
    train_indices = [index_by_window_id[window["window_id"]] for window in train_windows]
    test_indices = [index_by_window_id[window["window_id"]] for window in test_windows]

    prototype_labels = list(bundle["prototypes"].keys())
    prototype_vectors = np.asarray([bundle["prototypes"][label]["vector"] for label in prototype_labels], dtype=float) if prototype_labels else np.empty((0, feature_matrix.shape[1]), dtype=float)

    return BaselineInputs(
        train_windows=train_windows,
        test_windows=test_windows,
        train_matrix=feature_matrix[train_indices],
        test_matrix=feature_matrix[test_indices],
        train_sequences=np.asarray([window["premeal_values"] for window in train_windows], dtype=float),
        test_sequences=np.asarray([window["premeal_values"] for window in test_windows], dtype=float),
        prototype_labels=prototype_labels,
        prototype_vectors=prototype_vectors,
    )


def _evaluate_cosine_knn(inputs: BaselineInputs, settings: Settings) -> dict[str, Any]:
    train_norm = l2_normalize(inputs.train_matrix)
    predictions: list[str] = []
    y_true: list[str] = []
    records: list[dict[str, Any]] = []

    for query_window, query_vector in zip(inputs.test_windows, inputs.test_matrix):
        similarities = train_norm @ l2_normalize(query_vector.reshape(1, -1))[0]
        top_indices = np.argsort(similarities)[::-1][: settings.retrieval_top_k]
        top_labels = [inputs.train_windows[int(index)]["label"] for index in top_indices]
        metrics = _topk_metrics(query_window["label"], top_labels)
        predictions.append(top_labels[0])
        y_true.append(query_window["label"])
        records.append(metrics)

    baseline = classification_metrics(y_true, predictions, sorted(set(y_true)))
    noise_points = []
    rng = np.random.default_rng(settings.random_seed)
    for mode, levels in (("gaussian_noise", [0.0, 0.05, 0.1, 0.15]), ("feature_mask", [0.0, 0.1, 0.2])):
        for level in levels:
            top1_hits = []
            top3_hits = []
            for query_window, query_vector in zip(inputs.test_windows, inputs.test_matrix):
                noisy_vector = _sequence_noise(query_vector, mode, float(level), rng)
                similarities = train_norm @ l2_normalize(noisy_vector.reshape(1, -1))[0]
                top_indices = np.argsort(similarities)[::-1][: settings.retrieval_top_k]
                top_labels = [inputs.train_windows[int(index)]["label"] for index in top_indices]
                top1_hits.append(bool(top_labels and top_labels[0] == query_window["label"]))
                top3_hits.append(query_window["label"] in top_labels[:3])
            noise_points.append(
                {
                    "mode": mode,
                    "level": float(level),
                    "top1_accuracy": float(np.mean(top1_hits)) if top1_hits else 0.0,
                    "top3_hit_rate": float(np.mean(top3_hits)) if top3_hits else 0.0,
                }
            )

    return {
        "key": "cosine_knn",
        "label": "Cosine kNN",
        "family": "baseline",
        "available": True,
        "top1_accuracy": float(np.mean([record["top1_correct"] for record in records])) if records else 0.0,
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in records])) if records else 0.0,
        "mean_reciprocal_rank": float(np.mean([record["mrr"] for record in records])) if records else 0.0,
        "noise_stability": compute_noise_stability(noise_points),
        "additional_metrics": {
            "top5_hit_rate": float(np.mean([record["top5_hit"] for record in records])) if records else 0.0,
            "label_purity_top5": float(np.mean([record["label_purity_top5"] for record in records])) if records else 0.0,
            "balanced_accuracy": baseline["balanced_accuracy"],
            "macro_f1": baseline["macro_f1"],
        },
        "noise_points": noise_points,
        "notes": "Прямое kNN по cosine similarity в стандартизованном признаковом пространстве.",
    }


def _evaluate_dtw_family(inputs: BaselineInputs, settings: Settings, *, soft: bool) -> dict[str, Any]:
    distance_fn = _soft_dtw_distance if soft else _dtw_distance
    key = "soft_dtw_knn" if soft else "dtw_knn"
    label = "Soft-DTW kNN" if soft else "DTW kNN"
    predictions: list[str] = []
    y_true: list[str] = []
    records: list[dict[str, Any]] = []

    for query_window, query_sequence in zip(inputs.test_windows, inputs.test_sequences):
        distances = np.asarray([distance_fn(query_sequence, sequence) for sequence in inputs.train_sequences], dtype=float)
        top_indices = np.argsort(distances)[: settings.retrieval_top_k]
        top_labels = [inputs.train_windows[int(index)]["label"] for index in top_indices]
        metrics = _topk_metrics(query_window["label"], top_labels)
        predictions.append(top_labels[0])
        y_true.append(query_window["label"])
        records.append(metrics)

    baseline = classification_metrics(y_true, predictions, sorted(set(y_true)))
    noise_points = []
    rng = np.random.default_rng(settings.random_seed)
    levels = [0.0, 0.05, 0.1]
    for mode in ("gaussian_noise", "feature_mask"):
        for level in levels:
            top1_hits = []
            top3_hits = []
            for query_window, query_sequence in zip(inputs.test_windows, inputs.test_sequences):
                noisy_sequence = _sequence_noise(query_sequence, mode, float(level), rng)
                distances = np.asarray([distance_fn(noisy_sequence, sequence) for sequence in inputs.train_sequences], dtype=float)
                top_indices = np.argsort(distances)[: settings.retrieval_top_k]
                top_labels = [inputs.train_windows[int(index)]["label"] for index in top_indices]
                top1_hits.append(bool(top_labels and top_labels[0] == query_window["label"]))
                top3_hits.append(query_window["label"] in top_labels[:3])
            noise_points.append(
                {
                    "mode": mode,
                    "level": float(level),
                    "top1_accuracy": float(np.mean(top1_hits)) if top1_hits else 0.0,
                    "top3_hit_rate": float(np.mean(top3_hits)) if top3_hits else 0.0,
                }
            )

    return {
        "key": key,
        "label": label,
        "family": "baseline",
        "available": True,
        "top1_accuracy": float(np.mean([record["top1_correct"] for record in records])) if records else 0.0,
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in records])) if records else 0.0,
        "mean_reciprocal_rank": float(np.mean([record["mrr"] for record in records])) if records else 0.0,
        "noise_stability": compute_noise_stability(noise_points),
        "additional_metrics": {
            "top5_hit_rate": float(np.mean([record["top5_hit"] for record in records])) if records else 0.0,
            "label_purity_top5": float(np.mean([record["label_purity_top5"] for record in records])) if records else 0.0,
            "balanced_accuracy": baseline["balanced_accuracy"],
            "macro_f1": baseline["macro_f1"],
        },
        "noise_points": noise_points,
        "notes": "Последовательностный baseline по предпищевому CGM-сегменту.",
    }


def _evaluate_nearest_prototype(inputs: BaselineInputs) -> dict[str, Any]:
    if inputs.prototype_vectors.size == 0:
        return {
            "key": "nearest_prototype",
            "label": "Nearest prototype",
            "family": "baseline",
            "available": False,
            "top1_accuracy": None,
            "top3_hit_rate": None,
            "mean_reciprocal_rank": None,
            "noise_stability": None,
            "additional_metrics": {},
            "notes": "Прототипы недоступны.",
        }

    prototype_norm = l2_normalize(inputs.prototype_vectors)
    predictions: list[str] = []
    y_true: list[str] = []
    records: list[dict[str, Any]] = []
    for query_window, query_vector in zip(inputs.test_windows, inputs.test_matrix):
        similarities = prototype_norm @ l2_normalize(query_vector.reshape(1, -1))[0]
        order = np.argsort(similarities)[::-1]
        top_labels = [inputs.prototype_labels[int(index)] for index in order[: min(5, len(order))]]
        metrics = _topk_metrics(query_window["label"], top_labels)
        predictions.append(top_labels[0])
        y_true.append(query_window["label"])
        records.append(metrics)
    baseline = classification_metrics(y_true, predictions, sorted(set(y_true)))
    return {
        "key": "nearest_prototype",
        "label": "Nearest prototype",
        "family": "baseline",
        "available": True,
        "top1_accuracy": float(np.mean([record["top1_correct"] for record in records])) if records else 0.0,
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in records])) if records else 0.0,
        "mean_reciprocal_rank": float(np.mean([record["mrr"] for record in records])) if records else 0.0,
        "noise_stability": None,
        "additional_metrics": {
            "top5_hit_rate": float(np.mean([record["top5_hit"] for record in records])) if records else 0.0,
            "label_purity_top5": float(np.mean([record["label_purity_top5"] for record in records])) if records else 0.0,
            "balanced_accuracy": baseline["balanced_accuracy"],
            "macro_f1": baseline["macro_f1"],
        },
        "notes": "Сравнение с усредненными prototype-векторами train-памяти.",
    }


def _unavailable_kshape() -> dict[str, Any]:
    return {
        "key": "k_shape",
        "label": "k-Shape",
        "family": "baseline",
        "available": False,
        "top1_accuracy": None,
        "top3_hit_rate": None,
        "mean_reciprocal_rank": None,
        "noise_stability": None,
        "additional_metrics": {},
        "notes": "Baseline не собран в текущем artifact pipeline и не включен в UI как полноценный режим.",
    }


def evaluate_retrieval_baselines(bundle: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    inputs = _baseline_inputs(bundle)
    rows = [
        _evaluate_cosine_knn(inputs, settings),
        _evaluate_dtw_family(inputs, settings, soft=False),
        _evaluate_dtw_family(inputs, settings, soft=True),
        _evaluate_nearest_prototype(inputs),
        _unavailable_kshape(),
    ]
    return rows


def load_cached_baselines(settings: Settings) -> list[dict[str, Any]] | None:
    if not settings.comparison_metrics_path.exists():
        return None
    return json.loads(settings.comparison_metrics_path.read_text(encoding="utf-8"))


def save_baselines_cache(settings: Settings, rows: list[dict[str, Any]]) -> None:
    settings.comparison_metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def label_distribution_summary(distribution: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "label": display_label(key),
            "value": int(value),
        }
        for key, value in sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    ]
