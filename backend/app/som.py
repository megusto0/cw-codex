from __future__ import annotations

from collections import Counter
import json
import pickle
from typing import Any, Sequence

import numpy as np

from .config import Settings
from .memory import l2_normalize
from .pipeline import classification_metrics, display_label, display_meal_segment, format_share, group_by_records, json_ready


SOM_DESCRIPTOR = {
    "key": "som",
    "label": "Карта Кохонена",
    "scientific_description": "Карта самоорганизации для топологического поиска сходных случаев.",
    "short_description": "Топологическое neighborhood-based retrieval в пространстве признаков.",
    "representation_name": "Топологическая карта",
    "prototype_name": "Локальная область карты",
    "similarity_name": "Локальное сходство на карте",
    "supports_iterative_recall": False,
}


def _grid_coordinates(height: int, width: int) -> np.ndarray:
    return np.asarray([(row, col) for row in range(height) for col in range(width)], dtype=float)


def _train_som(train_matrix: np.ndarray, settings: Settings) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(settings.random_seed)
    height = settings.som_grid_height
    width = settings.som_grid_width
    codebook_size = height * width
    coordinates = _grid_coordinates(height, width)

    replace = train_matrix.shape[0] < codebook_size
    initial_indices = rng.choice(train_matrix.shape[0], size=codebook_size, replace=replace)
    weights = train_matrix[initial_indices].astype(np.float64).copy()

    history: list[dict[str, float]] = []
    for epoch in range(settings.som_epochs):
        order = rng.permutation(train_matrix.shape[0])
        progress = epoch / max(settings.som_epochs - 1, 1)
        learning_rate = settings.som_learning_rate * (0.25 + 0.75 * (1.0 - progress))
        sigma = max(0.6, settings.som_sigma * (0.2 + 0.8 * (1.0 - progress)))

        for sample_index in order:
            sample = train_matrix[sample_index]
            distances = np.linalg.norm(weights - sample, axis=1)
            bmu_index = int(np.argmin(distances))
            grid_distances = np.sum((coordinates - coordinates[bmu_index]) ** 2, axis=1)
            neighborhood = np.exp(-grid_distances / (2.0 * (sigma ** 2)))
            weights += learning_rate * neighborhood[:, None] * (sample - weights)

        epoch_distances = np.linalg.norm(weights[None, :, :] - train_matrix[:, None, :], axis=2)
        quantization_error = float(np.mean(np.min(epoch_distances, axis=1)))
        history.append(
            {
                "epoch": float(epoch + 1),
                "learning_rate": float(learning_rate),
                "sigma": float(sigma),
                "quantization_error": quantization_error,
            }
        )

    return weights.astype(np.float32), history


def _bmu_distances(weights: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.linalg.norm(weights - vector[None, :], axis=1)


def _bmu_index(weights: np.ndarray, vector: np.ndarray) -> int:
    return int(np.argmin(_bmu_distances(weights, vector)))


def _second_bmu_index(weights: np.ndarray, vector: np.ndarray) -> int:
    distances = _bmu_distances(weights, vector)
    order = np.argsort(distances)
    return int(order[1]) if len(order) > 1 else int(order[0])


def _neighboring_cells(cell_index: int, grid_shape: tuple[int, int]) -> list[int]:
    height, width = grid_shape
    row = cell_index // width
    col = cell_index % width
    cells: list[int] = []
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            next_row = row + delta_row
            next_col = col + delta_col
            if 0 <= next_row < height and 0 <= next_col < width:
                cells.append(next_row * width + next_col)
    return sorted(set(cells))


def _adjacent_cells(left: int, right: int, grid_shape: tuple[int, int]) -> bool:
    height, width = grid_shape
    left_row, left_col = divmod(left, width)
    right_row, right_col = divmod(right, width)
    return abs(left_row - right_row) <= 1 and abs(left_col - right_col) <= 1


def _map_distance(cell_a: int, cell_b: int, grid_shape: tuple[int, int]) -> float:
    _, width = grid_shape
    row_a, col_a = divmod(cell_a, width)
    row_b, col_b = divmod(cell_b, width)
    return float(abs(row_a - row_b) + abs(col_a - col_b))


def _cell_statistics(
    *,
    train_windows: Sequence[dict[str, Any]],
    train_matrix: np.ndarray,
    weights: np.ndarray,
    train_assignments: np.ndarray,
    grid_shape: tuple[int, int],
) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    height, width = grid_shape
    codebook_size = height * width
    for cell_index in range(codebook_size):
        member_indices = np.where(train_assignments == cell_index)[0]
        member_windows = [train_windows[int(index)] for index in member_indices]
        label_counts = Counter(window["label"] for window in member_windows)
        dominant_label = label_counts.most_common(1)[0][0] if label_counts else None
        purity = (label_counts[dominant_label] / len(member_windows)) if dominant_label and member_windows else 0.0
        representative_ids: list[str] = []
        if len(member_indices):
            member_vectors = train_matrix[member_indices]
            distances = np.linalg.norm(member_vectors - weights[cell_index][None, :], axis=1)
            representative_order = np.argsort(distances)[:3]
            representative_ids = [member_windows[int(index)]["window_id"] for index in representative_order]

        stats[cell_index] = {
            "cell_index": cell_index,
            "row": cell_index // width,
            "col": cell_index % width,
            "count": int(len(member_windows)),
            "dominant_label": dominant_label,
            "dominant_label_display": display_label(dominant_label) if dominant_label else "Пустая ячейка",
            "purity": float(purity),
            "label_distribution": {label: int(count) for label, count in sorted(label_counts.items())},
            "representative_window_ids": representative_ids,
        }
    return stats


def _summarize_neighbor(
    *,
    query_window: dict[str, Any],
    candidate_window: dict[str, Any],
    same_cell: bool,
    map_distance: float,
) -> str:
    if same_cell:
        base = "Тот же локальный узел карты."
    elif map_distance <= 1.0:
        base = "Соседняя топологическая область."
    else:
        base = "Более удаленная область карты."

    if str(query_window["patient_id"]) == str(candidate_window["patient_id"]):
        return f"{base} Совпадение внутри того же пациента."
    return f"{base} Межпациентское совпадение."


def _retrieve_som_neighbors(
    *,
    query_window: dict[str, Any],
    query_vector: np.ndarray,
    weights: np.ndarray,
    train_matrix: np.ndarray,
    train_norm: np.ndarray,
    train_windows: Sequence[dict[str, Any]],
    train_assignments: np.ndarray,
    cell_stats: dict[int, dict[str, Any]],
    grid_shape: tuple[int, int],
    k: int,
) -> dict[str, Any]:
    query_norm = l2_normalize(query_vector.reshape(1, -1))[0]
    query_bmu = _bmu_index(weights, query_vector)
    query_second_bmu = _second_bmu_index(weights, query_vector)
    quantization_error = float(np.linalg.norm(query_vector - weights[query_bmu]))
    topographic_error = 0.0 if _adjacent_cells(query_bmu, query_second_bmu, grid_shape) else 1.0

    feature_similarity = np.clip((train_norm @ query_norm + 1.0) / 2.0, 0.0, 1.0)
    map_distances = np.asarray([_map_distance(query_bmu, int(cell_index), grid_shape) for cell_index in train_assignments], dtype=float)
    locality = np.exp(-0.9 * map_distances)
    local_similarity = feature_similarity * locality
    order = np.argsort(local_similarity)[::-1][:k]

    dominant_cell = cell_stats.get(query_bmu, {})
    dominant_label = dominant_cell.get("dominant_label")
    cluster_purity = float(dominant_cell.get("purity", 0.0))
    local_density = float(dominant_cell.get("count", 0)) / max(len(train_windows), 1)

    neighbors: list[dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        candidate_window = train_windows[int(index)]
        cell_index = int(train_assignments[int(index)])
        same_patient = str(candidate_window["patient_id"]) == str(query_window["patient_id"])
        neighbors.append(
            {
                "rank": rank,
                "window_id": candidate_window["window_id"],
                "label": candidate_window["label"],
                "label_display": candidate_window.get("label_display", display_label(candidate_window["label"])),
                "patient_id": candidate_window["patient_id"],
                "same_patient": same_patient,
                "relation_badge": "Тот же пациент" if same_patient else "Другой пациент",
                "cell_index": cell_index,
                "map_distance": float(map_distances[int(index)]),
                "similarity": float(local_similarity[int(index)]),
                "feature_similarity": float(feature_similarity[int(index)]),
                "reason": _summarize_neighbor(
                    query_window=query_window,
                    candidate_window=candidate_window,
                    same_cell=cell_index == query_bmu,
                    map_distance=float(map_distances[int(index)]),
                ),
                "window": candidate_window,
            }
        )

    local_region_cells = _neighboring_cells(query_bmu, grid_shape)
    local_label_counts = Counter()
    for cell_index in local_region_cells:
        for label, count in cell_stats.get(cell_index, {}).get("label_distribution", {}).items():
            local_label_counts[label] += count

    return {
        "query_bmu": query_bmu,
        "quantization_error": quantization_error,
        "topographic_error": topographic_error,
        "cluster_purity": cluster_purity,
        "local_density": local_density,
        "dominant_label": dominant_label,
        "dominant_label_display": display_label(dominant_label) if dominant_label else "Пустая область",
        "neighbors": neighbors,
        "local_region_cells": local_region_cells,
        "local_label_distribution": {label: int(count) for label, count in sorted(local_label_counts.items())},
    }


def evaluate_som_model(
    *,
    base_bundle: dict[str, Any],
    weights: np.ndarray,
    train_assignments: np.ndarray,
    cell_stats: dict[int, dict[str, Any]],
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    grid_shape = (settings.som_grid_height, settings.som_grid_width)
    windows = base_bundle["windows"]
    index_by_window_id = base_bundle["index_by_window_id"]
    feature_matrix = np.asarray(base_bundle["feature_matrix"], dtype=float)
    train_windows = [window for window in windows if window["split"] == "train" and window["usable_for_memory"]]
    test_windows = [window for window in windows if window["split"] == "test" and window["usable_for_memory"]]
    train_indices = [index_by_window_id[window["window_id"]] for window in train_windows]
    train_matrix = feature_matrix[train_indices]
    train_norm = l2_normalize(train_matrix)

    case_records: list[dict[str, Any]] = []
    predictions: list[str] = []
    y_true: list[str] = []

    for query_window in test_windows:
        query_vector = feature_matrix[index_by_window_id[query_window["window_id"]]]
        retrieval = _retrieve_som_neighbors(
            query_window=query_window,
            query_vector=query_vector,
            weights=weights,
            train_matrix=train_matrix,
            train_norm=train_norm,
            train_windows=train_windows,
            train_assignments=train_assignments,
            cell_stats=cell_stats,
            grid_shape=grid_shape,
            k=settings.retrieval_top_k,
        )
        top_labels = [item["label"] for item in retrieval["neighbors"]]
        top_patients = [item["patient_id"] for item in retrieval["neighbors"]]
        rank = next((index + 1 for index, label in enumerate(top_labels) if label == query_window["label"]), None)
        top_similarity_gap = (
            retrieval["neighbors"][0]["similarity"] - retrieval["neighbors"][1]["similarity"]
            if len(retrieval["neighbors"]) > 1
            else retrieval["neighbors"][0]["similarity"]
            if retrieval["neighbors"]
            else 0.0
        )
        predicted_label = top_labels[0] if top_labels else retrieval["dominant_label"] or "ambiguous"

        case_records.append(
            {
                "window_id": query_window["window_id"],
                "patient_id": query_window["patient_id"],
                "label": query_window["label"],
                "top_ids": [item["window_id"] for item in retrieval["neighbors"]],
                "top_labels": top_labels,
                "top_patients": top_patients,
                "top1_correct": bool(top_labels and top_labels[0] == query_window["label"]),
                "top3_hit": query_window["label"] in top_labels[:3],
                "top5_hit": query_window["label"] in top_labels[:5],
                "mrr": 1.0 / rank if rank else 0.0,
                "label_purity_top5": float(np.mean([label == query_window["label"] for label in top_labels])) if top_labels else 0.0,
                "same_patient_rate": float(np.mean([patient == query_window["patient_id"] for patient in top_patients])) if top_patients else 0.0,
                "cross_patient_top1": bool(top_patients and top_patients[0] != query_window["patient_id"]),
                "top1_similarity": retrieval["neighbors"][0]["similarity"] if retrieval["neighbors"] else 0.0,
                "top2_similarity": retrieval["neighbors"][1]["similarity"] if len(retrieval["neighbors"]) > 1 else 0.0,
                "top_weight_gap": top_similarity_gap,
                "attention_entropy": 0.0,
                "energy_before": 0.0,
                "energy_after": 0.0,
                "energy_drop": 0.0,
                "predicted_label": predicted_label,
                "prototype_label": retrieval["dominant_label"] or predicted_label,
                "prototype_top_gap": float(retrieval["cluster_purity"]),
                "prototype_distribution": retrieval["local_label_distribution"],
                "summary_text": (
                    f"Окно попадает в область карты с доминирующей меткой «{retrieval['dominant_label_display']}»; "
                    f"локальная плотность {retrieval['local_density']:.3f}, квантование {retrieval['quantization_error']:.3f}."
                ),
                "quantization_error": retrieval["quantization_error"],
                "topographic_error": retrieval["topographic_error"],
                "cluster_purity": retrieval["cluster_purity"],
                "local_density": retrieval["local_density"],
                "bmu_index": retrieval["query_bmu"],
            }
        )
        predictions.append(predicted_label)
        y_true.append(query_window["label"])

    per_patient = []
    for patient_id, records in group_by_records(case_records, "patient_id").items():
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

    retrieval_metrics = {
        "top1_accuracy": float(np.mean([record["top1_correct"] for record in case_records])) if case_records else 0.0,
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in case_records])) if case_records else 0.0,
        "top5_hit_rate": float(np.mean([record["top5_hit"] for record in case_records])) if case_records else 0.0,
        "mean_reciprocal_rank": float(np.mean([record["mrr"] for record in case_records])) if case_records else 0.0,
        "label_purity_top5": float(np.mean([record["label_purity_top5"] for record in case_records])) if case_records else 0.0,
        "prototype_purity": float(np.mean([record["cluster_purity"] for record in case_records])) if case_records else 0.0,
        "nearest_neighbor_consistency": float(np.mean([record["top1_correct"] for record in case_records])) if case_records else 0.0,
    }
    diagnostics = {
        "average_quantization_error": float(np.mean([record["quantization_error"] for record in case_records])) if case_records else 0.0,
        "average_topographic_error": float(np.mean([record["topographic_error"] for record in case_records])) if case_records else 0.0,
        "average_cluster_purity": float(np.mean([record["cluster_purity"] for record in case_records])) if case_records else 0.0,
        "average_local_density": float(np.mean([record["local_density"] for record in case_records])) if case_records else 0.0,
        "same_patient_top1_rate": float(np.mean([not record["cross_patient_top1"] for record in case_records])) if case_records else 0.0,
        "cross_patient_top1_rate": float(np.mean([record["cross_patient_top1"] for record in case_records])) if case_records else 0.0,
    }

    baseline_metrics = {
        "som": classification_metrics(y_true, predictions, sorted(set(y_true))),
    }

    successes = sorted(
        [record for record in case_records if record["top1_correct"]],
        key=lambda item: (item["cluster_purity"], -item["quantization_error"]),
        reverse=True,
    )
    failures = sorted(
        [record for record in case_records if not record["top1_correct"]],
        key=lambda item: (item["quantization_error"], item["cluster_purity"]),
    )
    ambiguous_cases = sorted(case_records, key=lambda item: (item["cluster_purity"], -item["topographic_error"]))
    topology_confusions = sorted(
        [record for record in case_records if record["prototype_label"] != record["label"]],
        key=lambda item: (item["cluster_purity"], item["quantization_error"]),
    )

    noise_robustness = evaluate_som_noise(
        base_bundle=base_bundle,
        weights=weights,
        train_assignments=train_assignments,
        cell_stats=cell_stats,
        settings=settings,
    )

    evaluation = {
        "retrieval_metrics": retrieval_metrics,
        "diagnostics": diagnostics,
        "baselines": baseline_metrics,
        "baseline_note": (
            "SOM используется как топологическая retrieval-модель; классификационные показатели здесь вторичны и нужны только для сопоставления."
        ),
        "per_patient": per_patient,
        "noise_robustness": noise_robustness,
        "noise_note": (
            "Шум и маскирование показывают, насколько устойчиво окно сохраняет положение на карте и состав локального neighborhood."
        ),
        "same_vs_cross_analysis": {
            "same_patient_top1_rate": diagnostics["same_patient_top1_rate"],
            "cross_patient_top1_rate": diagnostics["cross_patient_top1_rate"],
            "interpretation": "Карта Кохонена организует окна по соседству на карте, а не по явной recall-динамике или обученному эмбеддингу.",
            "cross_patient_meaning": "Межпациентские совпадения на карте полезны, когда похожие траектории попадают в общую локальную область.",
            "limitation": "Локальная топология может смешивать редкие классы, если карта слишком компактна для малой, но неоднородной выборки.",
        },
        "qualitative_examples": {"successes": successes[:5], "failures": failures[:5]},
        "failure_analysis": {
            "ambiguous_cases": ambiguous_cases[:5],
            "prototype_confusions": topology_confusions[:5],
            "interpretation": "Неудачные случаи чаще появляются на границах локальных областей карты или в ячейках со смешанным составом меток.",
        },
        "limitations": [
            "Карта самоорганизации чувствительна к выбору размеров сетки и не гарантирует идеальное разделение редких классов.",
            "SOM строится на той же малой ретроспективной выборке OhioT1DM, поэтому структура карты не должна трактоваться как устойчивая клиническая таксономия.",
            "Система не является медицинским изделием и не используется для выбора терапии или дозы инсулина.",
        ],
    }

    chart_data = {
        "noise_robustness": noise_robustness,
        "per_patient": per_patient,
        "cells": list(cell_stats.values()),
    }
    return evaluation, chart_data


def evaluate_som_noise(
    *,
    base_bundle: dict[str, Any],
    weights: np.ndarray,
    train_assignments: np.ndarray,
    cell_stats: dict[int, dict[str, Any]],
    settings: Settings,
) -> list[dict[str, Any]]:
    grid_shape = (settings.som_grid_height, settings.som_grid_width)
    rng = np.random.default_rng(settings.random_seed)
    feature_matrix = np.asarray(base_bundle["feature_matrix"], dtype=float)
    index_by_window_id = base_bundle["index_by_window_id"]
    windows = base_bundle["windows"]
    train_windows = [window for window in windows if window["split"] == "train" and window["usable_for_memory"]]
    test_windows = [window for window in windows if window["split"] == "test" and window["usable_for_memory"]]
    train_indices = [index_by_window_id[window["window_id"]] for window in train_windows]
    train_matrix = feature_matrix[train_indices]
    train_norm = l2_normalize(train_matrix)

    results: list[dict[str, Any]] = []
    levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    for mode in ("gaussian_noise", "feature_mask"):
        for level in levels:
            top1_hits: list[bool] = []
            top3_hits: list[bool] = []
            for window in test_windows:
                vector = feature_matrix[index_by_window_id[window["window_id"]]].copy()
                if mode == "gaussian_noise" and level > 0:
                    vector += rng.normal(0.0, level, size=vector.shape)
                elif mode == "feature_mask" and level > 0:
                    mask = rng.random(vector.shape) < level
                    vector[mask] = 0.0

                retrieval = _retrieve_som_neighbors(
                    query_window=window,
                    query_vector=vector,
                    weights=weights,
                    train_matrix=train_matrix,
                    train_norm=train_norm,
                    train_windows=train_windows,
                    train_assignments=train_assignments,
                    cell_stats=cell_stats,
                    grid_shape=grid_shape,
                    k=settings.retrieval_top_k,
                )
                top_labels = [item["label"] for item in retrieval["neighbors"]]
                top1_hits.append(bool(top_labels and top_labels[0] == window["label"]))
                top3_hits.append(window["label"] in top_labels[:3])

            results.append(
                {
                    "mode": mode,
                    "mode_display": "Гауссов шум" if mode == "gaussian_noise" else "Маскирование признаков",
                    "level": float(level),
                    "top1_accuracy": float(np.mean(top1_hits)) if top1_hits else 0.0,
                    "top3_hit_rate": float(np.mean(top3_hits)) if top3_hits else 0.0,
                }
            )
    return results


def build_som_bundle(
    base_bundle: dict[str, Any],
    settings: Settings,
    force: bool = False,
) -> dict[str, Any]:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    if settings.som_runtime_bundle_path.exists() and not force:
        with settings.som_runtime_bundle_path.open("rb") as handle:
            return pickle.load(handle)

    windows = base_bundle["windows"]
    index_by_window_id = base_bundle["index_by_window_id"]
    feature_matrix = np.asarray(base_bundle["feature_matrix"], dtype=np.float32)
    train_windows = [window for window in windows if window["split"] == "train" and window["usable_for_memory"]]
    train_indices = [index_by_window_id[window["window_id"]] for window in train_windows]
    train_matrix = feature_matrix[train_indices]

    weights, history = _train_som(train_matrix.astype(np.float64), settings)
    train_assignments = np.asarray([_bmu_index(weights, vector) for vector in train_matrix], dtype=int)
    cell_stats = _cell_statistics(
        train_windows=train_windows,
        train_matrix=train_matrix,
        weights=weights,
        train_assignments=train_assignments,
        grid_shape=(settings.som_grid_height, settings.som_grid_width),
    )
    evaluation, chart_data = evaluate_som_model(
        base_bundle=base_bundle,
        weights=weights,
        train_assignments=train_assignments,
        cell_stats=cell_stats,
        settings=settings,
    )

    artifact_bundle = {
        "descriptor": SOM_DESCRIPTOR,
        "config": {
            "grid_height": settings.som_grid_height,
            "grid_width": settings.som_grid_width,
            "epochs": settings.som_epochs,
            "learning_rate": settings.som_learning_rate,
            "sigma": settings.som_sigma,
        },
        "weights": weights,
        "history": history,
        "train_window_ids": [window["window_id"] for window in train_windows],
        "train_assignments": train_assignments.tolist(),
        "cell_stats": json_ready(cell_stats),
        "evaluation": json_ready(evaluation),
        "chart_data": json_ready(chart_data),
    }

    np.save(settings.som_weights_path, weights)
    settings.som_assignments_path.write_text(
        json.dumps(
            {
                "train_window_ids": artifact_bundle["train_window_ids"],
                "train_assignments": artifact_bundle["train_assignments"],
                "cell_stats": artifact_bundle["cell_stats"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    settings.som_metrics_path.write_text(json.dumps(artifact_bundle["evaluation"], indent=2), encoding="utf-8")
    settings.som_report_path.write_text(generate_som_report(base_bundle, artifact_bundle["evaluation"], artifact_bundle["config"]), encoding="utf-8")
    with settings.som_runtime_bundle_path.open("wb") as handle:
        pickle.dump(artifact_bundle, handle)
    return artifact_bundle


def load_som_bundle(settings: Settings, force: bool = False) -> dict[str, Any]:
    if force or not settings.som_runtime_bundle_path.exists():
        raise FileNotFoundError("SOM runtime bundle not found. Rebuild artifacts with python -m app.build.")
    with settings.som_runtime_bundle_path.open("rb") as handle:
        return pickle.load(handle)


def generate_som_report(base_bundle: dict[str, Any], evaluation: dict[str, Any], config: dict[str, Any]) -> str:
    dashboard = base_bundle["dashboard"]
    retrieval = evaluation["retrieval_metrics"]
    diagnostics = evaluation["diagnostics"]
    lines = [
        "# Карта Кохонена",
        "",
        "## Постановка",
        "Карта Кохонена используется как топологическая retrieval-модель для поиска сходных постпрандиальных CGM-окон на общей признаковой матрице OhioT1DM.",
        "",
        "## Конфигурация",
        f"- Размер карты: {config['grid_height']} x {config['grid_width']}",
        f"- Эпохи: {config['epochs']}",
        f"- Начальная скорость обучения: {config['learning_rate']}",
        f"- Начальная ширина neighborhood: {config['sigma']}",
        "",
        "## Данные",
        f"- Пациенты: {dashboard['patients_count']}",
        f"- Выделенные окна: {dashboard['total_meal_windows']}",
        f"- Пригодные окна: {dashboard['usable_meal_windows']}",
        f"- Train-память: {dashboard['memory_size']}",
        "",
        "## Retrieval-метрики",
        f"- Top-1: {retrieval['top1_accuracy']:.3f}",
        f"- Top-3: {retrieval['top3_hit_rate']:.3f}",
        f"- MRR: {retrieval['mean_reciprocal_rank']:.3f}",
        f"- Квантование: {diagnostics['average_quantization_error']:.3f}",
        f"- Топографическая ошибка: {diagnostics['average_topographic_error']:.3f}",
        f"- Локальная чистота: {diagnostics['average_cluster_purity']:.3f}",
        "",
        "## Интерпретация",
        (
            "Карта Кохонена полезна не как сильный предиктор, а как способ увидеть локальные области признакового пространства и проверить, "
            "сосредоточены ли похожие постпрандиальные окна в соседних ячейках карты."
        ),
        "",
        "## Ограничения",
        "- Малый размер выборки и редкие классы затрудняют стабильную интерпретацию топологии.",
        "- Топологическое соседство не следует трактовать как клиническую близость или основание для терапевтических решений.",
    ]
    return "\n".join(lines)
