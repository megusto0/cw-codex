from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import inspect
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .baselines import compute_noise_stability
from .config import Settings
from .memory import l2_normalize
from .pipeline import display_label, json_ready
from .siamese import (
    _encode_in_batches,
    _sequence_and_tabular_views,
    train_siamese_encoder,
)
from .som import (
    _bmu_index,
    _cell_statistics,
    _map_distance,
    _retrieve_som_neighbors,
    _train_som,
)


SEED_SET = (7, 13, 29)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _share(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _mean_std(values: Iterable[float]) -> dict[str, float | None]:
    data = [float(value) for value in values]
    if not data:
        return {"mean": None, "std": None}
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
    }


def _topk_metrics(true_label: str, top_labels: list[str]) -> tuple[bool, bool, float]:
    rank = next((index + 1 for index, label in enumerate(top_labels) if label == true_label), None)
    return bool(top_labels and top_labels[0] == true_label), true_label in top_labels[:3], 1.0 / rank if rank else 0.0


def summarize_corruption_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not points:
        return []

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        by_mode.setdefault(str(point["mode"]), []).append(point)

    target_levels = {
        "feature_mask": [0.1, 0.2],
        "gaussian_noise": [0.05, 0.1],
    }
    level_names = {
        ("feature_mask", 0.1): "Маскирование 10%",
        ("feature_mask", 0.2): "Маскирование 20%",
        ("gaussian_noise", 0.05): "Гауссово возмущение: малое",
        ("gaussian_noise", 0.1): "Гауссово возмущение: среднее",
    }

    rows: list[dict[str, Any]] = []
    for mode, mode_points in by_mode.items():
        ordered = sorted(mode_points, key=lambda item: float(item["level"]))
        clean = next((item for item in ordered if float(item["level"]) == 0.0), ordered[0])
        clean_top1 = float(clean["top1_accuracy"])
        clean_top3 = float(clean["top3_hit_rate"])
        for target in target_levels.get(mode, []):
            point = min(ordered, key=lambda item: abs(float(item["level"]) - target))
            target_label = level_names.get((mode, target), f"{mode} {target:.2f}")
            top1 = float(point["top1_accuracy"])
            top3 = float(point["top3_hit_rate"])
            rows.append(
                {
                    "mode": mode,
                    "mode_display": str(point.get("mode_display") or ("Маскирование признаков" if mode == "feature_mask" else "Гауссов шум")),
                    "level": float(point["level"]),
                    "label": target_label if abs(float(point["level"]) - target) <= 0.03 else f"{target_label} (уровень {float(point['level']):.2f})",
                    "top1_clean": clean_top1,
                    "top3_clean": clean_top3,
                    "top1_corrupted": top1,
                    "top3_corrupted": top3,
                    "top1_drop": clean_top1 - top1,
                    "top3_drop": clean_top3 - top3,
                    "top1_retention": (top1 / clean_top1) if clean_top1 > 0 else None,
                    "top3_retention": (top3 / clean_top3) if clean_top3 > 0 else None,
                }
            )
    return rows


def primary_corruption_retention(points: list[dict[str, Any]]) -> float | None:
    summary = summarize_corruption_points(points)
    preferred = next((row for row in summary if row["mode"] == "feature_mask" and abs(float(row["level"]) - 0.1) < 1e-6), None)
    if preferred:
        return preferred.get("top1_retention")
    retentions = [row.get("top1_retention") for row in summary if row.get("top1_retention") is not None]
    return float(np.mean(retentions)) if retentions else None


def _model_rows(bundle: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    hopfield_eval = bundle["evaluation"]
    siamese_eval = _load_json(settings.siamese_metrics_path) or {}
    som_eval = _load_json(settings.som_metrics_path) or {}

    rows = [
        {
            "key": "hopfield",
            "label": "Память Хопфилда",
            "family": "neural",
            "available": True,
            "retrieval_metrics": hopfield_eval.get("retrieval_metrics", {}),
            "diagnostics": hopfield_eval.get("diagnostics", {}),
            "noise_points": hopfield_eval.get("noise_robustness", []),
            "evaluation": hopfield_eval,
        },
        {
            "key": "siamese_temporal",
            "label": "Сиамская temporal-модель",
            "family": "neural",
            "available": bool(siamese_eval),
            "retrieval_metrics": siamese_eval.get("retrieval_metrics", {}),
            "diagnostics": siamese_eval.get("diagnostics", {}),
            "noise_points": siamese_eval.get("noise_robustness", []),
            "evaluation": siamese_eval,
        },
        {
            "key": "som",
            "label": "Карта Кохонена",
            "family": "neural",
            "available": bool(som_eval),
            "retrieval_metrics": som_eval.get("retrieval_metrics", {}),
            "diagnostics": som_eval.get("diagnostics", {}),
            "noise_points": som_eval.get("noise_robustness", []),
            "evaluation": som_eval,
        },
    ]

    for row in rows:
        metrics = row["retrieval_metrics"]
        row["top1_accuracy"] = metrics.get("top1_accuracy")
        row["top3_hit_rate"] = metrics.get("top3_hit_rate")
        row["mean_reciprocal_rank"] = metrics.get("mean_reciprocal_rank")
        row["noise_stability"] = compute_noise_stability(row["noise_points"])
        row["corruption_retention_top1_10"] = primary_corruption_retention(row["noise_points"])
        row["robustness_summary"] = summarize_corruption_points(row["noise_points"])
    return json_ready(rows)


def _baseline_rows(settings: Settings, baseline_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = baseline_rows if baseline_rows is not None else (_load_json(settings.comparison_metrics_path) or [])
    for row in rows:
        row["robustness_summary"] = summarize_corruption_points(row.get("noise_points", []))
        row["corruption_retention_top1_10"] = primary_corruption_retention(row.get("noise_points", []))
    return json_ready(rows)


def _compact_example(record: dict[str, Any], *, category: str) -> dict[str, Any]:
    top_labels = record.get("top_labels") or []
    top_patients = record.get("top_patients") or []
    true_label = str(record.get("label", "unknown"))
    retrieved_label = str(top_labels[0] if top_labels else record.get("predicted_label") or record.get("prototype_label") or "unknown")
    same_patient = bool(top_patients and str(top_patients[0]) == str(record.get("patient_id")))
    gap = None
    if record.get("top1_similarity") is not None and record.get("top2_similarity") is not None:
        gap = float(record["top1_similarity"]) - float(record["top2_similarity"])
    elif record.get("top_weight_gap") is not None:
        gap = float(record["top_weight_gap"])

    if category == "failure":
        difficulty = "Ближайший случай имеет другую исследовательскую метку; это указывает на перекрытие локальных паттернов."
    elif category == "ambiguous":
        difficulty = "Разрыв между первыми соседями мал, поэтому retrieval не следует считать устойчивым."
    elif category == "same_patient":
        difficulty = "Top-1 находится у того же пациента; такой пример полезен, но ограничивает вывод о межпациентской переносимости."
    else:
        difficulty = "Top-1 принадлежит другому пациенту; пример показывает возможное межпациентское сходство, но не является клинической рекомендацией."

    return {
        "query_id": record.get("window_id"),
        "true_label": true_label,
        "true_label_display": display_label(true_label),
        "retrieved_label": retrieved_label,
        "retrieved_label_display": display_label(retrieved_label),
        "top1_patient_relation": "same" if same_patient else "cross",
        "top1_top2_gap": gap,
        "short_explanation": record.get("summary_text") or difficulty,
        "why_difficult": difficulty,
    }


def curate_failure_analysis(model_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in model_rows:
        evaluation = model.get("evaluation") or {}
        examples = evaluation.get("qualitative_examples") or {}
        failure = evaluation.get("failure_analysis") or {}
        pool = list(examples.get("successes") or []) + list(examples.get("failures") or []) + list(failure.get("ambiguous_cases") or [])
        same_patient = [item for item in pool if item.get("top_patients") and str(item["top_patients"][0]) == str(item.get("patient_id"))]
        cross_patient = [item for item in pool if item.get("top_patients") and str(item["top_patients"][0]) != str(item.get("patient_id"))]
        output[model["key"]] = {
            "label": model["label"],
            "top1_failures": [_compact_example(item, category="failure") for item in (examples.get("failures") or [])[:3]],
            "ambiguous_cases": [_compact_example(item, category="ambiguous") for item in (failure.get("ambiguous_cases") or [])[:3]],
            "same_patient_dominant": [_compact_example(item, category="same_patient") for item in same_patient[:3]],
            "cross_patient_meaningful": [_compact_example(item, category="cross_patient") for item in cross_patient[:3]],
        }
    return output


def patient_generalization_summary(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes = {
        "hopfield": "Хопфилд остаётся более межпациентским: top-1 чаще приходит от другого пациента, что полезно для анализа общих паттернов, но качество совпадений умеренное.",
        "siamese_temporal": "Siamese-модель чаще извлекает внутри того же пациента; это может повышать локальное сходство, но требует осторожности при обсуждении переносимости.",
        "som": "SOM следует трактовать через локальную топологию: внутри- и межпациентские совпадения отражают состав соседних ячеек карты, а не клиническую близость.",
    }
    rows = []
    for model in model_rows:
        diagnostics = model.get("diagnostics") or {}
        rows.append(
            {
                "key": model["key"],
                "label": model["label"],
                "same_patient_top1_rate": diagnostics.get("same_patient_top1_rate"),
                "cross_patient_top1_rate": diagnostics.get("cross_patient_top1_rate"),
                "interpretation": notes.get(model["key"], "Наблюдение ограничено текущим ретроспективным split."),
            }
        )
    return rows


def _comparison_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    lines = [
        "# Сводка comparative evaluation",
        "",
        f"- Сгенерировано: {summary['generated_at']}",
        f"- Пациенты: {dataset['patients']}",
        f"- Выделенные окна: {dataset['extracted_windows']}",
        f"- Пригодные retrieval-окна: {dataset['usable_windows']}",
        f"- Train-память: {dataset['memory_windows']}",
        "",
        "## Основные retrieval-метрики",
        "| Модель | Top-1 | Top-3 | MRR | Сохранение Top-1 при 10% маскировании |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary["models"]:
        lines.append(
            f"| {row['label']} | {_share(row.get('top1_accuracy'))} | {_share(row.get('top3_hit_rate'))} | "
            f"{_score(row.get('mean_reciprocal_rank'))} | {_share(row.get('corruption_retention_top1_10'))} |"
        )

    lines.extend(["", "## Evaluation-only baselines"])
    for row in summary["baselines"]:
        if not row.get("available"):
            lines.append(f"- {row['label']}: недоступен. {row.get('notes', '')}")
            continue
        lines.append(
            f"- {row['label']}: Top-1 {_share(row.get('top1_accuracy'))}, Top-3 {_share(row.get('top3_hit_rate'))}, "
            f"MRR {_score(row.get('mean_reciprocal_rank'))}."
        )

    lines.extend(
        [
            "",
            "## Интерпретация",
            (
                "Сравнение остаётся retrieval-first: baselines нужны только для калибровки, "
                "а различия между нейросетевыми режимами не являются клиническим доказательством превосходства."
            ),
        ]
    )
    return "\n".join(lines)


def refresh_evaluation_artifacts(
    bundle: dict[str, Any],
    settings: Settings,
    baseline_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    models = _model_rows(bundle, settings)
    baselines = _baseline_rows(settings, baseline_rows)
    seed_stability = _load_json(settings.seed_stability_path)

    dashboard = bundle["dashboard"]
    summary = {
        "generated_at": _utc_now(),
        "dataset": {
            "patients": int(dashboard["patients_count"]),
            "extracted_windows": int(dashboard["total_meal_windows"]),
            "usable_windows": int(dashboard["usable_meal_windows"]),
            "memory_windows": int(dashboard["memory_size"]),
            "feature_dimension": int(dashboard["feature_dimension"]),
        },
        "models": models,
        "baselines": baselines,
        "robustness": {
            "definition": "Сохранение retrieval-качества измеряется как отношение Top-k после контролируемого искажения к чистому Top-k на том же query pool.",
            "rows": [
                {"model_key": row["key"], "model_label": row["label"], **item}
                for row in [*models, *baselines]
                for item in row.get("robustness_summary", [])
            ],
        },
        "patient_generalization": patient_generalization_summary(models),
        "failure_analysis": curate_failure_analysis(models),
        "seed_stability": seed_stability,
    }

    settings.latest_eval_summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    settings.latest_baselines_path.write_text(
        json.dumps(
            {
                "generated_at": summary["generated_at"],
                "dataset": summary["dataset"],
                "baselines": baselines,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings.latest_comparison_path.write_text(_comparison_markdown(summary), encoding="utf-8")
    return summary


def _evaluate_siamese_seed(bundle: dict[str, Any], settings: Settings, seed: int) -> dict[str, Any]:
    seed_settings = replace(settings, random_seed=seed)
    feature_matrix = np.asarray(bundle["feature_matrix"], dtype=np.float32)
    encoder = bundle["encoder"]
    windows = bundle["windows"]
    index_by_window_id = bundle["index_by_window_id"]
    sequence_view, tabular_view = _sequence_and_tabular_views(feature_matrix, encoder)
    train_windows = [window for window in windows if window["split"] == "train" and window["usable_for_memory"]]
    test_windows = [window for window in windows if window["split"] == "test" and window["usable_for_memory"]]
    train_indices = [index_by_window_id[window["window_id"]] for window in train_windows]
    test_indices = [index_by_window_id[window["window_id"]] for window in test_windows]
    train_labels = [window["label"] for window in train_windows]

    model, _ = train_siamese_encoder(sequence_view[train_indices], tabular_view[train_indices], train_labels, seed_settings)
    window_embeddings = _encode_in_batches(model, sequence_view, tabular_view)
    memory_embeddings = window_embeddings[train_indices]

    top1_hits: list[bool] = []
    top3_hits: list[bool] = []
    mrr_values: list[float] = []
    for query_window, query_index in zip(test_windows, test_indices):
        similarities = memory_embeddings @ window_embeddings[query_index]
        top_indices = np.argsort(similarities)[::-1][: settings.retrieval_top_k]
        top_labels = [train_windows[int(index)]["label"] for index in top_indices]
        top1, top3, mrr = _topk_metrics(query_window["label"], top_labels)
        top1_hits.append(top1)
        top3_hits.append(top3)
        mrr_values.append(mrr)

    return {
        "seed": seed,
        "top1_accuracy": float(np.mean(top1_hits)),
        "top3_hit_rate": float(np.mean(top3_hits)),
        "mean_reciprocal_rank": float(np.mean(mrr_values)),
    }


def _evaluate_som_seed(bundle: dict[str, Any], settings: Settings, seed: int) -> dict[str, Any]:
    seed_settings = replace(settings, random_seed=seed)
    feature_matrix = np.asarray(bundle["feature_matrix"], dtype=np.float32)
    windows = bundle["windows"]
    index_by_window_id = bundle["index_by_window_id"]
    train_windows = [window for window in windows if window["split"] == "train" and window["usable_for_memory"]]
    test_windows = [window for window in windows if window["split"] == "test" and window["usable_for_memory"]]
    train_indices = [index_by_window_id[window["window_id"]] for window in train_windows]
    train_matrix = feature_matrix[train_indices]

    weights, _ = _train_som(train_matrix.astype(np.float64), seed_settings)
    train_assignments = np.asarray([_bmu_index(weights, vector) for vector in train_matrix], dtype=int)
    cell_stats = _cell_statistics(
        train_windows=train_windows,
        train_matrix=train_matrix,
        weights=weights,
        train_assignments=train_assignments,
        grid_shape=(settings.som_grid_height, settings.som_grid_width),
    )
    train_norm = l2_normalize(train_matrix)

    top1_hits: list[bool] = []
    top3_hits: list[bool] = []
    mrr_values: list[float] = []
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
            grid_shape=(settings.som_grid_height, settings.som_grid_width),
            k=settings.retrieval_top_k,
        )
        top_labels = [item["label"] for item in retrieval["neighbors"]]
        top1, top3, mrr = _topk_metrics(query_window["label"], top_labels)
        top1_hits.append(top1)
        top3_hits.append(top3)
        mrr_values.append(mrr)

    return {
        "seed": seed,
        "top1_accuracy": float(np.mean(top1_hits)),
        "top3_hit_rate": float(np.mean(top3_hits)),
        "mean_reciprocal_rank": float(np.mean(mrr_values)),
    }


def compute_seed_stability(bundle: dict[str, Any], settings: Settings, seeds: tuple[int, ...] = SEED_SET) -> dict[str, Any]:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for key, label, evaluator in (
        ("siamese_temporal", "Сиамская temporal-модель", _evaluate_siamese_seed),
        ("som", "Карта Кохонена", _evaluate_som_seed),
    ):
        seed_results = []
        status = "ok"
        reason = None
        try:
            for seed in seeds:
                seed_results.append(evaluator(bundle, settings, seed))
        except Exception as exc:  # pragma: no cover - keeps artifact refresh resilient on machines without torch
            status = "unavailable"
            reason = str(exc)

        rows.append(
            {
                "key": key,
                "label": label,
                "status": status,
                "reason": reason,
                "seeds": list(seeds),
                "runs": seed_results,
                "summary": {
                    "top1_accuracy": _mean_std(item["top1_accuracy"] for item in seed_results),
                    "top3_hit_rate": _mean_std(item["top3_hit_rate"] for item in seed_results),
                    "mean_reciprocal_rank": _mean_std(item["mean_reciprocal_rank"] for item in seed_results),
                },
            }
        )

    payload = {
        "generated_at": _utc_now(),
        "n_seeds": len(seeds),
        "note": "Отчёт построен по трём seed’ам, чтобы ограничить время локальной пересборки. Значения отражают устойчивость trainable режимов на текущем 12-пациентном split.",
        "models": rows,
    }
    settings.seed_stability_path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    settings.seed_stability_report_path.write_text(seed_stability_markdown(payload), encoding="utf-8")
    return payload


def seed_stability_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Устойчивость trainable моделей по seed’ам",
        "",
        payload["note"],
        "",
        "| Модель | Top-1 mean±std | Top-3 mean±std | MRR mean±std |",
        "| --- | --- | --- | --- |",
    ]
    for model in payload["models"]:
        if model["status"] != "ok":
            lines.append(f"| {model['label']} | недоступно | недоступно | недоступно |")
            continue
        summary = model["summary"]
        top1 = summary["top1_accuracy"]
        top3 = summary["top3_hit_rate"]
        mrr = summary["mean_reciprocal_rank"]
        lines.append(
            f"| {model['label']} | {_score(top1['mean'])}±{_score(top1['std'])} | "
            f"{_score(top3['mean'])}±{_score(top3['std'])} | {_score(mrr['mean'])}±{_score(mrr['std'])} |"
        )
    lines.extend(
        [
            "",
            "Высокая дисперсия должна трактоваться как методологическое ограничение, а не скрываться за единственным лучшим запуском.",
        ]
    )
    return "\n".join(lines)


def audit_som(bundle: dict[str, Any], settings: Settings) -> dict[str, Any]:
    som_bundle = _load_pickle(settings.som_runtime_bundle_path)
    if not som_bundle:
        payload = {
            "generated_at": _utc_now(),
            "status": "unavailable",
            "reason": "SOM runtime bundle отсутствует.",
        }
        settings.som_audit_path.write_text("# Аудит SOM\n\nSOM runtime bundle отсутствует.\n", encoding="utf-8")
        return payload

    feature_matrix = np.asarray(bundle["feature_matrix"], dtype=np.float32)
    windows = bundle["windows"]
    index_by_window_id = bundle["index_by_window_id"]
    train_windows = [window for window in windows if window["split"] == "train" and window["usable_for_memory"]]
    test_windows = [window for window in windows if window["split"] == "test" and window["usable_for_memory"]]
    train_indices = [index_by_window_id[window["window_id"]] for window in train_windows]
    train_matrix = feature_matrix[train_indices]
    train_norm = l2_normalize(train_matrix)
    train_ids = {window["window_id"] for window in train_windows}

    weights = np.asarray(som_bundle["weights"], dtype=np.float32)
    train_assignments = np.asarray(som_bundle["train_assignments"], dtype=int)
    raw_cell_stats = som_bundle["cell_stats"]
    cell_stats = {int(key): value for key, value in raw_cell_stats.items()} if isinstance(raw_cell_stats, dict) else {
        int(item["cell_index"]): item for item in raw_cell_stats
    }
    grid_shape = (int(som_bundle["config"]["grid_height"]), int(som_bundle["config"]["grid_width"]))

    leakage_errors = 0
    ranking_errors = 0
    relation_errors = 0
    top1_hits: list[bool] = []
    top3_hits: list[bool] = []
    mrr_values: list[float] = []

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
        neighbors = retrieval["neighbors"]
        similarities = [float(item["similarity"]) for item in neighbors]
        if any(left < right - 1e-12 for left, right in zip(similarities, similarities[1:])):
            ranking_errors += 1
        for neighbor in neighbors:
            if neighbor["window_id"] not in train_ids or neighbor["window_id"] == query_window["window_id"]:
                leakage_errors += 1
            same_patient_expected = str(neighbor["patient_id"]) == str(query_window["patient_id"])
            if bool(neighbor["same_patient"]) != same_patient_expected:
                relation_errors += 1
        top_labels = [item["label"] for item in neighbors]
        top1, top3, mrr = _topk_metrics(query_window["label"], top_labels)
        top1_hits.append(top1)
        top3_hits.append(top3)
        mrr_values.append(mrr)

    recomputed = {
        "top1_accuracy": float(np.mean(top1_hits)),
        "top3_hit_rate": float(np.mean(top3_hits)),
        "mean_reciprocal_rank": float(np.mean(mrr_values)),
    }
    saved = (som_bundle.get("evaluation") or {}).get("retrieval_metrics", {})
    metric_delta = {
        key: abs(float(recomputed[key]) - float(saved.get(key, 0.0)))
        for key in recomputed
    }
    source = inspect.getsource(__import__("app.som", fromlist=[""]))
    no_siamese_dependency = "siamese" not in source.lower()
    strong_result_holds = all(value < 1e-12 for value in metric_delta.values()) and leakage_errors == 0 and ranking_errors == 0 and relation_errors == 0

    payload = {
        "generated_at": _utc_now(),
        "status": "ok" if strong_result_holds else "attention",
        "checked": {
            "query_count": len(test_windows),
            "train_memory_count": len(train_windows),
            "ranking_errors": ranking_errors,
            "leakage_errors": leakage_errors,
            "same_cross_relation_errors": relation_errors,
            "no_siamese_dependency_in_som_module": no_siamese_dependency,
        },
        "saved_metrics": {key: saved.get(key) for key in recomputed},
        "recomputed_metrics": recomputed,
        "metric_delta": metric_delta,
        "strong_result_holds": strong_result_holds,
    }
    settings.som_audit_path.write_text(som_audit_markdown(payload), encoding="utf-8")
    return payload


def som_audit_markdown(payload: dict[str, Any]) -> str:
    if payload.get("status") == "unavailable":
        return f"# Аудит SOM\n\n{payload.get('reason')}\n"

    checked = payload["checked"]
    lines = [
        "# Аудит оценки карты Кохонена",
        "",
        "## Что проверено",
        f"- Запросные окна: {checked['query_count']}",
        f"- Train-memory окна: {checked['train_memory_count']}",
        "- Top-k ranking пересчитан из SOM weights, BMU, расстояния на карте и локального feature similarity.",
        "- Проверено, что retrieved окна принадлежат train-memory и не совпадают с query identity.",
        "- Проверена корректность same-patient / cross-patient маркера.",
        "- Проверено, что модуль SOM не использует Siamese ranking artifacts.",
        "",
        "## Результаты проверок",
        f"- Ошибки ранжирования: {checked['ranking_errors']}",
        f"- Ошибки leakage/self-retrieval: {checked['leakage_errors']}",
        f"- Ошибки same/cross patient: {checked['same_cross_relation_errors']}",
        f"- Зависимость от Siamese artifacts в SOM module: {'нет' if checked['no_siamese_dependency_in_som_module'] else 'требует проверки'}",
        "",
        "## Сравнение сохранённых и пересчитанных метрик",
        "| Метрика | Сохранено | Пересчитано | Абсолютная разница |",
        "| --- | --- | --- | --- |",
    ]
    for key, saved_value in payload["saved_metrics"].items():
        lines.append(
            f"| {key} | {_score(saved_value)} | {_score(payload['recomputed_metrics'][key])} | {_score(payload['metric_delta'][key])} |"
        )
    lines.extend(
        [
            "",
            "## Вывод",
            (
                "Сильный результат SOM сохраняется после аудита: пересчитанные top-k метрики совпадают с сохранёнными, "
                "а признаков self-retrieval leakage или подмены Siamese ranking не обнаружено."
                if payload["strong_result_holds"]
                else "Аудит выявил расхождения; SOM результат следует считать требующим дополнительной проверки."
            ),
        ]
    )
    return "\n".join(lines)
