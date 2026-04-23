from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .baselines import (
    compute_noise_stability,
    evaluate_retrieval_baselines,
    label_distribution_summary,
    load_cached_baselines,
    save_baselines_cache,
)
from .config import Settings, get_settings
from .engines import RetrievalEngine, available_model_descriptors, create_engine_registry, normalize_model_key
from .pipeline import build_runtime_bundle


def _model_interpretation(model_key: str) -> str:
    if model_key == "hopfield":
        return "Память Хопфилда полезна для ассоциативной интерпретации и анализа recall-траектории."
    if model_key == "siamese_temporal":
        return "Сиамская temporal-модель ориентирована на retrieval-качество в обученном эмбеддинговом пространстве."
    return "Карта Кохонена полезна для анализа структуры данных и локальных топологических областей."


def _representation_size(engine: RetrievalEngine, dashboard_payload: dict[str, Any]) -> int:
    model_summary = dashboard_payload.get("model_summary", {})
    value = model_summary.get("representation_dimension")
    return int(value) if value is not None else int(dashboard_payload.get("feature_dimension", 0))


def _summary_line(model_key: str, raw: dict[str, Any]) -> str:
    top_neighbor = raw["top_k_memories"][0] if raw.get("top_k_memories") else None
    if model_key == "hopfield":
        target = top_neighbor["window"]["label_display"] if top_neighbor else "не определено"
        return (
            f"Запросное окно ассоциативно притягивается к памяти класса «{target}»; "
            f"top-1 совпадение {top_neighbor['similarity']:.3f}, восстановление сопровождается снижением энергии."
            if top_neighbor
            else "Память Хопфилда не нашла устойчивого совпадения."
        )
    if model_key == "siamese_temporal":
        return "Запросное окно расположено рядом с группой исторически сходных случаев в эмбеддинговом пространстве."
    return "Запросное окно попадает в локальную топологическую область карты Кохонена, где сосредоточены сходные случаи."


def _overlay_chart(query_window: dict[str, Any], neighbors: list[dict[str, Any]], *, top_n: int = 3) -> dict[str, Any]:
    series = [
        {
            "key": "query",
            "label": "Запрос",
            "values": query_window["full_curve_values"],
        }
    ]
    for index, neighbor in enumerate(neighbors[:top_n], start=1):
        series.append(
            {
                "key": f"neighbor_{index}",
                "label": f"Сосед {index}",
                "values": neighbor["window"]["full_curve_values"],
            }
        )
    return {
        "kind": "curve_overlay",
        "minutes": query_window["full_curve_minutes"],
        "series": series,
    }


@dataclass
class AppService:
    settings: Settings
    bundle: dict[str, Any] | None = None
    engines: dict[str, RetrievalEngine] | None = None
    baseline_rows: list[dict[str, Any]] | None = None

    def ensure_bundle(self, force_refresh: bool = False) -> dict[str, Any]:
        if self.bundle is None or force_refresh:
            self.bundle = build_runtime_bundle(force=force_refresh, settings=self.settings)
            self.engines = None
            self.baseline_rows = None
        return self.bundle

    def ensure_engines(self) -> dict[str, RetrievalEngine]:
        if self.engines is None:
            self.engines = create_engine_registry(self.settings, self.ensure_bundle())
        return self.engines

    def ensure_baselines(self) -> list[dict[str, Any]]:
        if self.baseline_rows is None:
            cached = load_cached_baselines(self.settings)
            if cached is not None:
                self.baseline_rows = cached
            else:
                self.baseline_rows = evaluate_retrieval_baselines(self.ensure_bundle(), self.settings)
                save_baselines_cache(self.settings, self.baseline_rows)
        return self.baseline_rows

    def get_engine(self, model: str | None = None) -> RetrievalEngine:
        normalized = normalize_model_key(model)
        engines = self.ensure_engines()
        if normalized not in engines:
            supported = ", ".join(sorted(engines))
            raise ValueError(f"Неизвестная retrieval-модель '{model}'. Допустимые значения: {supported}.")
        return engines[normalized]

    def models(self) -> dict[str, Any]:
        descriptors = available_model_descriptors()
        return {
            "default_model": "hopfield",
            "models": descriptors,
        }

    def health(self) -> dict[str, Any]:
        bundle = self.ensure_bundle()
        dashboard = bundle["dashboard"]
        return {
            "status": "ok",
            "generated_at": bundle["generated_at"],
            "patients": dashboard["patients_count"],
            "usable_windows": dashboard["usable_meal_windows"],
            "memory_size": dashboard["memory_size"],
            "feature_dimension": dashboard["feature_dimension"],
            "available_models": [item["key"] for item in available_model_descriptors()],
        }

    def dashboard(self, model: str = "hopfield") -> dict[str, Any]:
        engine = self.get_engine(model)
        engine_dashboard = engine.dashboard()
        retrieval = engine.evaluation()["retrieval_metrics"]
        comparison = self._model_comparison_panel()
        representation_size = _representation_size(engine, engine_dashboard)
        noise_stability = next((item["noise_stability"] for item in comparison if item["key"] == engine.descriptor["key"]), None)

        return {
            "title": self.settings.project_name,
            "subtitle": self.settings.project_subtitle,
            "disclaimer": (
                "Система не является медицинским изделием, не формирует клинические рекомендации и не предназначена для выбора дозы инсулина."
            ),
            "selected_model": engine.descriptor,
            "available_models": available_model_descriptors(),
            "headline_metrics": {
                "top1_accuracy": float(retrieval.get("top1_accuracy", 0.0)),
                "top3_hit_rate": float(retrieval.get("top3_hit_rate", 0.0)),
                "mean_reciprocal_rank": float(retrieval.get("mean_reciprocal_rank", 0.0)),
                "representation_size": representation_size,
                "representation_label": engine.descriptor.get("representation_name", "Представление"),
                "noise_stability": noise_stability,
            },
            "dataset_strip": {
                "patients": int(engine_dashboard["patients_count"]),
                "extracted_windows": int(engine_dashboard["total_meal_windows"]),
                "usable_windows": int(engine_dashboard["usable_meal_windows"]),
                "memory_windows": int(engine_dashboard["memory_size"]),
            },
            "model_comparison": comparison,
            "chart": {
                "kind": "label_distribution",
                "title": "Распределение исследовательских меток",
                "data": label_distribution_summary(engine_dashboard["label_distribution"]),
            },
            "interpretation": (
                f"{_model_interpretation(engine.descriptor['key'])} "
                "Результаты следует интерпретировать как сравнение retrieval-подходов и структуры признакового пространства, "
                "а не как доказательство клинической применимости."
            ),
            "limitations": engine.evaluation()["limitations"],
            "exclusions": engine_dashboard["exclusion_reasons"],
        }

    def windows(
        self,
        patient_id: str | None = None,
        label: str | None = None,
        meal_segment: str | None = None,
        split: str | None = None,
        query: str | None = None,
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        records = self.ensure_bundle()["windows"]
        filtered = []
        query_normalized = (query or "").strip().lower()
        for record in records:
            if patient_id and str(record["patient_id"]) != str(patient_id):
                continue
            if label and record["label"] != label:
                continue
            if meal_segment and record["meal_segment"] != meal_segment:
                continue
            if split and record["split"] != split:
                continue
            if query_normalized and query_normalized not in str(record["window_id"]).lower():
                continue
            filtered.append(record)
        filtered.sort(key=lambda item: item["meal_time"], reverse=True)
        return filtered[:limit]

    def window(self, window_id: str) -> dict[str, Any]:
        bundle = self.ensure_bundle()
        record = next((window for window in bundle["windows"] if window["window_id"] == window_id), None)
        if record is None:
            raise KeyError(f"Неизвестный идентификатор окна: {window_id}")
        return record

    def retrieve(self, *, model: str, window_id: str, k: int | None = None, beta: float | None = None, steps: int | None = None) -> dict[str, Any]:
        engine = self.get_engine(model)
        query_window = self.window(window_id)
        memory_index = query_window.get("memory_index")
        if memory_index is None:
            raise ValueError("Это окно исключено из retrieval-анализа, поскольку не входит в основной исследовательский срез памяти.")
        query_vector = np.asarray(self.ensure_bundle()["feature_matrix"][int(memory_index)], dtype=float)
        retrieval = engine.retrieve(
            query_window,
            query_vector,
            k=k or self.settings.retrieval_top_k,
            beta=beta or (self.settings.siamese_similarity_beta if engine.descriptor["key"] == "siamese_temporal" else self.settings.hopfield_beta),
            steps=steps or self.settings.recall_steps,
        )
        return self._normalize_retrieve_payload(engine, retrieval)

    def prototypes(self, model: str = "hopfield") -> list[dict[str, Any]]:
        return self.get_engine(model).prototypes()

    def prototype(self, label: str, model: str = "hopfield") -> dict[str, Any]:
        return self.get_engine(model).prototype(label)

    def evaluation(self, selected_model: str | None = None) -> dict[str, Any]:
        model_rows = self._model_comparison_panel()
        baseline_rows = self.ensure_baselines()
        available_baselines = [item for item in baseline_rows if item.get("available")]
        unavailable_baselines = [item for item in baseline_rows if not item.get("available")]

        comparison_rows = [
            {
                "key": item["key"],
                "label": item["label"],
                "family": "neural",
                "available": True,
                "top1_accuracy": item["top1_accuracy"],
                "top3_hit_rate": item["top3_hit_rate"],
                "mean_reciprocal_rank": item["mean_reciprocal_rank"],
                "noise_stability": item["noise_stability"],
                "secondary_metrics": item["secondary_metrics"],
                "notes": item["scientific_description"],
            }
            for item in model_rows
        ] + available_baselines

        chart_rows = [
            {
                "label": row["label"],
                "top1_accuracy": row["top1_accuracy"],
                "top3_hit_rate": row["top3_hit_rate"],
                "mean_reciprocal_rank": row["mean_reciprocal_rank"],
                "noise_stability": row["noise_stability"],
                "family": row["family"],
            }
            for row in comparison_rows
            if row.get("top1_accuracy") is not None
        ]

        stability_chart = {
            "kind": "noise_robustness",
            "series": [
                {
                    "key": row["key"],
                    "label": row["label"],
                    "family": row["family"],
                    "points": row.get("noise_points", []),
                }
                for row in comparison_rows
                if row.get("noise_points")
            ],
        }

        additional_metrics = {
            "models": [
                {
                    "key": row["key"],
                    "label": row["label"],
                    "top5_hit_rate": row["additional_metrics"].get("top5_hit_rate"),
                    "same_patient_top1": row["additional_metrics"].get("same_patient_top1_rate"),
                    "cross_patient_top1": row["additional_metrics"].get("cross_patient_top1_rate"),
                    "macro_f1": row["additional_metrics"].get("macro_f1"),
                    "balanced_accuracy": row["additional_metrics"].get("balanced_accuracy"),
                    "label_purity_top5": row["additional_metrics"].get("label_purity_top5"),
                }
                for row in model_rows
            ],
            "baselines": [
                {
                    "key": row["key"],
                    "label": row["label"],
                    "top5_hit_rate": row["additional_metrics"].get("top5_hit_rate"),
                    "macro_f1": row["additional_metrics"].get("macro_f1"),
                    "balanced_accuracy": row["additional_metrics"].get("balanced_accuracy"),
                    "label_purity_top5": row["additional_metrics"].get("label_purity_top5"),
                }
                for row in baseline_rows
                if row.get("available")
            ],
            "unavailable": unavailable_baselines,
        }

        prototype_block = self._prototype_comparison_block()
        selected_key = normalize_model_key(selected_model)
        return {
            "title": "Сравнение моделей",
            "subtitle": "Сравнение нейронных семейств в задаче поиска сходных постпрандиальных CGM-окон",
            "disclaimer": (
                "Сравнение моделей носит исследовательский характер и не предназначено для клинической интерпретации."
            ),
            "selected_model": selected_key if selected_key in {row["key"] for row in model_rows} else "hopfield",
            "comparison_rows": comparison_rows,
            "comparison_chart": {
                "kind": "primary_metrics",
                "data": chart_rows,
            },
            "stability_chart": stability_chart,
            "prototype_block": prototype_block,
            "additional_metrics": additional_metrics,
            "conclusion": (
                "На данной малой выборке модели демонстрируют разные сильные стороны: память Хопфилда полезна для ассоциативной интерпретации, "
                "Siamese-модель — для retrieval-качества, карта Кохонена — для визуализации структуры. "
                "Полученные различия не следует интерпретировать как клиническое превосходство одной модели."
            ),
        }

    def about(self) -> dict[str, Any]:
        return {
            "title": "Методология",
            "intro": (
                "Проект исследует, как разные нейросетевые архитектуры работают в задаче поиска сходных постпрандиальных CGM-окон на малой выборке OhioT1DM."
            ),
            "sections": [
                {
                    "title": "Задача",
                    "body": [
                        "Основная задача — retrieval сходных случаев, а не клиническое прогнозирование и не подбор терапии.",
                        "Центральный вопрос исследования: дают ли разные нейросетевые семейства убедимо различающиеся retrieval-результаты на малой выборке.",
                    ],
                },
                {
                    "title": "Данные OhioT1DM",
                    "body": [
                        "Используются ретроспективные записи CGM и контекст приема пищи из шести пациентов OhioT1DM.",
                        "Формируются постпрандиальные окна с единым train/test протоколом и исключением self-retrieval leakage.",
                    ],
                },
                {
                    "title": "Выделение постпрандиальных окон",
                    "body": [
                        "Каждое окно включает предпищевой фрагмент CGM, контекст приема пищи и постпрандиальный отклик.",
                        "Окна с недостаточным покрытием, конфликтом соседних приемов пищи или некорректным baseline исключаются из основного retrieval-среза.",
                    ],
                },
                {
                    "title": "Признаки",
                    "body": [
                        "Используются предпищевой CGM, delta-from-baseline, meal context, time context, patient context и heart-rate context при наличии.",
                        "Признаковое представление одинаково для всех моделей, чтобы сравнение оставалось честным.",
                    ],
                },
                {
                    "title": "Модели",
                    "body": [
                        "Память Хопфилда: associative recall над train-векторами признаков.",
                        "Сиамская temporal-модель: shared encoder, переводящий окно в эмбеддинговое пространство для nearest-neighbor retrieval.",
                        "Карта Кохонена: карта самоорганизации, сохраняющая топологию локальных областей признакового пространства.",
                    ],
                },
                {
                    "title": "Метрики",
                    "body": [
                        "Основные метрики: top-1 same-label retrieval, top-3 hit rate, MRR и noise stability.",
                        "Дополнительные метрики зависят от модели: energy drop для Хопфилда, neighborhood purity для Siamese, quantization/topographic error для SOM.",
                    ],
                },
                {
                    "title": "Ограничения",
                    "body": [
                        "Система не является медицинским изделием, не формирует клинические рекомендации и не предназначена для выбора дозы инсулина.",
                        "Результаты следует интерпретировать как сравнение retrieval-подходов и структуры признакового пространства, а не как доказательство клинической применимости.",
                    ],
                },
            ],
            "formulas": [
                {"title": "Hopfield similarity", "formula": "s_i = x_i · q"},
                {"title": "Hopfield recall", "formula": "q_next = Σ_i softmax(βs_i) x_i"},
                {"title": "Temporal retrieval", "formula": "sim(z_q, z_i) = cos(z_q, z_i)"},
                {"title": "SOM BMU", "formula": "bmu(x) = argmin_j ||x - w_j||"},
            ],
            "available_models": available_model_descriptors(),
        }

    def _normalize_retrieve_payload(self, engine: RetrievalEngine, raw: dict[str, Any]) -> dict[str, Any]:
        model_key = engine.descriptor["key"]
        neighbors = raw.get("top_k_memories", [])
        top_neighbor = neighbors[0] if neighbors else None
        query_window = raw["query_window"]

        if model_key == "hopfield":
            energy_values = [float(value) for value in raw.get("energy_values", [])]
            energy_drop = float(energy_values[0] - energy_values[-1]) if len(energy_values) > 1 else 0.0
            chart_payload = {
                "kind": "energy_trajectory",
                "points": [
                    {
                        "step": step.get("step"),
                        "energy": step.get("energy"),
                    }
                    for step in raw.get("recalled_steps", [])
                ],
            }
            primary_metrics = {
                "top1_similarity": top_neighbor["similarity"] if top_neighbor else None,
                "confidence_level": raw["uncertainty"]["level_label"],
                "top12_gap": raw["uncertainty"]["top_similarity_gap"],
                "energy_drop": energy_drop,
            }
            advanced = {
                "feature_block_similarity": {
                    neighbor["window_id"]: neighbor.get("feature_block_similarity", {})
                    for neighbor in neighbors[:3]
                },
                "recall_steps": raw.get("recalled_steps", []),
                "prototype_affinities": raw.get("prototype_distribution", {}),
                "entropy": raw["uncertainty"].get("attention_entropy"),
                "same_patient_share": raw["uncertainty"].get("same_patient_share"),
            }
        elif model_key == "siamese_temporal":
            neighborhood_purity = float(np.mean([neighbor["label"] == top_neighbor["label"] for neighbor in neighbors])) if top_neighbor else None
            chart_payload = _overlay_chart(query_window, neighbors)
            primary_metrics = {
                "top1_similarity": top_neighbor["similarity"] if top_neighbor else None,
                "top12_gap": raw["uncertainty"]["top_similarity_gap"],
                "neighborhood_purity": neighborhood_purity,
                "patient_relation": "same" if top_neighbor and top_neighbor["same_patient"] else "cross" if top_neighbor else None,
            }
            advanced = {
                "feature_block_similarity": {
                    neighbor["window_id"]: neighbor.get("feature_block_similarity", {})
                    for neighbor in neighbors[:3]
                },
                "embedding_notes": raw.get("retrieval_reason_note"),
                "prototype_distribution": raw.get("prototype_distribution", {}),
            }
        else:
            som_map = raw.get("som_map", {})
            active_cell_index = som_map.get("active_cell")
            cells = som_map.get("cells", [])
            active_cell = next((cell for cell in cells if cell.get("cell_index") == active_cell_index), None)
            chart_payload = {
                "kind": "som_grid",
                "grid_height": som_map.get("grid_height"),
                "grid_width": som_map.get("grid_width"),
                "active_cell": active_cell_index,
                "cells": cells,
            }
            primary_metrics = {
                "bmu_confidence": (1.0 / (1.0 + float(som_map.get("quantization_error", 0.0)))) if som_map else None,
                "cluster_purity": active_cell.get("purity") if active_cell else None,
                "quantization_error": som_map.get("quantization_error"),
                "topographic_error": som_map.get("topographic_error"),
            }
            advanced = {
                "neighbor_cells": som_map.get("neighbor_cells", []),
                "label_distribution": raw.get("prototype_distribution", {}),
                "feature_block_similarity": {
                    neighbor["window_id"]: neighbor.get("feature_block_similarity", {})
                    for neighbor in neighbors[:3]
                },
            }

        normalized_neighbors = [
            {
                "rank": index + 1,
                "window_id": neighbor["window_id"],
                "label": neighbor["window"]["label"],
                "label_display": neighbor["window"].get("label_display"),
                "patient_id": neighbor["patient_id"],
                "similarity": neighbor["similarity"],
                "same_patient": bool(neighbor.get("same_patient")),
                "relation_badge": "Тот же пациент" if neighbor.get("same_patient") else "Другой пациент",
                "reason": neighbor.get("explanation_text", ""),
                "map_distance": neighbor.get("map_distance"),
                "window": neighbor["window"],
            }
            for index, neighbor in enumerate(neighbors)
        ]

        return {
            "model": engine.descriptor,
            "query_window": query_window,
            "summary_text": _summary_line(model_key, raw),
            "primary_metrics": primary_metrics,
            "neighbors": normalized_neighbors,
            "chart_payload": chart_payload,
            "advanced": advanced,
        }

    def _model_comparison_panel(self) -> list[dict[str, Any]]:
        comparison: list[dict[str, Any]] = []
        classification_keys = {
            "hopfield": "hopfield",
            "siamese_temporal": "siamese",
            "som": "som",
        }
        for key, engine in self.ensure_engines().items():
            evaluation = engine.evaluation()
            metrics = evaluation["retrieval_metrics"]
            diagnostics = evaluation["diagnostics"]
            baseline_metrics = evaluation.get("baselines", {}).get(classification_keys[key], {})
            comparison.append(
                {
                    "key": key,
                    "label": engine.descriptor["label"],
                    "scientific_description": engine.descriptor["scientific_description"],
                    "top1_accuracy": float(metrics.get("top1_accuracy", 0.0)),
                    "top3_hit_rate": float(metrics.get("top3_hit_rate", 0.0)),
                    "mean_reciprocal_rank": float(metrics.get("mean_reciprocal_rank", 0.0)),
                    "noise_stability": compute_noise_stability(engine.noise()),
                    "secondary_metrics": self._secondary_metrics_for_model(key, diagnostics, metrics),
                    "additional_metrics": {
                        "top5_hit_rate": metrics.get("top5_hit_rate"),
                        "label_purity_top5": metrics.get("label_purity_top5"),
                        "same_patient_top1_rate": diagnostics.get("same_patient_top1_rate"),
                        "cross_patient_top1_rate": diagnostics.get("cross_patient_top1_rate"),
                        "balanced_accuracy": baseline_metrics.get("balanced_accuracy"),
                        "macro_f1": baseline_metrics.get("macro_f1"),
                    },
                    "noise_points": engine.noise(),
                }
            )
        comparison.sort(key=lambda item: item["key"])
        return comparison

    def _secondary_metrics_for_model(self, key: str, diagnostics: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
        if key == "hopfield":
            return {
                "average_energy_drop": diagnostics.get("average_energy_drop"),
                "recall_convergence_rate": None,
            }
        if key == "siamese_temporal":
            return {
                "neighborhood_purity": metrics.get("label_purity_top5"),
                "seed_consistency": None,
            }
        return {
            "quantization_error": diagnostics.get("average_quantization_error"),
            "topographic_error": diagnostics.get("average_topographic_error"),
            "local_cluster_purity": diagnostics.get("average_cluster_purity"),
        }

    def _prototype_comparison_block(self) -> list[dict[str, Any]]:
        hopfield = self.get_engine("hopfield").prototypes()[:2]
        siamese = self.get_engine("siamese_temporal").prototypes()[:2]
        som_cells = self.get_engine("som").prototypes()[:2]
        return [
            {
                "model": "hopfield",
                "label": "Память Хопфилда",
                "items": [
                    {
                        "title": item["label_display"],
                        "support": item["support_size"],
                        "purity": item["purity"],
                    }
                    for item in hopfield
                ],
            },
            {
                "model": "siamese_temporal",
                "label": "Сиамская temporal-модель",
                "items": [
                    {
                        "title": item["label_display"],
                        "support": item["support_size"],
                        "purity": item["purity"],
                    }
                    for item in siamese
                ],
            },
            {
                "model": "som",
                "label": "Карта Кохонена",
                "items": [
                    {
                        "title": item.get("dominant_label_display", "Пустая ячейка"),
                        "support": item.get("count", 0),
                        "purity": item.get("purity", 0.0),
                    }
                    for item in som_cells
                ],
            },
        ]


_SERVICE: AppService | None = None


def get_service(force_refresh: bool = False) -> AppService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AppService(settings=get_settings())
    _SERVICE.ensure_bundle(force_refresh=force_refresh)
    return _SERVICE
