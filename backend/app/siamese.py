from __future__ import annotations

from collections import Counter
import json
import pickle
from typing import Any, Sequence

import numpy as np

from .config import Settings
from .memory import stable_softmax
from .pipeline import (
    aggregate_label_weights,
    build_case_summary,
    classification_metrics,
    display_label,
    display_meal_segment,
    format_share,
    group_by_records,
    json_ready,
    prototype_meaning,
)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when torch is absent
    torch = None
    nn = None
    F = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


SIAMESE_DESCRIPTOR = {
    "key": "siamese_temporal",
    "label": "Сиамская retrieval-модель",
    "scientific_description": "Нейросетевая модель метрического пространства для извлечения похожих случаев.",
    "short_description": "Косинусное сходство в пространстве эмбеддингов.",
    "representation_name": "Эмбеддинг окна",
    "prototype_name": "Прототип в пространстве представлений",
    "similarity_name": "Косинусное сходство эмбеддингов",
    "supports_iterative_recall": False,
}


def require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError(
            "Для Siamese retrieval-модели требуется установленный PyTorch. "
            "Добавьте зависимость `torch` и пересоберите артефакты."
        ) from TORCH_IMPORT_ERROR


def _safe_l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _sequence_and_tabular_views(
    feature_matrix: np.ndarray,
    encoder: Any,
) -> tuple[np.ndarray, np.ndarray]:
    sequence_blocks = []
    for block_name in ("premeal_cgm", "premeal_delta", "missingness"):
        start, end = encoder.block_slices[block_name]
        sequence_blocks.append(feature_matrix[:, start:end])
    sequence = np.stack(sequence_blocks, axis=1).astype(np.float32)

    tabular_blocks = []
    for block_name in ("meal_context", "time_context", "patient_context", "heart_rate_context"):
        start, end = encoder.block_slices[block_name]
        tabular_blocks.append(feature_matrix[:, start:end])
    tabular = np.concatenate(tabular_blocks, axis=1).astype(np.float32)
    return sequence, tabular


class TemporalEmbeddingEncoder(nn.Module):
    def __init__(self, sequence_channels: int, tabular_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(sequence_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(24 + 16, 64),
            nn.GELU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, sequence: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        sequence_features = self.sequence_encoder(sequence).squeeze(-1)
        tabular_features = self.tabular_encoder(tabular)
        embeddings = self.embedding_head(torch.cat([sequence_features, tabular_features], dim=1))
        return F.normalize(embeddings, p=2, dim=1)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    identity_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = positive_mask & ~identity_mask

    exp_logits = torch.exp(logits) * (~identity_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positives_per_anchor = positive_mask.sum(dim=1)
    valid_mask = positives_per_anchor > 0

    if not bool(valid_mask.any()):
        return logits.new_tensor(0.0)

    positive_log_prob = (positive_mask.float() * log_prob).sum(dim=1) / positives_per_anchor.clamp(min=1)
    return -positive_log_prob[valid_mask].mean()


def _encode_in_batches(
    model: TemporalEmbeddingEncoder,
    sequence: np.ndarray,
    tabular: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    require_torch()
    model.eval()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, sequence.shape[0], batch_size):
            end = start + batch_size
            sequence_batch = torch.from_numpy(sequence[start:end])
            tabular_batch = torch.from_numpy(tabular[start:end])
            batch_embeddings = model(sequence_batch, tabular_batch).cpu().numpy()
            embeddings.append(batch_embeddings.astype(np.float32))
    return np.vstack(embeddings) if embeddings else np.empty((0, 0), dtype=np.float32)


def train_siamese_encoder(
    sequence: np.ndarray,
    tabular: np.ndarray,
    labels: Sequence[str],
    settings: Settings,
) -> tuple[TemporalEmbeddingEncoder, list[dict[str, float]]]:
    require_torch()
    torch.manual_seed(settings.random_seed)
    np.random.seed(settings.random_seed)

    label_to_index = {label: index for index, label in enumerate(sorted(set(labels)))}
    encoded_labels = np.asarray([label_to_index[label] for label in labels], dtype=np.int64)

    model = TemporalEmbeddingEncoder(
        sequence_channels=sequence.shape[1],
        tabular_dim=tabular.shape[1],
        embedding_dim=settings.siamese_embedding_dim,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.siamese_learning_rate,
        weight_decay=settings.siamese_weight_decay,
    )

    sequence_tensor = torch.from_numpy(sequence)
    tabular_tensor = torch.from_numpy(tabular)
    label_tensor = torch.from_numpy(encoded_labels)

    history: list[dict[str, float]] = []
    for epoch in range(settings.siamese_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(sequence_tensor, tabular_tensor)
        loss = supervised_contrastive_loss(embeddings, label_tensor, temperature=settings.siamese_temperature)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            similarities = embeddings @ embeddings.T
            positive_mask = label_tensor.unsqueeze(0) == label_tensor.unsqueeze(1)
            negative_mask = label_tensor.unsqueeze(0) != label_tensor.unsqueeze(1)
            average_positive = float(
                similarities[positive_mask].mean().cpu().item()
            )
            average_negative = float(similarities[negative_mask].mean().cpu().item()) if bool(negative_mask.any()) else 0.0
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(loss.detach().cpu().item()),
                "positive_similarity": average_positive,
                "negative_similarity": average_negative,
            }
        )
    return model, history


def build_embedding_prototypes(
    train_windows: Sequence[dict[str, Any]],
    index_by_window_id: dict[str, int],
    window_embeddings: np.ndarray,
) -> dict[str, dict[str, Any]]:
    prototypes: dict[str, dict[str, Any]] = {}
    grouped = group_by_records(train_windows, "label")

    for label, label_windows in grouped.items():
        supporting_embeddings = np.asarray(
            [window_embeddings[index_by_window_id[window["window_id"]]] for window in label_windows],
            dtype=float,
        )
        centroid = _safe_l2_normalize(supporting_embeddings.mean(axis=0, keepdims=True))[0]
        similarities = supporting_embeddings @ centroid
        representative_order = np.argsort(similarities)[::-1][:3]
        representative_window_ids = [label_windows[index]["window_id"] for index in representative_order]

        prototypes[label] = {
            "label": label,
            "label_display": display_label(label),
            "support_size": len(label_windows),
            "support_fraction": len(label_windows) / max(len(train_windows), 1),
            "purity": float(np.mean([window["label"] == label for window in label_windows])),
            "vector": centroid.tolist(),
            "mean_curve_minutes": label_windows[0]["full_curve_minutes"],
            "mean_curve_values": np.nanmean(
                np.asarray([window["full_curve_values"] for window in label_windows], dtype=float),
                axis=0,
            ).tolist(),
            "representative_window_ids": representative_window_ids,
            "interpretation_text": (
                f"{prototype_meaning(label)} В Siamese-режиме прототип описывает компактную область "
                "пространства эмбеддингов, а не только средний вектор исходных признаков."
            ),
            "typical_context": {
                "carbs": float(np.mean([window["carbs"] for window in label_windows])),
                "bolus": float(np.mean([window["bolus"] for window in label_windows])),
                "baseline_glucose": float(np.mean([window["baseline_glucose"] for window in label_windows])),
                "trend_30m": float(np.mean([window["trend_30m"] for window in label_windows])),
                "trend_90m": float(np.mean([window["trend_90m"] for window in label_windows])),
                "meal_segment_mode": Counter(window["meal_segment"] for window in label_windows).most_common(1)[0][0],
            },
            "prototype_geometry_note": (
                "Прототип агрегирует эмбеддинги train-окон и показывает репрезентативную область "
                "метрического пространства для данного класса."
            ),
        }
    return prototypes


def evaluate_siamese_model(
    base_bundle: dict[str, Any],
    window_embeddings: np.ndarray,
    memory_embeddings: np.ndarray,
    prototypes: dict[str, dict[str, Any]],
    model: TemporalEmbeddingEncoder,
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    windows = base_bundle["windows"]
    index_by_window_id = base_bundle["index_by_window_id"]
    train_windows = [window for window in windows if window["split"] == "train" and window["usable_for_memory"]]
    test_windows = [window for window in windows if window["split"] == "test" and window["usable_for_memory"]]
    memory_metadata = [
        {"window_id": window["window_id"], "label": window["label"], "patient_id": window["patient_id"]}
        for window in train_windows
    ]
    train_labels = [item["label"] for item in memory_metadata]

    prototype_labels = list(prototypes.keys())
    prototype_matrix = (
        _safe_l2_normalize(np.asarray([prototypes[label]["vector"] for label in prototype_labels], dtype=float))
        if prototype_labels
        else np.empty((0, window_embeddings.shape[1]), dtype=float)
    )

    case_records: list[dict[str, Any]] = []
    predictions: list[str] = []
    y_true: list[str] = []

    for query_window in test_windows:
        query_index = index_by_window_id[query_window["window_id"]]
        query_embedding = np.asarray(window_embeddings[query_index], dtype=float)
        similarities = memory_embeddings @ query_embedding
        weights = stable_softmax(settings.siamese_similarity_beta * similarities)
        top_indices = np.argsort(similarities)[::-1][: settings.retrieval_top_k]
        top_labels = [memory_metadata[index]["label"] for index in top_indices]
        top_patients = [memory_metadata[index]["patient_id"] for index in top_indices]
        label_weights = aggregate_label_weights(train_labels, weights)
        predicted_label = next(iter(label_weights)) if label_weights else top_labels[0]

        if prototype_labels:
            prototype_similarities = prototype_matrix @ query_embedding
            prototype_distribution = stable_softmax(prototype_similarities)
            prototype_label = prototype_labels[int(np.argmax(prototype_similarities))]
            sorted_prototypes = np.sort(prototype_distribution)
            prototype_gap = float(sorted_prototypes[-1] - sorted_prototypes[-2]) if len(sorted_prototypes) > 1 else float(sorted_prototypes[-1])
        else:
            prototype_distribution = np.asarray([], dtype=float)
            prototype_label = "ambiguous"
            prototype_gap = 0.0

        rank = next((index + 1 for index, label in enumerate(top_labels) if label == query_window["label"]), None)
        entropy = float(-np.sum(np.clip(weights, 1e-12, 1.0) * np.log(np.clip(weights, 1e-12, 1.0))))
        sorted_weights = np.sort(np.asarray(weights, dtype=float))
        gap = float(sorted_weights[-1] - sorted_weights[-2]) if len(sorted_weights) > 1 else float(sorted_weights[-1])
        top_similarity = float(similarities[top_indices[0]]) if len(top_indices) > 0 else 0.0
        second_similarity = float(similarities[top_indices[1]]) if len(top_indices) > 1 else 0.0
        top_weight = float(weights[top_indices[0]]) if len(top_indices) > 0 else 0.0
        second_weight = float(weights[top_indices[1]]) if len(top_indices) > 1 else 0.0

        case_record = {
            "window_id": query_window["window_id"],
            "patient_id": query_window["patient_id"],
            "label": query_window["label"],
            "top_ids": [memory_metadata[index]["window_id"] for index in top_indices],
            "top_labels": top_labels,
            "top_patients": top_patients,
            "top1_correct": bool(top_labels and top_labels[0] == query_window["label"]),
            "top3_hit": query_window["label"] in top_labels[:3],
            "top5_hit": query_window["label"] in top_labels[:5],
            "mrr": 1.0 / rank if rank else 0.0,
            "label_purity_top5": float(np.mean([label == query_window["label"] for label in top_labels])) if top_labels else 0.0,
            "same_patient_rate": float(np.mean([patient == query_window["patient_id"] for patient in top_patients])) if top_patients else 0.0,
            "cross_patient_top1": bool(top_patients and top_patients[0] != query_window["patient_id"]),
            "energy_before": 0.0,
            "energy_after": 0.0,
            "energy_drop": 0.0,
            "attention_entropy": entropy,
            "top_weight_gap": gap,
            "top1_similarity": top_similarity,
            "top2_similarity": second_similarity,
            "top1_weight": top_weight,
            "top2_weight": second_weight,
            "predicted_label": predicted_label,
            "prototype_label": prototype_label,
            "prototype_top_gap": prototype_gap,
            "prototype_distribution": dict(zip(prototype_labels, prototype_distribution.tolist())),
        }
        case_record["summary_text"] = build_case_summary(case_record)
        case_records.append(case_record)
        predictions.append(predicted_label)
        y_true.append(query_window["label"])

    prototype_purity = float(np.mean([prototype["purity"] for prototype in prototypes.values()])) if prototypes else 0.0
    retrieval_metrics = {
        "top1_accuracy": float(np.mean([record["top1_correct"] for record in case_records])) if case_records else 0.0,
        "top3_hit_rate": float(np.mean([record["top3_hit"] for record in case_records])) if case_records else 0.0,
        "top5_hit_rate": float(np.mean([record["top5_hit"] for record in case_records])) if case_records else 0.0,
        "mean_reciprocal_rank": float(np.mean([record["mrr"] for record in case_records])) if case_records else 0.0,
        "label_purity_top5": float(np.mean([record["label_purity_top5"] for record in case_records])) if case_records else 0.0,
        "prototype_purity": prototype_purity,
        "nearest_neighbor_consistency": float(np.mean([record["top1_correct"] for record in case_records])) if case_records else 0.0,
    }
    diagnostics = {
        "average_energy_before": 0.0,
        "average_energy_after": 0.0,
        "average_energy_drop": 0.0,
        "average_top_weight_gap": float(np.mean([record["top_weight_gap"] for record in case_records])) if case_records else 0.0,
        "average_attention_entropy": float(np.mean([record["attention_entropy"] for record in case_records])) if case_records else 0.0,
        "average_similarity_gap": float(np.mean([record["top1_similarity"] - record["top2_similarity"] for record in case_records])) if case_records else 0.0,
        "same_patient_top1_rate": float(np.mean([not record["cross_patient_top1"] for record in case_records])) if case_records else 0.0,
        "cross_patient_top1_rate": float(np.mean([record["cross_patient_top1"] for record in case_records])) if case_records else 0.0,
    }

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

    successes = sorted(
        [record for record in case_records if record["top1_correct"]],
        key=lambda item: (item["top_weight_gap"], item["top1_similarity"]),
        reverse=True,
    )
    failures = sorted(
        [record for record in case_records if not record["top1_correct"]],
        key=lambda item: (item["top1_similarity"] - item["top2_similarity"], -item["attention_entropy"]),
    )
    ambiguous_cases = sorted(
        case_records,
        key=lambda item: (item["top1_similarity"] - item["top2_similarity"], -item["attention_entropy"]),
    )
    prototype_confusions = sorted(
        [record for record in case_records if record["prototype_label"] != record["label"]],
        key=lambda item: (item["prototype_top_gap"], item["top_weight_gap"]),
    )

    base_baselines = dict(base_bundle["evaluation"]["baselines"])
    labels = sorted(set(y_true))
    base_baselines["siamese"] = classification_metrics(y_true, predictions, labels) if predictions else {
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "confusion_matrix": {"labels": labels, "matrix": []},
    }

    noise_robustness = evaluate_siamese_noise(
        base_bundle=base_bundle,
        memory_embeddings=memory_embeddings,
        model=model,
        settings=settings,
    )

    evaluation = {
        "model": SIAMESE_DESCRIPTOR,
        "retrieval_metrics": retrieval_metrics,
        "diagnostics": diagnostics,
        "baselines": base_baselines,
        "baseline_note": (
            "Классификационные базовые методы сохранены только как вторичный ориентир. "
            "Siamese-модель сравнивается с памятью Хопфилда как альтернативный retrieval-движок, а не как повод "
            "превратить проект в соревнование классификаторов."
        ),
        "per_patient": per_patient,
        "noise_robustness": noise_robustness,
        "noise_note": (
            "В Siamese-режиме шум и маскирование сначала искажают исходный вектор признаков, после чего оценивается "
            "деградация косинусного retrieval в пространстве эмбеддингов."
        ),
        "same_vs_cross_analysis": {
            "same_patient_top1_rate": diagnostics["same_patient_top1_rate"],
            "cross_patient_top1_rate": diagnostics["cross_patient_top1_rate"],
            "interpretation": (
                "Сиамская модель стремится группировать окна по сходству динамики и контекста, но пациент-специфические признаки "
                "всё равно могут удерживать часть совпадений внутри одного пациента."
            ),
            "cross_patient_meaning": (
                "Межпациентские совпадения особенно интересны, когда близки форма предпищевого CGM и контекст приема пищи, "
                "а не только идентификатор пациента."
            ),
            "limitation": (
                "Если внутрипациентские совпадения доминируют слишком сильно, это ограничивает переносимость "
                "эмбеддингового retrieval между пациентами."
            ),
        },
        "qualitative_examples": {"successes": successes[:5], "failures": failures[:5]},
        "failure_analysis": {
            "ambiguous_cases": ambiguous_cases[:5],
            "prototype_confusions": prototype_confusions[:5],
            "interpretation": (
                "Неудачные случаи для Siamese-модели обычно соответствуют малому разрыву между top-1 и top-2 косинусным сходством, "
                "повышенной энтропии весов и близости нескольких прототипов в пространстве представлений."
            ),
        },
        "limitations": [
            f"В текущем исследовательском срезе доступны {len({window['patient_id'] for window in base_bundle['windows']})} пациентов OhioT1DM, поэтому обучаемая эмбеддинговая модель работает в режиме малой выборки.",
            "Используемые метки являются детерминированными ретроспективными категориями и служат только для структурирования метрического пространства.",
            "Эксперимент проводится в ретроспективной постановке и не подтверждает применимость в реальном времени.",
            "Часть окон не содержит полных данных по частоте сердечных сокращений, поэтому носимый контекст неполон.",
            "Умеренные retrieval-метрики и малая выборка не позволяют трактовать модель как клинически валидированную систему.",
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
            for name, metrics in base_baselines.items()
        },
    }
    return evaluation, chart_data


def evaluate_siamese_noise(
    base_bundle: dict[str, Any],
    memory_embeddings: np.ndarray,
    model: TemporalEmbeddingEncoder,
    settings: Settings,
) -> list[dict[str, Any]]:
    encoder = base_bundle["encoder"]
    feature_matrix = base_bundle["feature_matrix"]
    windows = base_bundle["windows"]
    index_by_window_id = base_bundle["index_by_window_id"]
    test_windows = [window for window in windows if window["split"] == "test" and window["usable_for_memory"]]
    train_windows = [window for window in windows if window["split"] == "train" and window["usable_for_memory"]]
    train_labels = [window["label"] for window in train_windows]

    noise_points: list[dict[str, Any]] = []
    levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    rng = np.random.default_rng(settings.random_seed)

    sequence_view, tabular_view = _sequence_and_tabular_views(feature_matrix, encoder)

    for mode in ("gaussian_noise", "feature_mask"):
        for level in levels:
            top1_hits: list[bool] = []
            top3_hits: list[bool] = []
            for window in test_windows:
                index = index_by_window_id[window["window_id"]]
                sequence = sequence_view[index].copy()
                tabular = tabular_view[index].copy()

                if mode == "gaussian_noise" and level > 0:
                    sequence += rng.normal(0.0, level, size=sequence.shape).astype(np.float32)
                    tabular += rng.normal(0.0, level, size=tabular.shape).astype(np.float32)
                elif mode == "feature_mask" and level > 0:
                    sequence_mask = rng.random(sequence.shape) < level
                    tabular_mask = rng.random(tabular.shape) < level
                    sequence[sequence_mask] = 0.0
                    tabular[tabular_mask] = 0.0

                query_embedding = _encode_in_batches(
                    model,
                    sequence.reshape(1, *sequence.shape).astype(np.float32),
                    tabular.reshape(1, -1).astype(np.float32),
                )[0]
                similarities = memory_embeddings @ query_embedding
                top_indices = np.argsort(similarities)[::-1][: settings.retrieval_top_k]
                top_labels = [train_labels[index] for index in top_indices]
                top1_hits.append(bool(top_labels and top_labels[0] == window["label"]))
                top3_hits.append(window["label"] in top_labels[:3])

            noise_points.append(
                {
                    "mode": mode,
                    "mode_display": "Гауссов шум" if mode == "gaussian_noise" else "Маскирование признаков",
                    "level": float(level),
                    "top1_accuracy": float(np.mean(top1_hits)) if top1_hits else 0.0,
                    "top3_hit_rate": float(np.mean(top3_hits)) if top3_hits else 0.0,
                }
            )
    return noise_points


def load_trained_encoder(settings: Settings, config: dict[str, Any]) -> TemporalEmbeddingEncoder:
    require_torch()
    model = TemporalEmbeddingEncoder(
        sequence_channels=int(config["sequence_channels"]),
        tabular_dim=int(config["tabular_dim"]),
        embedding_dim=int(config["embedding_dim"]),
    )
    state = torch.load(settings.siamese_state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def build_siamese_bundle(
    base_bundle: dict[str, Any],
    settings: Settings,
    force: bool = False,
) -> dict[str, Any]:
    require_torch()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    if settings.siamese_runtime_bundle_path.exists() and not force:
        with settings.siamese_runtime_bundle_path.open("rb") as handle:
            return pickle.load(handle)

    windows = base_bundle["windows"]
    encoder = base_bundle["encoder"]
    feature_matrix = np.asarray(base_bundle["feature_matrix"], dtype=np.float32)
    index_by_window_id = base_bundle["index_by_window_id"]

    sequence_view, tabular_view = _sequence_and_tabular_views(feature_matrix, encoder)
    train_windows = [window for window in windows if window["split"] == "train" and window["usable_for_memory"]]
    train_indices = [index_by_window_id[window["window_id"]] for window in train_windows]
    train_labels = [window["label"] for window in train_windows]

    model, training_history = train_siamese_encoder(
        sequence=sequence_view[train_indices],
        tabular=tabular_view[train_indices],
        labels=train_labels,
        settings=settings,
    )
    window_embeddings = _encode_in_batches(model, sequence_view, tabular_view)
    memory_embeddings = window_embeddings[train_indices]
    prototypes = build_embedding_prototypes(train_windows, index_by_window_id, window_embeddings)
    evaluation, chart_data = evaluate_siamese_model(
        base_bundle=base_bundle,
        window_embeddings=window_embeddings,
        memory_embeddings=memory_embeddings,
        prototypes=prototypes,
        model=model,
        settings=settings,
    )

    config = {
        "sequence_channels": int(sequence_view.shape[1]),
        "sequence_length": int(sequence_view.shape[2]),
        "tabular_dim": int(tabular_view.shape[1]),
        "embedding_dim": int(settings.siamese_embedding_dim),
        "epochs": int(settings.siamese_epochs),
        "temperature": float(settings.siamese_temperature),
        "learning_rate": float(settings.siamese_learning_rate),
        "weight_decay": float(settings.siamese_weight_decay),
    }

    artifact_bundle = {
        "descriptor": SIAMESE_DESCRIPTOR,
        "config": config,
        "training_history": training_history,
        "window_embeddings": window_embeddings,
        "memory_embeddings": memory_embeddings,
        "train_window_ids": [window["window_id"] for window in train_windows],
        "prototypes": json_ready(prototypes),
        "evaluation": json_ready(evaluation),
        "chart_data": json_ready(chart_data),
    }

    torch.save(model.state_dict(), settings.siamese_state_path)
    settings.siamese_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.save(settings.siamese_window_embeddings_path, window_embeddings)
    np.save(settings.siamese_memory_embeddings_path, memory_embeddings)
    test_indices = [index_by_window_id[window["window_id"]] for window in windows if window["split"] == "test" and window["usable_for_memory"]]
    np.save(settings.siamese_test_embeddings_path, window_embeddings[test_indices])
    settings.siamese_metrics_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    settings.siamese_prototypes_path.write_text(json.dumps(json_ready(prototypes), indent=2), encoding="utf-8")
    settings.siamese_report_path.write_text(
        generate_siamese_report(base_bundle, evaluation, prototypes, config),
        encoding="utf-8",
    )
    with settings.siamese_runtime_bundle_path.open("wb") as handle:
        pickle.dump(artifact_bundle, handle)
    return artifact_bundle


def load_siamese_bundle(settings: Settings, force: bool = False) -> dict[str, Any]:
    if force or not settings.siamese_runtime_bundle_path.exists():
        raise FileNotFoundError("Siamese runtime bundle not found. Rebuild artifacts with python -m app.build.")
    with settings.siamese_runtime_bundle_path.open("rb") as handle:
        return pickle.load(handle)


def generate_siamese_report(
    base_bundle: dict[str, Any],
    evaluation: dict[str, Any],
    prototypes: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> str:
    dashboard = base_bundle["dashboard"]
    retrieval = evaluation["retrieval_metrics"]
    diagnostics = evaluation["diagnostics"]
    lines = [
        "# Siamese retrieval-модель",
        "",
        "## Архитектура",
        (
            f"Использован компактный Siamese-энкодер: 1D-CNN по предпищевой последовательности CGM "
            f"(3 канала, длина {config['sequence_length']}) и MLP по табличному контексту "
            f"({config['tabular_dim']} признаков) с выходом в пространство эмбеддингов размерности {config['embedding_dim']}."
        ),
        "",
        "## Постановка",
        "Модель обучается в offline-режиме на train-окнах с supervised contrastive loss. Метки используются только для организации метрического пространства и не меняют retrieval-first природу интерфейса.",
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
        f"- Top-5: {retrieval['top5_hit_rate']:.3f}",
        f"- MRR: {retrieval['mean_reciprocal_rank']:.3f}",
        f"- Средний gap top-1/top-2: {diagnostics['average_similarity_gap']:.3f}",
        f"- Средняя энтропия весов: {diagnostics['average_attention_entropy']:.3f}",
        "",
        "## Прототипы",
    ]
    for label, prototype in prototypes.items():
        lines.append(
            f"- {display_label(label)}: поддержка {prototype['support_size']} окон ({format_share(prototype['support_fraction'])}), "
            f"чистота {prototype['purity']:.2f}, доминирующий сегмент {display_meal_segment(prototype['typical_context']['meal_segment_mode'])}."
        )

    lines.extend(
        [
            "",
            "## Интерпретация",
            "Siamese-модель выбрана как второй retrieval-движок, поскольку она остается нейросетевой, но не превращает проект в соревнование классификаторов. Она позволяет сравнить ассоциативную память Хопфилда с эмбеддинговым метрическим пространством на одной и той же задаче поиска похожих случаев.",
            "",
            "## Ограничения",
            "- Малая выборка пациентов ограничивает устойчивость обучения нейросетевого энкодера.",
            "- Косинусное сходство эмбеддингов не следует трактовать как клиническую рекомендацию.",
            "- Исследование носит ретроспективный характер и не подтверждает применимость в реальном времени.",
        ]
    )
    return "\n".join(lines)
