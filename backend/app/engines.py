from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings
from .memory import stable_softmax
from .pipeline import display_label, json_ready
from .siamese import build_siamese_bundle, load_siamese_bundle, load_trained_encoder
from .som import SOM_DESCRIPTOR, _adjacent_cells, _map_distance, _neighboring_cells, build_som_bundle, load_som_bundle


HOPFIELD_DESCRIPTOR = {
    "key": "hopfield",
    "label": "Память Хопфилда",
    "scientific_description": "Ассоциативная память с итеративным восстановлением состояния и энергетической диагностикой.",
    "short_description": "Ассоциативный retrieval-baseline с trajectory и energy.",
    "representation_name": "Признаковое представление",
    "prototype_name": "Прототип в пространстве признаков",
    "similarity_name": "Сходство после recall",
    "supports_iterative_recall": True,
}

SIAMESE_TEMPORAL_DESCRIPTOR = {
    "key": "siamese_temporal",
    "label": "Сиамская temporal-модель",
    "scientific_description": "Нейросетевая temporal-модель метрического пространства для поиска сходных постпрандиальных окон.",
    "short_description": "1D-CNN энкодер и nearest-neighbor retrieval в эмбеддинговом пространстве.",
    "representation_name": "Эмбеддинговое пространство",
    "prototype_name": "Локальный центр эмбеддингового пространства",
    "similarity_name": "Косинусное сходство эмбеддингов",
    "supports_iterative_recall": False,
}

SOM_UI_DESCRIPTOR = {
    **SOM_DESCRIPTOR,
    "label": "Карта Кохонена",
    "scientific_description": "Карта самоорганизации для топологического поиска сходных постпрандиальных окон.",
    "short_description": "Локальные области карты и neighborhood-based retrieval.",
    "representation_name": "Топологическая карта",
    "prototype_name": "Локальная область карты",
    "similarity_name": "Локальное сходство на карте",
}


def available_model_descriptors() -> list[dict[str, Any]]:
    return [HOPFIELD_DESCRIPTOR, SIAMESE_TEMPORAL_DESCRIPTOR, SOM_UI_DESCRIPTOR]


def normalize_model_key(model: str | None) -> str:
    normalized = (model or "hopfield").strip().lower()
    aliases = {
        "hopfield": "hopfield",
        "siamese": "siamese_temporal",
        "siamese_temporal": "siamese_temporal",
        "som": "som",
    }
    return aliases.get(normalized, normalized)


def _safe_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _softmax_distribution(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    similarities = matrix @ _safe_normalize(np.asarray(vector, dtype=float))
    return stable_softmax(similarities)


def _uncertainty_level(top_similarity_gap: float, attention_entropy: float) -> tuple[str, str]:
    if top_similarity_gap >= 0.12 and attention_entropy <= 2.0:
        return "high", "Высокая уверенность"
    if top_similarity_gap >= 0.05 and attention_entropy <= 4.2:
        return "medium", "Умеренная уверенность"
    return "low", "Низкая уверенность"


def _dashboard_model_summary(
    *,
    descriptor: dict[str, Any],
    memory_size: int,
    representation_size: int,
    retrieval_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "key": descriptor["key"],
        "label": descriptor["label"],
        "scientific_description": descriptor["scientific_description"],
        "memory_size": memory_size,
        "representation_dimension": representation_size,
        "primary_metric_label": "Top-3 извлечение",
        "primary_metric_value": float(retrieval_metrics.get("top3_hit_rate", 0.0)),
        "interpretation": (
            f"Модель «{descriptor['label']}» рассматривается как retrieval-подход на малой выборке и не должна трактоваться как клинически валидированная система."
        ),
    }


def _match_explanation(
    *,
    query_window: dict[str, Any],
    candidate_window: dict[str, Any],
    top_blocks: list[tuple[str, float]],
    candidate_label: str,
    model_label: str,
) -> str:
    block_names = [name for name, _ in top_blocks]
    fragments: list[str] = []
    if "premeal_cgm" in block_names:
        fragments.append("Похожая форма предпищевого CGM.")
    if "meal_context" in block_names:
        fragments.append("Сходный контекст приема пищи.")
    if "premeal_delta" in block_names:
        fragments.append("Согласованный baseline/trend.")
    if "time_context" in block_names:
        fragments.append("Близкий временной контекст.")
    if str(candidate_window["patient_id"]) == str(query_window["patient_id"]):
        fragments.append("Тот же пациент.")
    else:
        fragments.append("Другой пациент.")
    fragments.append(f"В терминах модели ближе всего к классу «{display_label(candidate_label).lower()}».")
    return " ".join(fragments)


@dataclass
class RetrievalEngine(ABC):
    settings: Settings
    base_bundle: dict[str, Any]

    @property
    @abstractmethod
    def descriptor(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def dashboard(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query_window: dict[str, Any], query_vector: np.ndarray, k: int, beta: float, steps: int) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def prototypes(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def prototype(self, label: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluation(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def noise(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def window(self, window_id: str) -> dict[str, Any]:
        record = next((window for window in self.base_bundle["windows"] if window["window_id"] == window_id), None)
        if record is None:
            raise KeyError(f"Неизвестный идентификатор окна: {window_id}")
        return record

    def _feature_block_similarity(self, query_vector: np.ndarray, candidate_window_id: str) -> dict[str, float]:
        candidate_vector = self.base_bundle["feature_matrix"][self.base_bundle["index_by_window_id"][candidate_window_id]]
        return self.base_bundle["encoder"].block_similarity(np.asarray(query_vector, dtype=float), np.asarray(candidate_vector, dtype=float))

    def _prototype_distribution(self, vector: np.ndarray, prototypes: dict[str, dict[str, Any]]) -> dict[str, float]:
        if not prototypes:
            return {}
        labels = list(prototypes.keys())
        matrix = np.asarray([prototypes[label]["vector"] for label in labels], dtype=float)
        matrix = np.asarray([_safe_normalize(row) for row in matrix], dtype=float)
        weights = _softmax_distribution(matrix, np.asarray(vector, dtype=float))
        return {label: float(weight) for label, weight in zip(labels, weights)}

    def _wrap_dashboard(self, payload: dict[str, Any], representation_size: int) -> dict[str, Any]:
        evaluation = self.evaluation()
        return {
            **payload,
            "selected_model": self.descriptor,
            "model_summary": _dashboard_model_summary(
                descriptor=self.descriptor,
                memory_size=int(payload["memory_size"]),
                representation_size=representation_size,
                retrieval_metrics=evaluation["retrieval_metrics"],
            ),
        }


@dataclass
class HopfieldEngine(RetrievalEngine):
    @property
    def descriptor(self) -> dict[str, Any]:
        return HOPFIELD_DESCRIPTOR

    def dashboard(self) -> dict[str, Any]:
        payload = dict(self.base_bundle["dashboard"])
        payload["title"] = self.settings.project_name
        payload["subtitle"] = self.settings.project_subtitle
        payload["disclaimer"] = (
            "Ретроспективный исследовательский прототип. Не является медицинским изделием и не формирует клинические рекомендации."
        )
        payload["headline_summary"] = (
            f"Память Хопфилда показывает top-1 {payload['headline_metrics']['top1_accuracy'] * 100:.1f}% и top-3 {payload['headline_metrics']['top3_hit_rate'] * 100:.1f}% "
            "на отложенных окнах; ее сильная сторона — ассоциативная интерпретация через recall и энергию."
        )
        payload["interpretation_note"] = (
            "Проект сравнивает retrieval-подходы на малой выборке OhioT1DM и не предназначен для клинической интерпретации."
        )
        return self._wrap_dashboard(payload, representation_size=int(payload["feature_dimension"]))

    def retrieve(self, query_window: dict[str, Any], query_vector: np.ndarray, k: int, beta: float, steps: int) -> dict[str, Any]:
        memory_model = self.base_bundle["memory_model"]
        retrieval = memory_model.retrieve(query_vector, k=k, beta=beta, steps=steps)
        prototypes = self.base_bundle["prototypes"]
        prototype_distribution = self._prototype_distribution(np.asarray(retrieval["recalled_vector"], dtype=float), prototypes)
        top_matches = []
        for item in retrieval["top_k"]:
            candidate_window = self.window(item["window_id"])
            block_similarity = self._feature_block_similarity(query_vector, item["window_id"])
            top_blocks = sorted(block_similarity.items(), key=lambda entry: entry[1], reverse=True)[:3]
            top_matches.append(
                {
                    **item,
                    "same_patient": str(candidate_window["patient_id"]) == str(query_window["patient_id"]),
                    "window": candidate_window,
                    "feature_block_similarity": block_similarity,
                    "top_blocks": [[name, float(value)] for name, value in top_blocks],
                    "prototype_affinity": float(prototype_distribution.get(item["label"], 0.0)),
                    "explanation_text": _match_explanation(
                        query_window=query_window,
                        candidate_window=candidate_window,
                        top_blocks=top_blocks,
                        candidate_label=item["label"],
                        model_label=self.descriptor["label"],
                    ),
                }
            )

        weights = np.asarray(retrieval["weights"], dtype=float)
        top_weight_gap = float(np.sort(weights)[-1] - np.sort(weights)[-2]) if len(weights) > 1 else float(weights[0]) if len(weights) else 0.0
        similarities = np.asarray([item["similarity"] for item in top_matches], dtype=float)
        top_similarity_gap = (
            float(np.sort(similarities)[-1] - np.sort(similarities)[-2]) if len(similarities) > 1 else float(similarities[0]) if len(similarities) else 0.0
        )
        attention_entropy = float(-np.sum(np.clip(weights, 1e-12, 1.0) * np.log(np.clip(weights, 1e-12, 1.0)))) if len(weights) else 0.0
        level_key, level_label = _uncertainty_level(top_similarity_gap, attention_entropy)

        return {
            "model": self.descriptor,
            "query_window": query_window,
            "query_vector": np.asarray(query_vector, dtype=float).tolist(),
            "recalled_vector": np.asarray(retrieval["recalled_vector"], dtype=float).tolist(),
            "recalled_steps": json_ready(retrieval["trajectory"]),
            "top_k_memories": json_ready(top_matches),
            "similarities": np.asarray(retrieval["similarities"], dtype=float).tolist(),
            "weights": weights.tolist(),
            "energy_values": [float(step["energy"]) for step in retrieval["trajectory"]],
            "prototype_distribution": prototype_distribution,
            "uncertainty": {
                "level": level_key,
                "level_label": level_label,
                "top_similarity_gap": top_similarity_gap,
                "top_weight_gap": top_weight_gap,
                "attention_entropy": attention_entropy,
                "prototype_gap": 0.0,
                "same_patient_share": float(np.mean([match["same_patient"] for match in top_matches])) if top_matches else 0.0,
                "summary_text": (
                    f"Запросное окно ассоциативно притягивается к памяти класса «{display_label(top_matches[0]['label']) if top_matches else 'не определено'}»; "
                    f"top-1 gap {top_similarity_gap:.3f}, снижение энергии {self.evaluation()['diagnostics']['average_energy_drop']:.3f}."
                ),
            },
            "trajectory_note": "Память Хопфилда выполняет итеративный recall, поэтому основная диагностика строится по траектории энергии.",
            "retrieval_reason_note": "Соседи отбираются по ассоциативному сходству после recall; блоки признаков в advanced-панели показывают, откуда берется это совпадение.",
            "prototype_note": "Прототипные веса здесь вспомогательны и используются только как дополнительный ориентир.",
            "plain_language_explanation_text": (
                "Память Хопфилда извлекает похожие случаи через ассоциативный recall над train-памятью; результат следует интерпретировать как исследовательский retrieval."
            ),
        }

    def prototypes(self) -> list[dict[str, Any]]:
        return [
            {
                **prototype,
                "model": self.descriptor,
                "prototype_space_note": "Прототип усредняет окна в исходном пространстве признаков.",
                "representative_windows": [self.window(window_id) for window_id in prototype["representative_window_ids"]],
            }
            for prototype in self.base_bundle["prototypes"].values()
        ]

    def prototype(self, label: str) -> dict[str, Any]:
        if label not in self.base_bundle["prototypes"]:
            raise KeyError(f"Неизвестная метка прототипа: {label}")
        prototype = self.base_bundle["prototypes"][label]
        return {
            **prototype,
            "model": self.descriptor,
            "prototype_space_note": "Прототип усредняет окна в исходном пространстве признаков.",
            "representative_windows": [self.window(window_id) for window_id in prototype["representative_window_ids"]],
        }

    def evaluation(self) -> dict[str, Any]:
        return {
            **self.base_bundle["evaluation"],
            "model": self.descriptor,
            "comparison_note": (
                "Сравнение моделей носит исследовательский характер и не предназначено для клинической интерпретации."
            ),
        }

    def noise(self) -> list[dict[str, Any]]:
        return self.base_bundle["chart_data"]["noise_robustness"]


@dataclass
class SiameseTemporalEngine(RetrievalEngine):
    bundle: dict[str, Any] | None = None
    encoder_model: Any | None = None

    @property
    def descriptor(self) -> dict[str, Any]:
        return SIAMESE_TEMPORAL_DESCRIPTOR

    def ensure_loaded(self) -> dict[str, Any]:
        if self.bundle is None:
            try:
                self.bundle = load_siamese_bundle(self.settings, force=False)
            except FileNotFoundError:
                self.bundle = build_siamese_bundle(self.base_bundle, self.settings, force=True)
        return self.bundle

    def ensure_encoder(self) -> Any:
        bundle = self.ensure_loaded()
        if self.encoder_model is None:
            self.encoder_model = load_trained_encoder(self.settings, bundle["config"])
        return self.encoder_model

    def dashboard(self) -> dict[str, Any]:
        payload = dict(self.base_bundle["dashboard"])
        evaluation = self.evaluation()
        payload["title"] = self.settings.project_name
        payload["subtitle"] = self.settings.project_subtitle
        payload["headline_metrics"] = evaluation["retrieval_metrics"]
        payload["headline_summary"] = (
            f"Сиамская temporal-модель показывает top-1 {evaluation['retrieval_metrics']['top1_accuracy'] * 100:.1f}% и top-3 {evaluation['retrieval_metrics']['top3_hit_rate'] * 100:.1f}% "
            "на отложенных окнах; ее сильная сторона — retrieval-качество в обученном эмбеддинговом пространстве."
        )
        payload["interpretation_note"] = (
            "Проект исследует поведение retrieval-моделей на малой выборке и не доказывает клиническое превосходство ни одной из них."
        )
        payload["disclaimer"] = (
            "Ретроспективный исследовательский прототип. Не является медицинским изделием и не формирует клинические рекомендации."
        )
        return self._wrap_dashboard(payload, representation_size=int(self.ensure_loaded()["config"]["embedding_dim"]))

    def retrieve(self, query_window: dict[str, Any], query_vector: np.ndarray, k: int, beta: float, steps: int) -> dict[str, Any]:
        bundle = self.ensure_loaded()
        window_embeddings = np.asarray(bundle["window_embeddings"], dtype=float)
        memory_embeddings = np.asarray(bundle["memory_embeddings"], dtype=float)
        train_windows = [window for window in self.base_bundle["windows"] if window["split"] == "train" and window["usable_for_memory"]]
        index_by_window_id = self.base_bundle["index_by_window_id"]

        if query_window["window_id"] in index_by_window_id:
            query_embedding = np.asarray(window_embeddings[index_by_window_id[query_window["window_id"]]], dtype=float)
        else:
            encoder = self.base_bundle["encoder"]
            blocks = []
            for block_name in ("premeal_cgm", "premeal_delta", "missingness"):
                start, end = encoder.block_slices[block_name]
                blocks.append(np.asarray(query_vector[start:end], dtype=np.float32))
            sequence = np.stack(blocks, axis=0).astype(np.float32)
            tabular_blocks = []
            for block_name in ("meal_context", "time_context", "patient_context", "heart_rate_context"):
                start, end = encoder.block_slices[block_name]
                tabular_blocks.append(np.asarray(query_vector[start:end], dtype=np.float32))
            tabular = np.concatenate(tabular_blocks, axis=0).astype(np.float32)
            from .siamese import _encode_in_batches

            query_embedding = _encode_in_batches(
                self.ensure_encoder(),
                sequence.reshape(1, *sequence.shape).astype(np.float32),
                tabular.reshape(1, -1).astype(np.float32),
            )[0]

        similarities = memory_embeddings @ query_embedding
        weights = stable_softmax(float(beta) * similarities)
        top_indices = np.argsort(similarities)[::-1][:k]
        prototypes = bundle["prototypes"]
        prototype_distribution = self._prototype_distribution(query_embedding, prototypes)
        top_matches = []
        for rank, memory_index in enumerate(top_indices):
            candidate_window = train_windows[int(memory_index)]
            block_similarity = self._feature_block_similarity(query_vector, candidate_window["window_id"])
            top_blocks = sorted(block_similarity.items(), key=lambda entry: entry[1], reverse=True)[:3]
            top_matches.append(
                {
                    "index": int(rank),
                    "window_id": candidate_window["window_id"],
                    "label": candidate_window["label"],
                    "patient_id": candidate_window["patient_id"],
                    "similarity": float(similarities[int(memory_index)]),
                    "weight": float(weights[int(memory_index)]),
                    "same_patient": str(candidate_window["patient_id"]) == str(query_window["patient_id"]),
                    "window": candidate_window,
                    "feature_block_similarity": block_similarity,
                    "top_blocks": [[name, float(value)] for name, value in top_blocks],
                    "prototype_affinity": float(prototype_distribution.get(candidate_window["label"], 0.0)),
                    "explanation_text": _match_explanation(
                        query_window=query_window,
                        candidate_window=candidate_window,
                        top_blocks=top_blocks,
                        candidate_label=candidate_window["label"],
                        model_label=self.descriptor["label"],
                    ),
                }
            )

        sorted_similarities = np.sort(np.asarray([item["similarity"] for item in top_matches], dtype=float))
        top_similarity_gap = (
            float(sorted_similarities[-1] - sorted_similarities[-2]) if len(sorted_similarities) > 1 else float(sorted_similarities[-1]) if len(sorted_similarities) else 0.0
        )
        attention_entropy = float(-np.sum(np.clip(weights, 1e-12, 1.0) * np.log(np.clip(weights, 1e-12, 1.0)))) if len(weights) else 0.0
        level_key, level_label = _uncertainty_level(top_similarity_gap, attention_entropy)
        neighborhood_purity = float(np.mean([item["label"] == top_matches[0]["label"] for item in top_matches])) if top_matches else 0.0

        return {
            "model": self.descriptor,
            "query_window": query_window,
            "query_vector": query_embedding.tolist(),
            "recalled_vector": query_embedding.tolist(),
            "recalled_steps": [],
            "top_k_memories": json_ready(top_matches),
            "similarities": similarities.tolist(),
            "weights": weights.tolist(),
            "energy_values": [],
            "prototype_distribution": prototype_distribution,
            "uncertainty": {
                "level": level_key,
                "level_label": level_label,
                "top_similarity_gap": top_similarity_gap,
                "top_weight_gap": 0.0,
                "attention_entropy": attention_entropy,
                "prototype_gap": 0.0,
                "same_patient_share": float(np.mean([match["same_patient"] for match in top_matches])) if top_matches else 0.0,
                "summary_text": (
                    f"Запросное окно расположено рядом с группой сходных случаев в эмбеддинговом пространстве; neighborhood purity {neighborhood_purity:.3f}."
                ),
            },
            "trajectory_note": "Temporal-модель не использует recall; диагностика строится по соседству и gap между ближайшими эмбеддингами.",
            "retrieval_reason_note": "Соседи выбираются по cosine similarity в эмбеддинговом пространстве, а блоки признаков показывают, за счет каких компонент возникает сходство.",
            "prototype_note": "Прототипы здесь вторичны и используются только как агрегированные центры для отладки структуры пространства.",
            "plain_language_explanation_text": (
                "Temporal-энкодер переводит окно в эмбеддинг; затем retrieval выполняется по cosine similarity среди train-окон памяти."
            ),
        }

    def prototypes(self) -> list[dict[str, Any]]:
        bundle = self.ensure_loaded()
        return [
            {
                **prototype,
                "model": self.descriptor,
                "prototype_space_note": prototype.get("prototype_geometry_note", "Прототип описывает локальный центр эмбеддингового пространства."),
                "representative_windows": [self.window(window_id) for window_id in prototype["representative_window_ids"]],
            }
            for prototype in bundle["prototypes"].values()
        ]

    def prototype(self, label: str) -> dict[str, Any]:
        bundle = self.ensure_loaded()
        if label not in bundle["prototypes"]:
            raise KeyError(f"Неизвестная метка прототипа: {label}")
        prototype = bundle["prototypes"][label]
        return {
            **prototype,
            "model": self.descriptor,
            "prototype_space_note": prototype.get("prototype_geometry_note", "Прототип описывает локальный центр эмбеддингового пространства."),
            "representative_windows": [self.window(window_id) for window_id in prototype["representative_window_ids"]],
        }

    def evaluation(self) -> dict[str, Any]:
        bundle = self.ensure_loaded()
        return {
            **bundle["evaluation"],
            "model": self.descriptor,
            "comparison_note": (
                "Сравнение моделей носит исследовательский характер и не предназначено для клинической интерпретации."
            ),
        }

    def noise(self) -> list[dict[str, Any]]:
        return self.ensure_loaded()["chart_data"]["noise_robustness"]


@dataclass
class SOMEngine(RetrievalEngine):
    bundle: dict[str, Any] | None = None

    @property
    def descriptor(self) -> dict[str, Any]:
        return SOM_UI_DESCRIPTOR

    def ensure_loaded(self) -> dict[str, Any]:
        if self.bundle is None:
            try:
                self.bundle = load_som_bundle(self.settings, force=False)
            except FileNotFoundError:
                self.bundle = build_som_bundle(self.base_bundle, self.settings, force=True)
        return self.bundle

    def dashboard(self) -> dict[str, Any]:
        payload = dict(self.base_bundle["dashboard"])
        evaluation = self.evaluation()
        payload["title"] = self.settings.project_name
        payload["subtitle"] = self.settings.project_subtitle
        payload["headline_metrics"] = evaluation["retrieval_metrics"]
        payload["headline_summary"] = (
            f"Карта Кохонена показывает top-1 {evaluation['retrieval_metrics']['top1_accuracy'] * 100:.1f}% и top-3 {evaluation['retrieval_metrics']['top3_hit_rate'] * 100:.1f}% "
            "на отложенных окнах; ее сильная сторона — топологическая структура соседства и визуализация локальных областей."
        )
        payload["interpretation_note"] = (
            "SOM следует интерпретировать как исследовательскую карту структуры данных, а не как оптимальный retrieval-движок по умолчанию."
        )
        payload["disclaimer"] = (
            "Ретроспективный исследовательский прототип. Не является медицинским изделием и не формирует клинические рекомендации."
        )
        representation_size = int(self.ensure_loaded()["config"]["grid_height"]) * int(self.ensure_loaded()["config"]["grid_width"])
        return self._wrap_dashboard(payload, representation_size=representation_size)

    def retrieve(self, query_window: dict[str, Any], query_vector: np.ndarray, k: int, beta: float, steps: int) -> dict[str, Any]:
        bundle = self.ensure_loaded()
        train_windows = [window for window in self.base_bundle["windows"] if window["split"] == "train" and window["usable_for_memory"]]
        train_indices = [self.base_bundle["index_by_window_id"][window["window_id"]] for window in train_windows]
        train_matrix = np.asarray(self.base_bundle["feature_matrix"][train_indices], dtype=float)
        train_norm = np.asarray([_safe_normalize(row) for row in train_matrix], dtype=float)
        weights = np.asarray(bundle["weights"], dtype=float)
        train_assignments = np.asarray(bundle["train_assignments"], dtype=int)
        cell_stats = {int(key): value for key, value in bundle["cell_stats"].items()} if isinstance(bundle["cell_stats"], dict) else {
            int(item["cell_index"]): item for item in bundle["cell_stats"]
        }
        grid_shape = (int(bundle["config"]["grid_height"]), int(bundle["config"]["grid_width"]))
        query_vector = np.asarray(query_vector, dtype=float)
        query_norm = _safe_normalize(query_vector)

        distances = np.linalg.norm(weights - query_vector[None, :], axis=1)
        query_bmu = int(np.argmin(distances))
        second_bmu = int(np.argsort(distances)[1]) if len(distances) > 1 else query_bmu
        quantization_error = float(distances[query_bmu])
        topographic_error = 0.0 if _adjacent_cells(query_bmu, second_bmu, grid_shape) else 1.0

        feature_similarity = np.clip((train_norm @ query_norm + 1.0) / 2.0, 0.0, 1.0)
        map_distances = np.asarray([_map_distance(query_bmu, int(cell), grid_shape) for cell in train_assignments], dtype=float)
        local_similarity = feature_similarity * np.exp(-0.9 * map_distances)
        order = np.argsort(local_similarity)[::-1][:k]
        neighbors = []
        for rank, index in enumerate(order, start=1):
            candidate_window = train_windows[int(index)]
            candidate_cell = int(train_assignments[int(index)])
            same_patient = str(candidate_window["patient_id"]) == str(query_window["patient_id"])
            neighbors.append(
                {
                    "index": rank - 1,
                    "window_id": candidate_window["window_id"],
                    "label": candidate_window["label"],
                    "patient_id": candidate_window["patient_id"],
                    "similarity": float(local_similarity[int(index)]),
                    "weight": float(feature_similarity[int(index)]),
                    "same_patient": same_patient,
                    "window": candidate_window,
                    "feature_block_similarity": self._feature_block_similarity(query_vector, candidate_window["window_id"]),
                    "top_blocks": [],
                    "prototype_affinity": None,
                    "map_distance": float(map_distances[int(index)]),
                    "cell_index": candidate_cell,
                    "explanation_text": (
                        "Та же ячейка карты." if candidate_cell == query_bmu else "Соседняя топологическая область."
                    )
                    + (" Тот же пациент." if same_patient else " Другой пациент."),
                }
            )

        active_cell = cell_stats.get(query_bmu, {})
        level_key, level_label = _uncertainty_level(
            float(local_similarity[order[0]] - local_similarity[order[1]]) if len(order) > 1 else float(local_similarity[order[0]]) if len(order) else 0.0,
            float(topographic_error),
        )
        local_region_cells = _neighboring_cells(query_bmu, grid_shape)

        return {
            "model": self.descriptor,
            "query_window": query_window,
            "query_vector": query_vector.tolist(),
            "recalled_vector": weights[query_bmu].tolist(),
            "recalled_steps": [],
            "top_k_memories": json_ready(neighbors),
            "similarities": local_similarity.tolist(),
            "weights": feature_similarity.tolist(),
            "energy_values": [],
            "prototype_distribution": active_cell.get("label_distribution", {}),
            "uncertainty": {
                "level": level_key,
                "level_label": level_label,
                "top_similarity_gap": float(local_similarity[order[0]] - local_similarity[order[1]]) if len(order) > 1 else 0.0,
                "top_weight_gap": 0.0,
                "attention_entropy": 0.0,
                "prototype_gap": 0.0,
                "same_patient_share": float(np.mean([match["same_patient"] for match in neighbors])) if neighbors else 0.0,
                "summary_text": (
                    f"Запросное окно попадает в область карты с доминирующей меткой «{active_cell.get('dominant_label_display', 'не определено')}»; "
                    f"квантование {quantization_error:.3f}, топографическая ошибка {topographic_error:.3f}."
                ),
            },
            "trajectory_note": "Карта Кохонена не использует recall; здесь важны BMU, квантование и локальная топология.",
            "retrieval_reason_note": "Соседи выбираются по близости на карте и по локальному feature similarity внутри топологического neighborhood.",
            "prototype_note": "Для SOM вместо прототипов используются локальные области карты и их состав меток.",
            "plain_language_explanation_text": (
                "Карта Кохонена сопоставляет окно с локальной топологической областью, где сосредоточены похожие случаи train-памяти."
            ),
            "som_map": {
                "grid_height": grid_shape[0],
                "grid_width": grid_shape[1],
                "active_cell": query_bmu,
                "neighbor_cells": local_region_cells,
                "cells": list(cell_stats.values()),
                "quantization_error": quantization_error,
                "topographic_error": topographic_error,
            },
        }

    def prototypes(self) -> list[dict[str, Any]]:
        bundle = self.ensure_loaded()
        cells = list(bundle["cell_stats"].values()) if isinstance(bundle["cell_stats"], dict) else list(bundle["cell_stats"])
        ranked = sorted(cells, key=lambda item: (item["count"], item["purity"]), reverse=True)
        return ranked[:6]

    def prototype(self, label: str) -> dict[str, Any]:
        bundle = self.ensure_loaded()
        cells = list(bundle["cell_stats"].values()) if isinstance(bundle["cell_stats"], dict) else list(bundle["cell_stats"])
        match = next((cell for cell in cells if str(cell["cell_index"]) == str(label)), None)
        if match is None:
            raise KeyError(f"Неизвестная ячейка карты: {label}")
        return match

    def evaluation(self) -> dict[str, Any]:
        bundle = self.ensure_loaded()
        return {
            **bundle["evaluation"],
            "model": self.descriptor,
            "comparison_note": (
                "Сравнение моделей носит исследовательский характер и не предназначено для клинической интерпретации."
            ),
        }

    def noise(self) -> list[dict[str, Any]]:
        return self.ensure_loaded()["chart_data"]["noise_robustness"]


def create_engine_registry(settings: Settings, base_bundle: dict[str, Any]) -> dict[str, RetrievalEngine]:
    return {
        "hopfield": HopfieldEngine(settings=settings, base_bundle=base_bundle),
        "siamese_temporal": SiameseTemporalEngine(settings=settings, base_bundle=base_bundle),
        "som": SOMEngine(settings=settings, base_bundle=base_bundle),
    }
