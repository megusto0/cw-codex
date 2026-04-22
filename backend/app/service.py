from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings, get_settings
from .pipeline import DISPLAY_LABELS, build_runtime_bundle


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


@dataclass
class AppService:
    settings: Settings
    bundle: dict[str, Any] | None = None

    def ensure_bundle(self, force_refresh: bool = False) -> dict[str, Any]:
        if self.bundle is None or force_refresh:
            self.bundle = build_runtime_bundle(force=force_refresh, settings=self.settings)
        return self.bundle

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
        }

    def dashboard(self) -> dict[str, Any]:
        return self.ensure_bundle()["dashboard"]

    def windows(
        self,
        patient_id: str | None = None,
        label: str | None = None,
        meal_segment: str | None = None,
        split: str | None = None,
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        records = self.ensure_bundle()["windows"]
        filtered = []
        for record in records:
            if patient_id and str(record["patient_id"]) != str(patient_id):
                continue
            if label and record["label"] != label:
                continue
            if meal_segment and record["meal_segment"] != meal_segment:
                continue
            if split and record["split"] != split:
                continue
            filtered.append(record)
        filtered.sort(key=lambda item: item["meal_time"], reverse=True)
        return filtered[:limit]

    def window(self, window_id: str) -> dict[str, Any]:
        bundle = self.ensure_bundle()
        record = next((window for window in bundle["windows"] if window["window_id"] == window_id), None)
        if record is None:
            raise KeyError(f"Unknown window id: {window_id}")
        return record

    def retrieve(self, window_id: str, k: int, beta: float, steps: int) -> dict[str, Any]:
        bundle = self.ensure_bundle()
        query_window = self.window(window_id)
        memory_index = query_window.get("memory_index")
        if memory_index is None:
            raise ValueError("This meal window is excluded from memory retrieval because it is outside the core analysis set.")
        query_vector = bundle["feature_matrix"][int(memory_index)]
        return self._retrieve_from_vector(query_window, query_vector, k=k, beta=beta, steps=steps)

    def custom_query(self, payload: dict[str, Any], k: int, beta: float, steps: int) -> dict[str, Any]:
        bundle = self.ensure_bundle()
        query_window = bundle["encoder"].build_custom_window(payload)
        query_vector = bundle["encoder"].transform_window(query_window)
        return self._retrieve_from_vector(query_window, query_vector, k=k, beta=beta, steps=steps)

    def prototypes(self) -> list[dict[str, Any]]:
        return list(self.ensure_bundle()["prototypes"].values())

    def prototype(self, label: str) -> dict[str, Any]:
        bundle = self.ensure_bundle()
        if label not in bundle["prototypes"]:
            raise KeyError(f"Unknown prototype label: {label}")
        prototype = bundle["prototypes"][label]
        representative_windows = [self.window(window_id) for window_id in prototype["representative_window_ids"]]
        return {**prototype, "representative_windows": representative_windows}

    def evaluation(self) -> dict[str, Any]:
        return self.ensure_bundle()["evaluation"]

    def noise(self) -> list[dict[str, Any]]:
        return self.ensure_bundle()["chart_data"]["noise_robustness"]

    def about(self) -> dict[str, Any]:
        return self.ensure_bundle()["about"]

    def _retrieve_from_vector(
        self,
        query_window: dict[str, Any],
        query_vector: np.ndarray,
        k: int,
        beta: float,
        steps: int,
    ) -> dict[str, Any]:
        bundle = self.ensure_bundle()
        retrieval = bundle["memory_model"].retrieve(query_vector, k=k, beta=beta, steps=steps)
        prototype_distribution = self._prototype_distribution(np.asarray(retrieval["recalled_vector"], dtype=float))
        top_matches = []

        for item in retrieval["top_k"]:
            candidate_window = self.window(item["window_id"])
            candidate_vector = bundle["feature_matrix"][bundle["index_by_window_id"][item["window_id"]]]
            block_similarity = bundle["encoder"].block_similarity(np.asarray(query_vector), np.asarray(candidate_vector))
            top_blocks = sorted(block_similarity.items(), key=lambda entry: entry[1], reverse=True)[:3]
            explanation_text = self._explain_match(query_window, candidate_window, top_blocks, item["label"])
            top_matches.append(
                {
                    **item,
                    "same_patient": str(candidate_window["patient_id"]) == str(query_window["patient_id"]),
                    "window": candidate_window,
                    "feature_block_similarity": block_similarity,
                    "top_blocks": top_blocks,
                    "explanation_text": explanation_text,
                }
            )

        trajectory = retrieval["trajectory"]
        energy_values = [entry["energy"] for entry in trajectory]
        summary_text = self._build_summary_text(query_window, top_matches, prototype_distribution, energy_values)
        return {
            "query_window": query_window,
            "query_vector": retrieval["query_vector"],
            "recalled_vector": retrieval["recalled_vector"],
            "recalled_steps": trajectory,
            "top_k_memories": top_matches,
            "similarities": retrieval["similarities"],
            "weights": retrieval["weights"],
            "energy_values": energy_values,
            "prototype_distribution": prototype_distribution,
            "plain_language_explanation_text": summary_text,
        }

    def _prototype_distribution(self, vector: np.ndarray) -> dict[str, float]:
        bundle = self.ensure_bundle()
        prototypes = bundle["prototypes"]
        if not prototypes:
            return {}
        prototype_labels = list(prototypes.keys())
        prototype_matrix = np.asarray([prototypes[label]["vector"] for label in prototype_labels], dtype=float)
        prototype_matrix = prototype_matrix / np.maximum(np.linalg.norm(prototype_matrix, axis=1, keepdims=True), 1e-12)
        normalized_vector = vector / max(np.linalg.norm(vector), 1e-12)
        similarities = prototype_matrix @ normalized_vector
        logits = np.exp(similarities - np.max(similarities))
        weights = logits / np.sum(logits)
        return {label: float(weight) for label, weight in zip(prototype_labels, weights)}

    def _explain_match(
        self,
        query_window: dict[str, Any],
        candidate_window: dict[str, Any],
        top_blocks: list[tuple[str, float]],
        candidate_label: str,
    ) -> str:
        sentences = []
        block_names = [block for block, _ in top_blocks]
        if "premeal_cgm" in block_names:
            sentences.append("The pre-meal CGM shape is closely aligned.")
        if "meal_context" in block_names:
            sentences.append("Carbs, bolus, baseline, and trend context are similar.")
        if "heart_rate_context" in block_names and not candidate_window["heart_rate_missing"]:
            sentences.append("Wearable heart-rate context also lines up.")
        if str(candidate_window["patient_id"]) == str(query_window["patient_id"]):
            sentences.append("This is a same-patient memory.")
        else:
            sentences.append("This is a cross-patient memory, which makes the match less patient-specific.")
        if candidate_label in DISPLAY_LABELS:
            sentences.append(f"It sits nearest to the {DISPLAY_LABELS[candidate_label].lower()} pattern.")
        return " ".join(sentences)

    def _build_summary_text(
        self,
        query_window: dict[str, Any],
        top_matches: list[dict[str, Any]],
        prototype_distribution: dict[str, float],
        energy_values: list[float],
    ) -> str:
        dominant_prototype = next(iter(sorted(prototype_distribution.items(), key=lambda item: item[1], reverse=True)), None)
        if top_matches:
            lead_match = top_matches[0]
            lead_phrase = (
                f"The strongest memory is {DISPLAY_LABELS.get(lead_match['label'], lead_match['label']).lower()} "
                f"from patient {lead_match['patient_id']} with similarity {lead_match['similarity']:.2f}."
            )
        else:
            lead_phrase = "No stored memories were returned."
        prototype_phrase = (
            f"The recalled vector is closest to the {DISPLAY_LABELS.get(dominant_prototype[0], dominant_prototype[0]).lower()} prototype."
            if dominant_prototype
            else "No prototype distribution is available."
        )
        if energy_values:
            energy_phrase = f"Energy moved from {energy_values[0]:.2f} to {energy_values[-1]:.2f} over recall steps."
        else:
            energy_phrase = "No recall steps were recorded."
        return " ".join([lead_phrase, prototype_phrase, energy_phrase, "This is retrospective analysis only, not clinical advice."])


_SERVICE: AppService | None = None


def get_service(force_refresh: bool = False) -> AppService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AppService(settings=get_settings())
    _SERVICE.ensure_bundle(force_refresh=force_refresh)
    return _SERVICE

