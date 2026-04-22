from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


@dataclass
class ContinuousHopfieldMemory:
    memory_matrix: np.ndarray | None = None
    metadata: list[dict[str, Any]] | None = None

    def fit(self, x_memory: np.ndarray, metadata: list[dict[str, Any]]) -> "ContinuousHopfieldMemory":
        if x_memory.ndim != 2:
            raise ValueError("x_memory must be a 2D matrix")
        self.memory_matrix = l2_normalize(np.asarray(x_memory, dtype=float))
        self.metadata = metadata
        return self

    def _require_fitted(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if self.memory_matrix is None or self.metadata is None:
            raise RuntimeError("ContinuousHopfieldMemory is not fitted")
        return self.memory_matrix, self.metadata

    def energy(self, query: np.ndarray, beta: float = 8.0) -> float:
        memory_matrix, _ = self._require_fitted()
        q = l2_normalize(np.asarray(query, dtype=float).reshape(1, -1))[0]
        similarities = beta * (memory_matrix @ q)
        return float(-logsumexp(similarities) / beta + 0.5 * np.dot(q, q))

    def recall(self, query: np.ndarray, steps: int = 3, beta: float = 8.0) -> dict[str, Any]:
        memory_matrix, metadata = self._require_fitted()
        current = l2_normalize(np.asarray(query, dtype=float).reshape(1, -1))[0]
        trajectory: list[dict[str, Any]] = []

        for step in range(steps):
            similarities = memory_matrix @ current
            weights = stable_softmax(beta * similarities)
            recalled = l2_normalize((weights @ memory_matrix).reshape(1, -1))[0]
            entropy = float(-np.sum(np.clip(weights, 1e-12, 1.0) * np.log(np.clip(weights, 1e-12, 1.0))))
            top_weights = np.sort(weights)[-2:]
            gap = float(top_weights[-1] - top_weights[-2]) if len(top_weights) > 1 else float(top_weights[-1])
            dominant_index = int(np.argmax(weights))
            trajectory.append(
                {
                    "step": step + 1,
                    "vector": recalled.tolist(),
                    "energy": self.energy(current, beta=beta),
                    "entropy": entropy,
                    "top_weight_gap": gap,
                    "dominant_memory": metadata[dominant_index],
                }
            )
            current = recalled

        final_similarities = memory_matrix @ current
        final_weights = stable_softmax(beta * final_similarities)
        return {
            "query_vector": l2_normalize(np.asarray(query, dtype=float).reshape(1, -1))[0].tolist(),
            "trajectory": trajectory,
            "final_vector": current.tolist(),
            "final_similarities": final_similarities,
            "final_weights": final_weights,
        }

    def get_top_k(self, query: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        memory_matrix, metadata = self._require_fitted()
        q = l2_normalize(np.asarray(query, dtype=float).reshape(1, -1))[0]
        similarities = memory_matrix @ q
        indices = np.argsort(similarities)[::-1][:k]
        return [
            {
                "index": int(index),
                "similarity": float(similarities[index]),
                "metadata": metadata[int(index)],
            }
            for index in indices
        ]

    def retrieve(self, query: np.ndarray, k: int = 5, beta: float = 8.0, steps: int = 3) -> dict[str, Any]:
        memory_matrix, metadata = self._require_fitted()
        recall_result = self.recall(query, steps=steps, beta=beta)
        similarities = np.asarray(recall_result["final_similarities"], dtype=float)
        weights = np.asarray(recall_result["final_weights"], dtype=float)
        indices = np.argsort(similarities)[::-1][:k]
        top_k = []
        for index in indices:
            index = int(index)
            top_k.append(
                {
                    "index": index,
                    "window_id": metadata[index]["window_id"],
                    "label": metadata[index]["label"],
                    "patient_id": metadata[index]["patient_id"],
                    "similarity": float(similarities[index]),
                    "weight": float(weights[index]),
                }
            )
        return {
            "query_vector": recall_result["query_vector"],
            "recalled_vector": recall_result["final_vector"],
            "trajectory": recall_result["trajectory"],
            "top_k": top_k,
            "similarities": similarities.tolist(),
            "weights": weights.tolist(),
        }

    def save(self, path: str | Path) -> None:
        memory_matrix, metadata = self._require_fitted()
        base = Path(path)
        np.savez_compressed(base.with_suffix(".npz"), memory_matrix=memory_matrix)
        base.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ContinuousHopfieldMemory":
        base = Path(path)
        data = np.load(base.with_suffix(".npz"))
        metadata = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(memory_matrix=data["memory_matrix"], metadata=metadata)

