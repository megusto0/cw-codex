from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def resolve_default_data_dir(workspace_root: Path = WORKSPACE_ROOT) -> Path:
    env_data_dir = os.environ.get("OHIO_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir).resolve()

    for candidate_name in ("data", "OhioT1DM"):
        candidate = (workspace_root / candidate_name).resolve()
        if candidate.exists():
            return candidate

    return (workspace_root / "OhioT1DM").resolve()


@dataclass(frozen=True)
class Settings:
    project_name: str = "Postprandial Retrieval Lab"
    project_subtitle: str = "Сравнение нейронных сетей в задаче поиска сходных постпрандиальных CGM-окон"
    version: str = "0.1.0"
    random_seed: int = 7
    premeal_minutes: int = 90
    postmeal_minutes: int = 180
    grid_minutes: int = 5
    retrieval_top_k: int = 5
    hopfield_beta: float = 8.0
    recall_steps: int = 3
    siamese_embedding_dim: int = 48
    siamese_epochs: int = 70
    siamese_learning_rate: float = 0.001
    siamese_weight_decay: float = 0.0001
    siamese_temperature: float = 0.2
    siamese_similarity_beta: float = 10.0
    som_grid_height: int = 5
    som_grid_width: int = 5
    som_epochs: int = 45
    som_learning_rate: float = 0.35
    som_sigma: float = 2.2
    data_dir: Path = WORKSPACE_ROOT / "OhioT1DM"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    reports_dir: Path = PROJECT_ROOT / "artifacts" / "reports"
    datasets_dir: Path = PROJECT_ROOT / "artifacts" / "datasets"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    runtime_bundle_path: Path = PROJECT_ROOT / "artifacts" / "models" / "runtime_bundle.pkl"
    latest_metrics_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "latest_metrics.json"
    latest_report_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "latest_report.md"
    chart_data_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "chart_data.json"
    comparison_metrics_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "comparison_metrics.json"
    latest_eval_summary_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "latest_eval_summary.json"
    latest_baselines_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "latest_baselines.json"
    latest_comparison_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "latest_comparison.md"
    som_audit_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "som_audit.md"
    seed_stability_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "seed_stability.json"
    seed_stability_report_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "seed_stability.md"
    windows_dataset_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "meal_windows.csv"
    windows_json_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "meal_windows.json"
    feature_matrix_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "feature_matrix.npy"
    feature_metadata_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "feature_metadata.json"
    hopfield_weights_path: Path = PROJECT_ROOT / "artifacts" / "models" / "hopfield_memory.npz"
    siamese_state_path: Path = PROJECT_ROOT / "artifacts" / "models" / "siamese_encoder.pt"
    siamese_config_path: Path = PROJECT_ROOT / "artifacts" / "models" / "siamese_config.json"
    siamese_runtime_bundle_path: Path = PROJECT_ROOT / "artifacts" / "models" / "siamese_runtime_bundle.pkl"
    siamese_window_embeddings_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "siamese_window_embeddings.npy"
    siamese_memory_embeddings_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "siamese_memory_embeddings.npy"
    siamese_test_embeddings_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "siamese_test_embeddings.npy"
    siamese_metrics_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "siamese_metrics.json"
    siamese_prototypes_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "siamese_prototypes.json"
    siamese_report_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "siamese_report.md"
    som_runtime_bundle_path: Path = PROJECT_ROOT / "artifacts" / "models" / "som_runtime_bundle.pkl"
    som_weights_path: Path = PROJECT_ROOT / "artifacts" / "models" / "som_weights.npy"
    som_assignments_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "som_assignments.json"
    som_metrics_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "som_metrics.json"
    som_report_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "som_report.md"
    coursework_report_path: Path = PROJECT_ROOT / "docs" / "coursework_report.md"

    def as_public_dict(self) -> dict:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        data_dir = resolve_default_data_dir()
        _SETTINGS = Settings(data_dir=data_dir)
    return _SETTINGS
