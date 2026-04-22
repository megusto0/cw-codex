from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class Settings:
    project_name: str = "RL Therapy Lab - Hopfield Postprandial Memory"
    version: str = "0.1.0"
    premeal_minutes: int = 90
    postmeal_minutes: int = 180
    grid_minutes: int = 5
    retrieval_top_k: int = 5
    hopfield_beta: float = 8.0
    recall_steps: int = 3
    data_dir: Path = WORKSPACE_ROOT / "OhioT1DM"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    reports_dir: Path = PROJECT_ROOT / "artifacts" / "reports"
    datasets_dir: Path = PROJECT_ROOT / "artifacts" / "datasets"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    runtime_bundle_path: Path = PROJECT_ROOT / "artifacts" / "models" / "runtime_bundle.pkl"
    latest_metrics_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "latest_metrics.json"
    latest_report_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "latest_report.md"
    chart_data_path: Path = PROJECT_ROOT / "artifacts" / "reports" / "chart_data.json"
    windows_dataset_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "meal_windows.csv"
    windows_json_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "meal_windows.json"
    feature_matrix_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "feature_matrix.npy"
    feature_metadata_path: Path = PROJECT_ROOT / "artifacts" / "datasets" / "feature_metadata.json"
    hopfield_weights_path: Path = PROJECT_ROOT / "artifacts" / "models" / "hopfield_memory.npz"
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
        data_dir = Path(os.environ.get("OHIO_DATA_DIR", str(WORKSPACE_ROOT / "OhioT1DM"))).resolve()
        _SETTINGS = Settings(data_dir=data_dir)
    return _SETTINGS
