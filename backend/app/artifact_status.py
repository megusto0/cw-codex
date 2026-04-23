from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from .config import Settings, get_settings


def _bundle_data_dir(bundle_path: Path) -> Path | None:
    if not bundle_path.exists():
        return None
    try:
        with bundle_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    settings_payload = payload.get("settings")
    if not isinstance(settings_payload, dict):
        return None
    data_dir = settings_payload.get("data_dir")
    if not data_dir:
        return None
    return Path(str(data_dir)).resolve()


def artifacts_match_current_data_source(settings: Settings) -> dict[str, Any]:
    expected_data_dir = settings.data_dir.resolve()
    bundle_data_dir = _bundle_data_dir(settings.runtime_bundle_path)
    required_paths = [
        settings.runtime_bundle_path,
        settings.siamese_runtime_bundle_path,
        settings.som_runtime_bundle_path,
        settings.comparison_metrics_path,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    matches = not missing_paths and bundle_data_dir == expected_data_dir
    return {
        "matches": matches,
        "expected_data_dir": str(expected_data_dir),
        "bundle_data_dir": str(bundle_data_dir) if bundle_data_dir is not None else None,
        "missing_paths": missing_paths,
    }


def main() -> int:
    status = artifacts_match_current_data_source(get_settings())
    print(json.dumps(status, ensure_ascii=False))
    return 0 if status["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
