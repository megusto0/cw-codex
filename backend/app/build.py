from __future__ import annotations

from .baselines import evaluate_retrieval_baselines, save_baselines_cache
from .config import get_settings
from .pipeline import build_runtime_bundle, refresh_report_documents
from .siamese import build_siamese_bundle
from .som import build_som_bundle


def main() -> None:
    settings = get_settings()
    bundle = build_runtime_bundle(force=True, settings=settings)
    build_siamese_bundle(bundle, settings=settings, force=True)
    build_som_bundle(bundle, settings=settings, force=True)
    save_baselines_cache(settings, evaluate_retrieval_baselines(bundle, settings))
    refresh_report_documents(bundle, settings=settings)
    print(f"Artifacts refreshed in {settings.artifacts_dir}")


if __name__ == "__main__":
    main()
