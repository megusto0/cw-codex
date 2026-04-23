from __future__ import annotations

from .baselines import evaluate_retrieval_baselines, save_baselines_cache
from .config import get_settings
from .evaluation_quality import audit_som, compute_seed_stability, refresh_evaluation_artifacts
from .pipeline import build_runtime_bundle, refresh_report_documents
from .siamese import build_siamese_bundle
from .som import build_som_bundle


def main() -> None:
    settings = get_settings()
    bundle = build_runtime_bundle(force=True, settings=settings)
    build_siamese_bundle(bundle, settings=settings, force=True)
    build_som_bundle(bundle, settings=settings, force=True)
    baseline_rows = evaluate_retrieval_baselines(bundle, settings)
    save_baselines_cache(settings, baseline_rows)
    compute_seed_stability(bundle, settings=settings)
    audit_som(bundle, settings=settings)
    refresh_evaluation_artifacts(bundle, settings=settings, baseline_rows=baseline_rows)
    refresh_report_documents(bundle, settings=settings)
    print(f"Artifacts refreshed in {settings.artifacts_dir}")


if __name__ == "__main__":
    main()
