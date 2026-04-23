from __future__ import annotations

import json

from .config import get_settings
from .pipeline import build_runtime_bundle
from .siamese import build_siamese_bundle


def main() -> None:
    settings = get_settings()
    base_bundle = build_runtime_bundle(force=False, settings=settings)
    bundle = build_siamese_bundle(base_bundle, settings=settings, force=False)
    print(json.dumps(bundle["evaluation"]["retrieval_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
