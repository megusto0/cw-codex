from __future__ import annotations

from .config import get_settings
from .pipeline import build_runtime_bundle
from .siamese import build_siamese_bundle


def main() -> None:
    settings = get_settings()
    base_bundle = build_runtime_bundle(force=False, settings=settings)
    build_siamese_bundle(base_bundle, settings=settings, force=True)
    print(f"Siamese encoder trained and saved to {settings.siamese_state_path}")


if __name__ == "__main__":
    main()
