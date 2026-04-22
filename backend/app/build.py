from __future__ import annotations

from .config import get_settings
from .pipeline import build_runtime_bundle


def main() -> None:
    settings = get_settings()
    build_runtime_bundle(force=True, settings=settings)
    print(f"Artifacts refreshed in {settings.artifacts_dir}")


if __name__ == "__main__":
    main()
