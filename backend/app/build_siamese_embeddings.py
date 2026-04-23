from __future__ import annotations

from .config import get_settings
from .pipeline import build_runtime_bundle
from .siamese import build_siamese_bundle


def main() -> None:
    settings = get_settings()
    base_bundle = build_runtime_bundle(force=False, settings=settings)
    bundle = build_siamese_bundle(base_bundle, settings=settings, force=False)
    print(f"Window embeddings: {settings.siamese_window_embeddings_path}")
    print(f"Memory embeddings: {settings.siamese_memory_embeddings_path}")
    print(f"Embedding dimension: {bundle['config']['embedding_dim']}")


if __name__ == "__main__":
    main()
