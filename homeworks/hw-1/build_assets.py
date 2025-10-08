"""CLI entrypoint to build embeddings, datasets, and indexes."""
from __future__ import annotations

from src.pipeline import build_corpus


def main() -> None:
    stats = build_corpus()
    print("Build complete")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
