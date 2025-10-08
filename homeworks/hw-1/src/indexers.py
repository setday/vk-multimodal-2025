"""Index builders based on Annoy."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from annoy import AnnoyIndex

from .config import CONFIG


class AnnoyVectorIndex:
    def __init__(self, dimension: int, metric: str | None = None, n_trees: int | None = None) -> None:
        self.dimension = dimension
        self.metric = metric or CONFIG.index.metric
        self.n_trees = n_trees or CONFIG.index.n_trees

        print(f"Initializing Annoy index: dim={self.dimension}, metric={self.metric}, n_trees={self.n_trees}")
        self.index = AnnoyIndex(self.dimension, self.metric)
        self.id_map = []

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        if len(vectors) != len(ids):
            raise ValueError("Vectors and ids length mismatch")
        
        self.id_map = []
        for idx, (vector, sample_id) in enumerate(zip(vectors, ids)):
            self.index.add_item(idx, vector.tolist())
            self.id_map.append(sample_id)

        self.index.build(self.n_trees)

    def load(self, path: Path) -> None:
        self.index.load(str(path))
        metadata_path = path.with_suffix(".ids.npy")
        self.id_map = np.load(metadata_path, allow_pickle=True).tolist()

    def save(self, path: Path) -> None:
        self.index.save(str(path))
        np.save(
            path.with_suffix(".ids.npy"),
            np.array(self.id_map, dtype=object),
        )

    def query(self, vector: np.ndarray, top_k: int | None = None) -> list[tuple[str, float]]:
        assert vector.shape == (self.dimension,), f"Vector shape mismatch: expected ({self.dimension},), got {vector.shape}"
        assert self.id_map, "Index not built or loaded"

        top_k = top_k or CONFIG.index.top_k

        indices, distances = self.index.get_nns_by_vector(
            vector=vector.tolist(),
            n=top_k,
            # search_k=CONFIG.index.search_k,
            include_distances=True,
        )

        results: list[tuple[str, float]] = []
        for idx, distance in zip(indices, distances):
            sample_id = self.id_map[idx]
            results.append((sample_id, distance))
        return results
