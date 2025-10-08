"""Runtime retrieval service utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image

from .config import CONFIG
from .embeddings import ImageEncoder, TextEncoder
from .indexers import AnnoyVectorIndex


@dataclass
class RetrievalAssets:
    metadata: pd.DataFrame
    image_index: AnnoyVectorIndex
    caption_index: AnnoyVectorIndex
    image_embeddings: np.ndarray
    caption_embeddings: np.ndarray


class RetrievalService:
    def __init__(self) -> None:
        cfg = CONFIG
        cfg.prepare()
        paths = cfg.paths

        metadata_df = pd.read_parquet(paths.omni_metadata_path).set_index("id")
        self.metadata = metadata_df

        self.image_embeddings = np.load(paths.embeddings_dir / "image_embeddings.npy")
        self.caption_embeddings = np.load(paths.embeddings_dir / "caption_embeddings.npy")

        self.image_index = AnnoyVectorIndex(
            dimension=self.image_embeddings.shape[1],
        )
        self.image_index.load(paths.indexes_dir / "image.ann")

        self.caption_index = AnnoyVectorIndex(
            dimension=self.caption_embeddings.shape[1],
        )
        self.caption_index.load(paths.indexes_dir / "caption.ann")

        self.paths = paths
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def _resolve_image(self, reference: str | Path | Image.Image) -> Image.Image:
        if isinstance(reference, Image.Image):
            return reference
        path = Path(reference)
        if not path.is_absolute():
            path = self.paths.root / path
        image = Image.open(path)
        return image

    def _metadata_for(self, sample_id: str) -> dict:
        row = self.metadata.loc[sample_id].to_dict()
        row["id"] = sample_id
        image_path = row.get("image_path")
        if image_path:
            row["image_path"] = str((self.paths.root / image_path).resolve())
        return row

    def search_similar_images(self, image: str | Path | Image.Image, top_k: int | None = None) -> list[dict]:
        candidate_image = self._resolve_image(image)
        embedding = self.image_encoder.encode([candidate_image]).numpy()[0]
        results = self.image_index.query(embedding, top_k=top_k)
        return [self._metadata_for(sample_id) | {"distance": float(distance)} for sample_id, distance in results]

    def search_by_caption(self, caption: str, top_k: int | None = None) -> list[dict]:
        embedding = self.text_encoder.encode([caption]).numpy()[0]
        results = self.caption_index.query(embedding, top_k=top_k)
        return [self._metadata_for(sample_id) | {"distance": float(distance)} for sample_id, distance in results]
