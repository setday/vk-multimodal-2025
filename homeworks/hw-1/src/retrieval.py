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


class RetrievalService:
    def __init__(self) -> None:
        cfg = CONFIG
        cfg.prepare()
        paths = cfg.paths
        
        def _parse_tags(value: object) -> list[str]:
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except json.JSONDecodeError:
                    return [part.strip() for part in value.split(",") if part.strip()]
            elif isinstance(value, (list, tuple, set)):
                return [str(item) for item in value]
            return []

        metadata_df = pd.read_parquet(paths.omni_metadata_path).set_index("id")
        if "tags" in metadata_df.columns:
            metadata_df["tags"] = metadata_df["tags"].apply(_parse_tags)
        else:
            metadata_df["tags"] = [[] for _ in range(len(metadata_df))]
        metadata_df["style"] = metadata_df["style"].fillna("unknown")
        metadata_df["genre"] = metadata_df["genre"].fillna("unknown")
        self.metadata = metadata_df
        self.id_to_idx = {sample_id: idx for idx, sample_id in enumerate(self.metadata.index)}

        self.omni_tag_matrix = np.load(paths.indexes_dir / "labels" / "omni_tag_embeddings.npy")
        with open(paths.indexes_dir / "labels" / "omni_tags.json", "r", encoding="utf-8") as fin:
            self.omni_tags: list[str] = json.load(fin)

        self.caption_embeddings = np.load(paths.embeddings_dir / "caption_embeddings.npy")
        self.image_embeddings = np.load(paths.embeddings_dir / "image_embeddings.npy")

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

    def search_omni(
        self,
        text_query: str | None = None,
        styles: Optional[Iterable[int]] = None,
        genres: Optional[Iterable[int]] = None,
        extra_tags: Optional[Iterable[str]] = None,
        top_k: int | None = None,
    ) -> list[dict]:
        candidate_ids = self._filter_candidates(
            styles=styles or [],
            genres=genres or [],
            extra_tags=extra_tags or [],
        )
        if not candidate_ids:
            return []

        candidate_vectors = self.caption_embeddings[[self.id_to_idx[sample_id] for sample_id in candidate_ids]]
        
        if text_query:
            temp_index = AnnoyVectorIndex(dimension=candidate_vectors.shape[1]) # Building runs quickly, so we can afford it in inference
            temp_index.build(candidate_vectors, candidate_ids)

            requested_top_k = top_k or CONFIG.index.top_k
            requested_top_k = min(requested_top_k, len(candidate_ids))

            text_embedding = self.text_encoder.encode([text_query]).numpy()[0]

            results = temp_index.query(text_embedding, top_k=requested_top_k)
        else:
            results = [(sample_id, 0.0) for sample_id in candidate_ids[: top_k or len(candidate_ids)]]

        formatted = [self._metadata_for(sample_id) | {"distance": float(distance)} for sample_id, distance in results]
        return formatted

    def _filter_candidates(
        self,
        styles: Iterable[int],
        genres: Iterable[int],
        extra_tags: Iterable[str],
    ) -> list[str]:
        df = self.metadata
        mask = pd.Series(True, index=df.index)
        if styles:
            mask &= df["style"].isin(list(styles))
        if genres:
            mask &= df["genre"].isin(list(genres))
        if extra_tags:
            required_tags = set(extra_tags)
            mask &= df["tags"].apply(lambda tags: required_tags.issubset(set(tags)))
        return df.index[mask].tolist()
