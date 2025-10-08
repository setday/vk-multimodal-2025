"""End-to-end pipeline for building multimodal retrieval assets."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from .config import CONFIG
from .data import WikiArtStream
from .embeddings import CaptionGenerator, ImageEncoder, TextEncoder
from .indexers import AnnoyVectorIndex
from .utils import batched, set_seed


def _ensure_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _collect_metadata(sample: dict) -> dict:
    cfg = CONFIG.dataset
    return {
        "id": str(sample.get(cfg.id_column, "")),
        "artist": sample.get(cfg.artist_column, "unknown"),
        "style": sample.get(cfg.style_column, "unknown"),
        "genre": sample.get(cfg.genre_column, "unknown"),
    }


def build_corpus(sample_size: int | None = None) -> dict:
    cfg = CONFIG
    cfg.prepare()
    set_seed(cfg.dataset.seed)

    sample_size = sample_size or cfg.dataset.sample_size
    paths = cfg.paths
    image_dir = paths.root / "artifacts" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_encoder = ImageEncoder()
    captioner = CaptionGenerator()
    text_encoder = TextEncoder()

    records: list[dict] = []
    image_embeddings: list[np.ndarray] = []
    caption_embeddings: list[np.ndarray] = []
    captions: list[str] = []
    ids: list[str] = []
    seen_ids: set[str] = set()

    dataset_iter = WikiArtStream(sample_size)
    iterator = tqdm(dataset_iter, total=sample_size, desc="Streaming WikiArt")
    batch_size = CONFIG.models.batch_size

    for batch in batched(iterator, batch_size):
        images: list[Image.Image] = []
        batch_metadata: list[dict] = []
        for sample in batch:
            meta = _collect_metadata(sample)

            meta["id"] = meta["id"] or f"sample_{len(records) + len(images)}"
            if meta["id"] in seen_ids:
                suffix = len(seen_ids)
                meta["id"] = f"{meta['id']}_{suffix}"

            seen_ids.add(meta["id"])
            ids.append(meta["id"])

            batch_metadata.append(meta)

            image = _ensure_image(sample[CONFIG.dataset.image_column])
            images.append(image)

            image_path = image_dir / f"{meta['id']}.jpg"
            image.save(image_path, "JPEG", quality=100)
            meta["image_path"] = str(image_path.relative_to(paths.root))
            
        img_feats = image_encoder.encode(images).numpy()
        image_embeddings.append(img_feats)

        batch_captions = captioner.generate(images)
        captions.extend(batch_captions)
        text_feats = text_encoder.encode(batch_captions).numpy()
        caption_embeddings.append(text_feats)

        for meta, caption in zip(batch_metadata, batch_captions):
            meta["caption"] = caption
            records.append(meta)

    image_matrix = np.vstack(image_embeddings)
    caption_matrix = np.vstack(caption_embeddings)

    metadata_df = pd.DataFrame(records)
    metadata_df.to_parquet(cfg.paths.omni_metadata_path, index=False)

    np.save(paths.embeddings_dir / "image_embeddings.npy", image_matrix)
    np.save(paths.embeddings_dir / "caption_embeddings.npy", caption_matrix)

    build_indexes(
        image_vectors=image_matrix,
        caption_vectors=caption_matrix,
        ids=ids,
    )

    def pathes_to_str(d):
        if isinstance(d, dict):
            return {k: pathes_to_str(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [pathes_to_str(v) for v in d]
        elif isinstance(d, tuple):
            return tuple(pathes_to_str(v) for v in d)
        elif isinstance(d, Path):
            return str(d)
        elif isinstance(d, ellipsis):
            return "..."
        else:
            return d
        
    with open(paths.root / "artifacts" / "config_snapshot.json", "w", encoding="utf-8") as fout:
        json.dump(
            {
                "config": pathes_to_str(asdict(cfg)),
                "num_items": len(ids),
            },
            fout,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "num_items": len(ids),
        "image_embeddings": image_matrix.shape,
        "caption_embeddings": caption_matrix.shape,
    }


def build_indexes(
    image_vectors: np.ndarray,
    caption_vectors: np.ndarray,
    ids: list[str],
) -> None:
    cfg = CONFIG
    paths = cfg.paths

    image_index_path = paths.indexes_dir / "image.ann"
    caption_index_path = paths.indexes_dir / "caption.ann"

    image_index = AnnoyVectorIndex(dimension=image_vectors.shape[1])
    image_index.build(image_vectors, ids)
    image_index.save(image_index_path)

    caption_index = AnnoyVectorIndex(dimension=caption_vectors.shape[1])
    caption_index.build(caption_vectors, ids)
    caption_index.save(caption_index_path)
