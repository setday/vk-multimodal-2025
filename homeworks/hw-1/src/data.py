"""Dataset streaming utilities."""
from __future__ import annotations

from typing import Iterator

from datasets import IterableDataset, load_dataset

from .config import CONFIG
from .utils import set_seed


class WikiArtStream:
    """Wrapper around streaming dataset with deterministic shuffle."""

    def __init__(self, sample_size: int | None = None) -> None:
        self.cfg = CONFIG.dataset
        self.sample_size = sample_size or self.cfg.sample_size

        set_seed(self.cfg.seed)
        
        streaming_ds = load_dataset(
            self.cfg.name,
            split=self.cfg.split,
            streaming=self.cfg.streaming,
            trust_remote_code=True,
        )
        assert isinstance(streaming_ds, IterableDataset)
        self.dataset: IterableDataset = streaming_ds.shuffle(
            seed=self.cfg.seed,
            buffer_size=self.cfg.shuffle_buffer,
        )

    def __iter__(self) -> Iterator[dict]:
        for idx, sample in enumerate(self.dataset):
            if idx >= self.sample_size:
                break
            yield sample
