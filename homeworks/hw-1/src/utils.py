"""Utility helpers for reproducibility and tensor handling."""
from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def torch_no_grad() -> Iterator[None]:
    with torch.no_grad():
        yield


def batched(iterable: Sequence | Iterable, batch_size: int) -> Iterator[list]:
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
