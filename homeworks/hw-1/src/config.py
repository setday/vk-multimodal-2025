"""Global configuration for the multimodal retrieval MVP."""
from dataclasses import dataclass, field
from pathlib import Path
from types import EllipsisType


@dataclass(frozen=True)
class Paths:
    """Paths used across the project."""

    root: Path = Path(__file__).resolve().parents[1]
    cache_dir: Path = root / "cache"
    embeddings_dir: Path = root / "artifacts" / "embeddings"
    indexes_dir: Path = root / "artifacts" / "indexes"
    omni_metadata_path: Path = root / "artifacts" / "datasets" / "omni_metadata.parquet"

    def ensure(self) -> None:
        for path in [
            self.cache_dir,
            self.embeddings_dir,
            self.indexes_dir,
            self.omni_metadata_path.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset parameters."""

    name: str = "huggan/wikiart"
    split: str = "train"
    streaming: bool = True
    seed: int = 42
    sample_size: int = 5000
    shuffle_buffer: int = 2048
    image_column: str = "image"
    id_column: str = "id"
    artist_column: str = "artist"
    style_column: str = "style"
    genre_column: str = "genre"


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers and hyper-parameters."""

    image_encoder: str = "openai/clip-vit-base-patch32"
    caption_model: str = "Salesforce/blip-image-captioning-large"
    vlm_model: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"
    batch_size: int = 8


@dataclass(frozen=True)
class IndexConfig:
    """Parameters for vector indexes."""

    metric: str = "angular"
    n_trees: int = 64
    search_k: int | EllipsisType = ...
    top_k: int = 10


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration dataclass grouping all project level settings."""

    paths: Paths = field(default_factory=Paths)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

    def prepare(self) -> None:
        self.paths.ensure()


CONFIG = RetrievalConfig()
