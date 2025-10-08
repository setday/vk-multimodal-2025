"""Publish generated captions to HuggingFace Datasets."""
from __future__ import annotations

import argparse

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from src.config import CONFIG


def publish(repo_id: str, private: bool = False, revision: str | None = None, api_token: str | None = None) -> None:
    paths = CONFIG.paths
    metadata_dir = paths.omni_metadata_path
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_dir}")

    df = pd.read_parquet(metadata_dir)[["id", "image_path", "caption"]]
    df["image_row"] = [
        None # FIXME: It should be a picture there, but I have no memory xD (Image.open(paths.root / x).convert("RGB"))
        for x in df["image_path"].to_list()
    ]
    df.drop(columns=["image_path"], inplace=True)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset_dict = DatasetDict({"train": dataset})

    api = HfApi(token=api_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    dataset_dict.push_to_hub(
        repo_id,
        revision=revision,
        token=api_token,
        private=private,
    )
    print(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish caption dataset to HuggingFace")
    parser.add_argument("repo_id", type=str, help="Destination repo id, e.g. username/wikiart-captions")
    parser.add_argument("--private", action="store_true", help="Create the dataset as private")
    parser.add_argument("--revision", type=str, default=None, help="Target revision or branch")
    parser.add_argument("--api_token", type=str, default=None, help="Hugging Face API token")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    publish(repo_id=args.repo_id, private=args.private, revision=args.revision, api_token=args.api_token)
