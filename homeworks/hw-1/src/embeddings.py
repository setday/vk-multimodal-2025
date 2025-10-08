"""Encoders for images, captions, and text."""
from __future__ import annotations

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    CLIPModel,
    CLIPProcessor,
)

from .config import CONFIG
from .utils import torch_no_grad


def get_device() -> torch.device:
    cfg_device = CONFIG.models.device
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert cfg_device != "cuda" or torch.cuda.is_available(), "CUDA is not available"
    return torch.device(cfg_device)


class ImageEncoder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(CONFIG.models.image_encoder)
        self.processor = CLIPProcessor.from_pretrained(CONFIG.models.image_encoder, use_fast=True)

        self.model = self.model.to(get_device())

    def encode(self, images: list[Image.Image]) -> torch.Tensor:
        with torch_no_grad():
            inputs = self.processor(
                images=images,
                return_tensors="pt",
            ).to(get_device())
            outputs = self.model.get_image_features(**inputs)
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        return outputs.cpu()


class TextEncoder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(CONFIG.models.vlm_model)
        self.processor = CLIPProcessor.from_pretrained(CONFIG.models.vlm_model, use_fast=True)

        self.model = self.model.to(get_device())

    def encode(self, texts: list[str]) -> torch.Tensor:
        with torch_no_grad():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
            ).to(get_device())
            outputs = self.model.get_text_features(**inputs)
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        return outputs.cpu()


class CaptionGenerator:
    def __init__(self):
        self.model = BlipForConditionalGeneration.from_pretrained(CONFIG.models.caption_model)
        self.processor = AutoProcessor.from_pretrained(CONFIG.models.caption_model, use_fast=True)
        
        self.model = self.model.to(get_device())

    def generate(self, images: list[Image.Image], max_length: int = 64) -> list[str]:
        with torch_no_grad():
            inputs = self.processor(
                images=images,
                return_tensors="pt",
            ).to(get_device())
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,
            )
            captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return [caption.strip() for caption in captions]
