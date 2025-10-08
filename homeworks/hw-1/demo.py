"""Interactive Gradio demo for multimodal art retrieval."""
from __future__ import annotations

from functools import lru_cache

import gradio as gr

from src.retrieval import RetrievalService
import pandas as pd
from pathlib import Path


@lru_cache(maxsize=1)
def get_service() -> RetrievalService:
    return RetrievalService()


def _load_mapping(mapping_dir: Path) -> dict[str, dict[str, str]]:
    mapping = {}
    for name in ["artist", "genre", "style"]:
        path = mapping_dir / f"{name}.csv"

        try:
            df = pd.read_csv(path)["name"].to_list()
            mapping[name] = {i: v for i, v in enumerate(df)}
        except Exception as e:
            print(f"Error loading {path}: {e}")
            mapping[name] = {}
    return mapping

MAPPING = _load_mapping(Path("artifacts/type_mappings"))

def set_new_mapping_location(new_path: Path) -> None:
    global MAPPING
    MAPPING = _load_mapping(new_path)

def _format_results(results: list[dict]) -> list[tuple[str, str]]:
    gallery_items: list[tuple[str, str]] = []
    for item in results:
        artist_disp = MAPPING["artist"].get(item.get("artist", 0), "unknown")
        style_disp = MAPPING["style"].get(item.get("style", 0), "unknown")
        genre_disp = MAPPING["genre"].get(item.get("genre", 0), "unknown")
        image_path = item.get("image_path", "")
        caption = item.get("caption", "")
        tags_display = ", ".join(item.get("tags", []) or [])

        gallery_caption_parts = [artist_disp]
        if style_disp != "unknown":
            gallery_caption_parts.append(style_disp)
        if genre_disp != "unknown":
            gallery_caption_parts.append(genre_disp)
        if caption:
            gallery_caption_parts.append(caption)
        if tags_display:
            gallery_caption_parts.append(f"Tags: {tags_display}")
        gallery_items.append((image_path, " | ".join(gallery_caption_parts)))
    return gallery_items


def image_to_image(query_image: str | None, top_k: int) -> list[tuple[str, str]]:
    if not query_image:
        return []
    service = get_service()
    results = service.search_similar_images(query_image, top_k=top_k)
    return _format_results(results)


def caption_to_image(query_text: str, top_k: int) -> list[tuple[str, str]]:
    if not query_text.strip():
        return []
    service = get_service()
    results = service.search_by_caption(query_text, top_k=top_k)
    return _format_results(results)


def omni_to_image(
    query_text: str,
    styles: list[str],
    genres: list[str],
    tags: list[str],
    top_k: int,
) -> list[tuple[str, str]]:
    if not any([query_text.strip(), styles, genres, tags]):
        return []
    service = get_service()
    results = service.search_omni(
        text_query=query_text.strip() or None,
        styles=styles,
        genres=genres,
        extra_tags=tags,
        top_k=top_k,
    )
    return _format_results(results)


def build_demo() -> gr.Blocks:
    service = get_service()
    metadata = service.metadata.reset_index()
    styles = sorted({style for style in metadata["style"].dropna().unique() if style})
    genres = sorted({genre for genre in metadata["genre"].dropna().unique() if genre})
    tags = service.omni_tags

    style_choices = [(style, MAPPING["style"].get(style, "unknown")) for style in styles]
    genre_choices = [(genre, MAPPING["genre"].get(genre, "unknown")) for genre in genres]

    with gr.Blocks(title="WikiArt Multimodal Retrieval") as demo:
        gr.Markdown(
            """
            # WikiArt Multimodal Retrieval
            """
        )

        with gr.Tab("Image -> Image"):
            with gr.Row():
                image_input = gr.Image(type="filepath", label="Запросное изображение")
                top_k_slider = gr.Slider(3, 20, value=10, step=1, label="Top-K")
            image_gallery = gr.Gallery(label="Изображения", columns=5, height="auto")
            run_btn = gr.Button("Найти похожие")
            run_btn.click(
                image_to_image,
                inputs=[image_input, top_k_slider],
                outputs=image_gallery,
            )

        with gr.Tab("Caption -> Image"):
            with gr.Row():
                caption_input = gr.Textbox(label="Текстовый запрос", lines=3)
                caption_topk = gr.Slider(3, 20, value=10, step=1, label="Top-K")
            caption_gallery = gr.Gallery(label="Изображения", columns=5, height="auto")
            caption_run = gr.Button("Найти по описанию")
            caption_run.click(
                caption_to_image,
                inputs=[caption_input, caption_topk],
                outputs=caption_gallery,
            )

        with gr.Tab("Omni Search"):
            query_box = gr.Textbox(label="Свободный запрос", lines=2)
            with gr.Row():
                style_select = gr.CheckboxGroup(choices=style_choices, label="Стиль")
                genre_select = gr.CheckboxGroup(choices=genre_choices, label="Жанр")
                tag_select = gr.CheckboxGroup(choices=tags, label="Zero-Shot теги")
            omni_topk = gr.Slider(3, 20, value=10, step=1, label="Top-K")
            omni_gallery = gr.Gallery(label="Изображения", columns=5, height="auto")
            omni_run = gr.Button("Объединить запрос")
            omni_run.click(
                omni_to_image,
                inputs=[query_box, style_select, genre_select, tag_select, omni_topk],
                outputs=omni_gallery,
            )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
