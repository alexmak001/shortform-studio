"""Video renderer for Duo Mode Shorts."""

from __future__ import annotations

import os
import random
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def five_word_caption_clips(
    text: str,
    start: float,
    duration: float,
    W: int,
    H: int,
    *,
    speaker: str | None = None,
    words_per_chunk: int = 5,
    font_size: int = 64,
    font: str | None = None,
    y_pos: int | None = None,
    stroke_width: int = 3,
    min_chunk_sec: float = 0.18,
    overlap_sec: float = 0.05,
) -> list:
    if not text or not text.strip():
        return []
    if duration <= 0:
        return []

    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
    if not tokens:
        return []

    chunks: list[str] = []
    for i in range(0, len(tokens), max(words_per_chunk, 1)):
        chunk_tokens = tokens[i : i + max(words_per_chunk, 1)]
        chunks.append(" ".join(chunk_tokens))

    num_chunks = len(chunks)
    if num_chunks == 0:
        return []

    chunk_duration = max(min_chunk_sec, duration / num_chunks)
    print(
        f"Caption chunks={num_chunks} chunk_duration={chunk_duration:.2f}s start={start:.2f}s"
    )

    caption_y = y_pos if y_pos is not None else int(H * 0.68)
    clips: list[object] = []
    for i, chunk in enumerate(chunks):
        caption = make_safe_caption_clip(
            chunk,
            start + i * chunk_duration,
            chunk_duration + overlap_sec,
            W,
            H,
            y=caption_y,
            speaker=speaker,
            font=font or "Arial",
            font_size=font_size,
            stroke_width=stroke_width,
        )
        caption = with_bounce_in(
            caption,
            bounce_from=0.95,
            bounce_to=1.0,
            bounce_sec=0.08,
        )
        clips.append(caption)
    return clips


def make_safe_caption_clip(
    text: str,
    start: float,
    duration: float,
    W: int,
    H: int,
    *,
    y: int | None = None,
    speaker: str | None = None,
    font: str = "Arial",
    font_size: int = 80,
    color: str = "white",
    stroke_color: str = "black",
    stroke_width: int = 5,
    max_width_ratio: float = 0.94,
    margin_px: int = 24,
):
    from moviepy import ImageClip
    max_w = int(W * max_width_ratio)
    max_h = H - (margin_px * 2)
    pad = max(margin_px, stroke_width * 2)
    tag_gap = 10
    tag_pad_x = 14
    tag_pad_y = 8
    tag_font_size = max(16, int(font_size * 0.58))

    def _resolve_font(size: int) -> ImageFont.ImageFont:
        if font and os.path.exists(font):
            try:
                return ImageFont.truetype(font, size)
            except Exception:
                pass
        for path in (
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ):
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def _wrap_text(draw: ImageDraw.ImageDraw, text_value: str, font_obj: ImageFont.ImageFont) -> str:
        words = text_value.split()
        if not words:
            return ""
        lines: list[str] = []
        current: list[str] = []
        max_line_w = max_w - (pad * 2)
        for word in words:
            candidate = " ".join(current + [word])
            bbox = draw.textbbox((0, 0), candidate, font=font_obj, stroke_width=stroke_width)
            if bbox[2] - bbox[0] <= max_line_w or not current:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
        return "\n".join(lines)

    def _speaker_tag_style(name: str | None) -> tuple[tuple[int, int, int, int], str]:
        if not name:
            return (128, 128, 128, 255), ""
        label = name.upper()
        if label == "JOHN":
            return (0, 229, 255, 255), label
        if label == "CARTOON_DAD":
            return (255, 212, 0, 255), label
        return (128, 128, 128, 255), label

    def _render_caption(size: int) -> Image.Image:
        font_obj = _resolve_font(size)
        tag_font_obj = _resolve_font(tag_font_size)
        dummy = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)
        wrapped = _wrap_text(draw, text, font_obj)
        if not wrapped:
            wrapped = text
        bbox = draw.multiline_textbbox(
            (0, 0),
            wrapped,
            font=font_obj,
            stroke_width=stroke_width,
            spacing=int(size * 0.15),
        )
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        img_w = int(text_w + (pad * 2))
        img_h = int(text_h + (pad * 2))
        tag_color, tag_label = _speaker_tag_style(speaker)
        tag_h = 0
        tag_w = 0
        if tag_label:
            tag_bbox = draw.textbbox((0, 0), tag_label, font=tag_font_obj)
            tag_w = tag_bbox[2] - tag_bbox[0] + (tag_pad_x * 2)
            tag_h = tag_bbox[3] - tag_bbox[1] + (tag_pad_y * 2)
            img_w = int(max(img_w, tag_w))
            img_h = int(img_h + tag_h + tag_gap)
        canvas = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        if tag_label:
            tag_x = 0
            tag_y = 0
            draw.rectangle(
                [tag_x, tag_y, tag_x + tag_w, tag_y + tag_h],
                fill=tag_color,
            )
            draw.text(
                (tag_x + tag_pad_x, tag_y + tag_pad_y),
                tag_label,
                font=tag_font_obj,
                fill=(0, 0, 0, 255),
            )
        box_y = tag_h + (tag_gap if tag_label else 0)
        draw.rectangle(
            [0, box_y, img_w, box_y + (text_h + (pad * 2))],
            fill=(0, 0, 0, 115),
        )
        draw.multiline_text(
            (pad, pad + box_y),
            wrapped,
            font=font_obj,
            fill=(255, 255, 255, 255),
            stroke_fill=(0, 0, 0, 255),
            stroke_width=6,
            spacing=int(size * 0.15),
        )
        return canvas

    img = _render_caption(font_size)
    while (img.height > max_h or img.width > max_w) and font_size > 16:
        font_size = max(16, int(font_size * 0.9))
        img = _render_caption(font_size)
    if img.height > max_h or img.width > max_w:
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, resample=Image.LANCZOS)

    desired_y = int(H * 0.62) if y is None else y
    min_y = margin_px
    max_y = max(margin_px, H - img.height - margin_px)
    safe_y = max(min_y, min(desired_y, max_y))
    if safe_y != desired_y:
        print(f"Caption y adjusted {desired_y} -> {safe_y}")

    clip = ImageClip(np.array(img))
    clip = clip.with_start(start).with_duration(duration)
    clip = clip.with_position(("center", safe_y))
    return clip


def with_bounce_in(
    clip,
    *,
    bounce_from: float = 0.95,
    bounce_to: float = 1.0,
    bounce_sec: float = 0.08,
):
    duration = clip.duration or 0
    if duration <= 0:
        return clip
    bounce_sec = min(bounce_sec, duration)
    if bounce_sec <= 0:
        return clip

    def scale_at(t: float) -> float:
        if t <= 0:
            return bounce_from
        if t <= bounce_sec:
            return bounce_from + (bounce_to - bounce_from) * (t / bounce_sec)
        return bounce_to

    return clip.resized(lambda t: scale_at(t))


def with_character_transition(
    img_clip,
    *,
    final_pos: tuple[int, int],
    side: str,
    transition: str = "slide",
    trans_sec: float = 0.12,
    slide_px: int = 120,
):
    duration = img_clip.duration or 0
    if duration <= 0:
        return img_clip.with_position(final_pos)
    trans_sec = min(trans_sec, duration)
    if trans_sec <= 0:
        return img_clip.with_position(final_pos)

    final_x, final_y = final_pos
    if transition == "fade":
        def opacity_at(t: float) -> float:
            if t <= 0:
                return 0.0
            if t <= trans_sec:
                return t / trans_sec
            return 1.0

        return img_clip.with_opacity(lambda t: opacity_at(t)).with_position(final_pos)

    offset = -slide_px if side == "left" else slide_px

    def pos_at(t: float) -> tuple[float, float]:
        if t <= 0:
            return (final_x + offset, final_y)
        if t <= trans_sec:
            x = final_x + offset * (1 - (t / trans_sec))
            return (x, final_y)
        return (final_x, final_y)

    return img_clip.with_position(lambda t: pos_at(t))


def render_shorts_video(
    timed_dialogue: list[dict[str, object]],
    audio_path: str = "temp/duo_audio.mp3",
    output_path: str = "temp/output_short.mp4",
    bg_video_path: str = "temp/brainRotVideos/default.mp4",
) -> str:
    from moviepy import (
        AudioFileClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        vfx,
    )

    john_path = os.path.join("temp", "john_character_cutout.png")
    dad_path = os.path.join("temp", "cartoon_dad_transparent.png")

    width, height = 1080, 1920
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration

    background = VideoFileClip(bg_video_path)
    if background.duration <= audio_duration:
        background = background.with_effects([vfx.Loop(duration=audio_duration)])
        start_t = 0.0
    else:
        start_t = random.uniform(0, background.duration - audio_duration)
    background = background.subclipped(start_t, start_t + audio_duration)
    scale = max(width / background.w, height / background.h)
    background = background.resized(scale)
    background = background.cropped(
        x_center=background.w / 2,
        y_center=background.h / 2,
        width=width,
        height=height,
    )

    overlays: list[object] = []
    for entry in timed_dialogue:
        speaker = str(entry["speaker"]).upper()
        text = str(entry["text"])
        start = float(entry["start"])
        line_duration = float(entry["duration"])
        img_path = john_path if speaker == "JOHN" else dad_path

        img_clip = ImageClip(img_path).resized(width=int(width * 0.5))
        img_clip = img_clip.with_start(start).with_duration(line_duration)
        img_y = height - img_clip.h - int(height * 0.05)
        if speaker == "CARTOON_DAD":
            img_x = int(width * 0.25) - int(img_clip.w / 2)
            side = "left"
        else:
            img_x = int(width * 0.75) - int(img_clip.w / 2)
            side = "right"
        img_clip = with_character_transition(
            img_clip,
            final_pos=(img_x, img_y),
            side=side,
            transition="slide",
            trans_sec=0.12,
        )

        text_y = max(int(height * 0.05), img_y - int(height * 0.40))
        caption_clips = five_word_caption_clips(
            text,
            start,
            line_duration,
            width,
            height,
            speaker=speaker,
            words_per_chunk=5,
            font_size=72,
            font="Menlo",
            y_pos=text_y,
            stroke_width=6,
        )

        overlays.append(img_clip)
        overlays.extend(caption_clips)

    # Future features: active speaker glow, karaoke word timing, toy example cards.
    composite = CompositeVideoClip([background] + overlays, size=(width, height))
    composite = composite.with_duration(audio_duration).with_audio(audio_clip)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    composite.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=30,
    )
    return output_path
