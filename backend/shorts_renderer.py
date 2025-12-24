"""Video renderer for Duo Mode Shorts."""

from __future__ import annotations

import os


def render_shorts_video(
    timed_dialogue: list[dict[str, object]],
    audio_path: str = "temp/duo_audio.mp3",
    output_path: str = "temp/output_short.mp4",
) -> str:
    from moviepy import (
        AudioFileClip,
        CompositeVideoClip,
        ImageClip,
        TextClip,
        VideoFileClip,
        vfx,
    )

    bg_path = os.path.join("temp", "Subway Surfers brainrot.mp4")
    john_path = os.path.join("temp", "john_character_cutout.png")
    dad_path = os.path.join("temp", "cartoon_dad_transparent.png")

    width, height = 1080, 1920
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration

    background = VideoFileClip(bg_path)
    if background.duration < audio_duration:
        background = background.with_effects([vfx.Loop(duration=audio_duration)])
    background = background.subclipped(0, audio_duration)
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

        img_clip = ImageClip(img_path).resized(width=int(width * 0.4))
        img_clip = img_clip.with_start(start).with_duration(line_duration)
        img_y = height - img_clip.h - int(height * 0.05)
        img_clip = img_clip.with_position(("center", img_y))

        try:
            text_clip = TextClip(
                text=text,
                method="caption",
                size=(int(width * 0.9), None),
                font_size=42,
                color="white",
                stroke_color="black",
                stroke_width=3,
                font="Arial-Bold",
            )
        except Exception:
            text_clip = TextClip(
                text=text,
                method="caption",
                size=(int(width * 0.9), None),
                font_size=42,
                color="white",
                stroke_color="black",
                stroke_width=3,
            )
        text_clip = text_clip.with_start(start).with_duration(line_duration)
        text_y = max(int(height * 0.05), img_y - text_clip.h - int(height * 0.03))
        text_clip = text_clip.with_position(("center", text_y))

        overlays.extend([img_clip, text_clip])

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
