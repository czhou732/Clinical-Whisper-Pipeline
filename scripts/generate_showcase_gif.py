#!/usr/bin/env python3
"""
Generate a LinkedIn-optimized Clinical Whisper showcase GIF.

Spec:
- 1080x1080 square
- 10 fps
- Hard cuts between scenes
- Clinical serif/sans aesthetic
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1080
HEIGHT = 1080
FPS = 10
TARGET_MAX_BYTES = 8 * 1024 * 1024

TITLE_FRAMES = 15   # 1.5s
SCENE1_FRAMES = 35  # 3.5s
SCENE2_FRAMES = 35  # 3.5s
SCENE3_FRAMES = 130 # 13.0s
SCENE4_FRAMES = 35  # 3.5s
TOTAL_FRAMES = TITLE_FRAMES + SCENE1_FRAMES + SCENE2_FRAMES + SCENE3_FRAMES + SCENE4_FRAMES

ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = ROOT / "assets" / "showcase_frames"
OUT_GIF = ROOT / "assets" / "clinical-whisper-showcase-30s.gif"
OUT_GIF_OPT = ROOT / "assets" / "clinical-whisper-showcase-30s-linkedin.gif"

FONT_SERIF = "/System/Library/Fonts/Supplemental/Georgia.ttf"
FONT_SANS = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_SANS_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"

OFFWHITE = (248, 249, 250, 255)
WHITE = (255, 255, 255, 255)
NAVY = (0, 51, 102, 255)
DARK = (51, 51, 51, 255)
GRAY = (96, 96, 96, 255)
LIGHT_GRAY = (206, 212, 218, 255)
TEXT = (0, 0, 0, 255)

MARGIN = 1
TOP_BAR_H = 0
BORDER_W = 2
OVERLAY_H = 42
INNER_PAD = 40
CONTAINER_X0 = MARGIN
CONTAINER_Y0 = MARGIN
CONTAINER_X1 = WIDTH - MARGIN
CONTAINER_Y1 = HEIGHT - MARGIN
CONTENT_X0 = CONTAINER_X0 + INNER_PAD
CONTENT_X1 = CONTAINER_X1 - INNER_PAD
CONTENT_Y0 = CONTAINER_Y0 + INNER_PAD
CONTENT_Y1 = CONTAINER_Y1 - INNER_PAD
OVERLAY_Y = CONTAINER_Y1 - OVERLAY_H - 12


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def tsize(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, fnt: ImageFont.FreeTypeFont, fill: tuple[int, int, int, int]) -> None:
    w, _ = tsize(draw, text, fnt)
    x = (WIDTH - w) // 2
    draw.text((x, y), text, font=fnt, fill=fill)


def draw_centered_in_range(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    x0: int,
    x1: int,
) -> None:
    w, _ = tsize(draw, text, fnt)
    x = x0 + max(0, (x1 - x0 - w) // 2)
    draw.text((x, y), text, font=fnt, fill=fill)


def fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    max_size: int,
    min_size: int,
    max_width: int,
) -> ImageFont.FreeTypeFont:
    for size in range(max_size, min_size - 1, -1):
        fnt = font(font_path, size)
        w, _ = tsize(draw, text, fnt)
        if w <= max_width:
            return fnt
    return font(font_path, min_size)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words = text.split(" ")
    out: list[str] = []
    cur = ""
    for w in words:
        cand = w if not cur else f"{cur} {w}"
        cw, _ = tsize(draw, cand, fnt)
        if cw <= max_w:
            cur = cand
        else:
            if cur:
                out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out or [text]


def base_frame() -> Image.Image:
    img = Image.new("RGBA", (WIDTH, HEIGHT), OFFWHITE)
    draw = ImageDraw.Draw(img)
    draw.rectangle((CONTAINER_X0, CONTAINER_Y0, CONTAINER_X1, CONTAINER_Y1), fill=WHITE, outline=NAVY, width=BORDER_W)
    return img


def step_overlay(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.rectangle((CONTENT_X0, OVERLAY_Y, CONTENT_X1, OVERLAY_Y + OVERLAY_H), fill=(255, 255, 255, 230), outline=None)
    label_font = font(FONT_SANS_BOLD, 16)
    draw.text((CONTENT_X0 + 14, OVERLAY_Y + 10), text.upper(), font=label_font, fill=TEXT)


def draw_title_scene(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)

    # Title block: no box; only a horizontal rule below title.
    title_font = fit_font_size(
        draw,
        "CLINICAL WHISPER PIPELINE",
        FONT_SERIF,
        max_size=74,
        min_size=48,
        max_width=(CONTENT_X1 - CONTENT_X0) - 80,  # at least 40px padding each side
    )
    subtitle_font = font(FONT_SANS, 38)
    bullet_font = font(FONT_SANS, 34)
    url_font = font(FONT_SANS, 31)
    footer_font = font(FONT_SERIF, 34)

    # Build a vertically centered content stack to remove top-pinned dead space.
    title_text = "CLINICAL WHISPER PIPELINE"
    subtitle_1 = "Offline-First ASR . Async Queue ."
    subtitle_2 = "Scalable Clinical Inference"
    feature_lines = [
        "Ingestion API returns job_id in milliseconds",
        "Queue-backed worker orchestration",
        "Structured JSON for Streamlit reporting",
    ]
    open_source = "Open Source · Hugging Face Demo"
    url_1 = "huggingface.co/spaces/ChengdongPeter/"
    url_2 = "Clinical-Whisper"
    footer = "Neuro-Systems Group"

    _, title_h = tsize(draw, title_text, title_font)
    _, sub_h = tsize(draw, subtitle_1, subtitle_font)
    _, bullet_h = tsize(draw, feature_lines[0], bullet_font)
    _, open_h = tsize(draw, open_source, font(FONT_SANS, 30))
    _, url_h = tsize(draw, url_1, url_font)
    _, footer_h = tsize(draw, footer, footer_font)

    total_h = (
        title_h + 20 + 1 + 24 +  # title + gap + rule + gap
        sub_h + 10 + sub_h + 48 +  # subtitles
        bullet_h + 16 + bullet_h + 16 + bullet_h + 28 +  # features
        open_h + 30 + url_h + 8 + url_h + 44 + footer_h
    )

    block_top = CONTENT_Y0 + max(0, (CONTENT_Y1 - CONTENT_Y0 - total_h) // 2)
    y = block_top

    draw_centered_in_range(draw, title_text, y, title_font, TEXT, CONTENT_X0 + 40, CONTENT_X1 - 40)
    y += title_h + 20
    draw.line((CONTENT_X0 + 40, y, CONTENT_X1 - 40, y), fill=LIGHT_GRAY, width=1)
    y += 24

    draw_centered_in_range(draw, subtitle_1, y, subtitle_font, TEXT, CONTENT_X0, CONTENT_X1)
    y += sub_h + 10
    draw_centered_in_range(draw, subtitle_2, y, subtitle_font, TEXT, CONTENT_X0, CONTENT_X1)
    y += sub_h + 48

    for idx, line in enumerate(feature_lines):
        draw_centered_in_range(draw, line, y, bullet_font, TEXT, CONTENT_X0, CONTENT_X1)
        y += bullet_h + (16 if idx < len(feature_lines) - 1 else 28)

    draw_centered_in_range(draw, open_source, y, font(FONT_SANS, 30), TEXT, CONTENT_X0, CONTENT_X1)
    y += open_h + 30
    draw_centered_in_range(draw, url_1, y, url_font, TEXT, CONTENT_X0, CONTENT_X1)
    y += url_h + 8
    draw_centered_in_range(draw, url_2, y, url_font, TEXT, CONTENT_X0, CONTENT_X1)
    y += url_h + 44
    draw_centered_in_range(draw, footer, y, footer_font, TEXT, CONTENT_X0, CONTENT_X1)


def draw_upload_scene(img: Image.Image, local_idx: int) -> None:
    draw = ImageDraw.Draw(img)

    panel_x0 = CONTENT_X0
    panel_y0 = CONTENT_Y0 + 10
    panel_x1 = CONTENT_X1

    draw.text((panel_x0, panel_y0), "INGESTION API (WAV/MP3/M4A)", font=font(FONT_SERIF, 40), fill=TEXT)
    draw.line((panel_x0, panel_y0 + 56, panel_x1, panel_y0 + 56), fill=LIGHT_GRAY, width=1)

    drop_x0 = panel_x0
    drop_y0 = panel_y0 + 92
    drop_x1 = panel_x1
    drop_y1 = drop_y0 + 390

    # Single visible dropzone border only.
    draw.rectangle((drop_x0, drop_y0, drop_x1, drop_y1), fill=WHITE, outline=LIGHT_GRAY, width=1)
    # dashed border
    for x in range(drop_x0 + 14, drop_x1 - 14, 18):
        draw.line((x, drop_y0 + 14, min(x + 10, drop_x1 - 14), drop_y0 + 14), fill=LIGHT_GRAY, width=1)
        draw.line((x, drop_y1 - 14, min(x + 10, drop_x1 - 14), drop_y1 - 14), fill=LIGHT_GRAY, width=1)
    for y in range(drop_y0 + 14, drop_y1 - 14, 18):
        draw.line((drop_x0 + 14, y, drop_x0 + 14, min(y + 10, drop_y1 - 14)), fill=LIGHT_GRAY, width=1)
        draw.line((drop_x1 - 14, y, drop_x1 - 14, min(y + 10, drop_y1 - 14)), fill=LIGHT_GRAY, width=1)

    drag_text_y = drop_y0 + 68
    limit_text_y = drop_y0 + 144
    draw.text((drop_x0 + 42, drag_text_y), "Drag and drop file for ingestion", font=font(FONT_SANS_BOLD, 52), fill=TEXT)
    draw.text((drop_x0 + 42, limit_text_y), "Immediate API response: job_id + queued status", font=font(FONT_SANS, 33), fill=TEXT)

    btn_w = 360
    btn_h = 84
    btn_x0 = drop_x0 + 42
    # Keep button inside dashed dropzone: below limit text, above bottom edge.
    btn_y0 = limit_text_y + 64
    # Outlined by default; solid only on final hover beat.
    hover = local_idx >= (SCENE1_FRAMES - 5)
    btn_fill = NAVY if hover else WHITE
    btn_text = TEXT
    draw.rectangle((btn_x0, btn_y0, btn_x0 + btn_w, btn_y0 + btn_h), fill=btn_fill, outline=NAVY, width=2)
    bw, bh = tsize(draw, "SUBMIT JOB", font(FONT_SANS_BOLD, 53))
    draw.text((btn_x0 + (btn_w - bw) // 2, btn_y0 + (btn_h - bh) // 2 - 4), "SUBMIT JOB", font=font(FONT_SANS_BOLD, 53), fill=btn_text)

    file_y0 = drop_y1 + 36
    # Uploaded file indicator: single solid light-gray border.
    draw.rectangle((drop_x0, file_y0, drop_x1, file_y0 + 140), fill=WHITE, outline=LIGHT_GRAY, width=1)
    draw.text((drop_x0 + 24, file_y0 + 30), "test_audio.m4a -> job_8f31a2", font=font(FONT_SANS_BOLD, 42), fill=TEXT)
    dots = "." * (local_idx % 4)
    draw.text((drop_x0 + 24, file_y0 + 88), f"Queue status: queued {dots}", font=font(FONT_SANS, 42), fill=TEXT)

    step_overlay(draw, "STEP 1: INGESTION LAYER RETURNS JOB_ID")


def draw_processing_scene(img: Image.Image, local_idx: int) -> None:
    draw = ImageDraw.Draw(img)

    panel_x0 = CONTENT_X0
    panel_y0 = CONTENT_Y0 + 14
    panel_x1 = CONTENT_X1

    draw.text((panel_x0, panel_y0), "ASYNC PROCESSING PIPELINE", font=font(FONT_SERIF, 42), fill=TEXT)
    draw.line((panel_x0, panel_y0 + 56, panel_x1, panel_y0 + 56), fill=LIGHT_GRAY, width=1)

    labels = [
        "Secure storage + queue enqueue",
        "Broker dispatch to background workers",
        "Whisper + Pyannote + sentiment inference",
        "Write Output/[job_id]_analysis.json",
    ]

    p = (local_idx + 1) / SCENE2_FRAMES
    y = panel_y0 + 96
    bar_w = panel_x1 - panel_x0
    for i, label in enumerate(labels):
        stage_start = i * 0.22
        stage_end = min(1.0, stage_start + 0.34)
        stage_p = 0.0
        if p > stage_start:
            stage_p = min(1.0, (p - stage_start) / (stage_end - stage_start))

        y0 = y + i * 150
        y1 = y0 + 92
        draw.rectangle((panel_x0, y0, panel_x0 + bar_w, y1), fill=(240, 243, 248, 255), outline=None)

        fill_w = int((bar_w - 4) * stage_p)
        if fill_w > 0:
            draw.rectangle((panel_x0 + 2, y0 + 2, panel_x0 + 2 + fill_w, y1 - 2), fill=(0, 51, 102, 70), outline=None)

        draw.text((panel_x0 + 18, y0 + 22), label.upper(), font=font(FONT_SANS_BOLD, 25), fill=TEXT)
        pct = int(stage_p * 100)
        draw.text((panel_x0 + bar_w - 120, y0 + 18), f"{pct}%", font=font(FONT_SANS_BOLD, 30), fill=TEXT)

    step_overlay(draw, "STEP 2: QUEUE + SCALABLE INFERENCE WORKERS")


def build_report_canvas() -> Image.Image:
    report_w = WIDTH - 2 * 58
    report_h = 1660
    img = Image.new("RGBA", (report_w, report_h), WHITE)
    draw = ImageDraw.Draw(img)

    draw.text((26, 20), "Clinical Intelligence Report", font=font(FONT_SERIF, 44), fill=TEXT)
    draw.text((26, 72), "Confidential // Neuro-Systems Group", font=font(FONT_SANS, 24), fill=TEXT)

    y = 122
    draw.text((26, y), "SESSION VITALS", font=font(FONT_SERIF, 32), fill=TEXT)
    y += 46
    for name, value in [
        ("Overall sentiment", "6.8 / 10"),
        ("Distress segments", "3"),
        ("Speaker turns", "48"),
        ("Total words", "2,148"),
    ]:
        draw.text((34, y), name.upper(), font=font(FONT_SANS_BOLD, 22), fill=TEXT)
        draw.text((410, y - 2), value, font=font(FONT_SANS_BOLD, 30), fill=TEXT)
        y += 42

    y += 18
    draw.text((26, y), "EMOTIONAL ARC (SEGMENT SCORES)", font=font(FONT_SERIF, 32), fill=TEXT)
    y += 54
    chart_x0 = 34
    chart_y0 = y
    chart_x1 = report_w - 34
    chart_y1 = y + 220
    draw.line((chart_x0 + 2, chart_y0 + 158, chart_x1 - 2, chart_y0 + 158), fill=GRAY, width=1)
    pts = [
        (chart_x0 + 20, chart_y0 + 140), (chart_x0 + 96, chart_y0 + 126), (chart_x0 + 172, chart_y0 + 170),
        (chart_x0 + 248, chart_y0 + 152), (chart_x0 + 324, chart_y0 + 164), (chart_x0 + 400, chart_y0 + 120),
        (chart_x0 + 476, chart_y0 + 110), (chart_x0 + 552, chart_y0 + 150), (chart_x0 + 628, chart_y0 + 134),
        (chart_x0 + 704, chart_y0 + 126), (chart_x0 + 780, chart_y0 + 96), (chart_x0 + 856, chart_y0 + 102),
        (chart_x0 + 932, chart_y0 + 88),
    ]
    draw.line(pts, fill=NAVY, width=3)
    for p in pts:
        draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=NAVY)

    y = chart_y1 + 30
    draw.text((26, y), "TRANSCRIPT WITH SPEAKER LABELS", font=font(FONT_SERIF, 32), fill=TEXT)
    y += 46

    transcript_lines = [
        "[00:12] CLINICIAN  | How have the nights been since our last session?",
        "[00:18] PATIENT    | I wake around 3am and feel dread before the day starts.",
        "[00:34] CLINICIAN  | We can track that pattern and adjust sleep hygiene.",
        "[00:52] PATIENT    | Walking outside helps a little, but everything feels muted.",
        "[01:05] CLINICIAN  | That is a signal we can build on with repeatable routines.",
        "[01:25] PATIENT    | That feels manageable. I can commit for two weeks.",
        "[01:42] CLINICIAN  | We will review outcomes and refine interventions.",
        "[02:04] PATIENT    | Sleep is still unstable, but anxiety intensity is lower.",
        "[02:28] CLINICIAN  | Continue breathing protocol and evening screen restriction.",
        "[02:44] PATIENT    | Understood. I can follow that plan this week.",
        "[03:03] CLINICIAN  | Good. We will monitor and compare segment-level affect.",
    ]
    mono = font(FONT_MONO, 22)
    for line in transcript_lines:
        draw.text((34, y), line, font=mono, fill=TEXT)
        y += 42

    y += 18
    draw.text((26, y), "SEGMENT SENTIMENT SCORES", font=font(FONT_SERIF, 32), fill=TEXT)
    y += 48
    table_head = "Segment   Speaker      Score/10   Affect Flag"
    draw.text((34, y), table_head, font=mono, fill=TEXT)
    y += 34
    for row in [
        "01        Patient      2.1        High distress",
        "02        Clinician    6.5        Stable",
        "03        Patient      2.5        High distress",
        "04        Clinician    7.2        Stable",
        "05        Patient      4.8        Mild concern",
        "06        Clinician    8.0        Supportive",
        "07        Patient      5.5        Improving",
        "08        Clinician    7.8        Supportive",
    ]:
        draw.text((34, y), row, font=mono, fill=TEXT)
        y += 34

    y += 18
    draw.text((26, y), "AFFECT-FLAGGED LANGUAGE PATTERNS", font=font(FONT_SERIF, 32), fill=TEXT)
    y += 48
    flags = [
        "Pattern 1: recurrent dread statements in early morning context",
        "Pattern 2: anhedonia markers around previously enjoyable activities",
        "Pattern 3: positive response to behavioral activation prompts",
        "Pattern 4: improvement trend after guided breathing intervention",
    ]
    for fline in flags:
        wrapped = wrap_text(draw, fline, font(FONT_SANS, 28), report_w - 80)
        for ln in wrapped:
            draw.text((34, y), "- " + ln, font=font(FONT_SANS, 28), fill=TEXT)
            y += 36

    return img


def draw_report_scene(img: Image.Image, local_idx: int, report_canvas: Image.Image) -> None:
    draw = ImageDraw.Draw(img)

    viewport_x0 = CONTENT_X0
    viewport_y0 = CONTENT_Y0 + 12
    viewport_w = CONTENT_X1 - CONTENT_X0
    viewport_h = 860

    # 60 px/s scroll speed at 10 fps => 6 px per frame.
    scroll_px = local_idx * 6
    max_scroll = max(0, report_canvas.height - viewport_h)
    scroll_px = min(scroll_px, max_scroll)

    crop = report_canvas.crop((0, scroll_px, viewport_w, scroll_px + viewport_h))
    img.paste(crop, (viewport_x0, viewport_y0), crop)

    step_overlay(draw, "STEP 3: DATA SINK + REPORTING LAYER")


def draw_summary_scene(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)

    x0 = CONTENT_X0
    x1 = CONTENT_X1
    draw.text((x0, 200), "CLINICAL WHISPER SUMMARY", font=font(FONT_SERIF, 56), fill=TEXT)
    draw.line((x0, 270, x1, 270), fill=LIGHT_GRAY, width=1)

    y = 330
    for line in [
        "- FastAPI ingestion returns job_id in milliseconds",
        "- Queue + worker pool scales inference on CPU/GPU",
        "- Streamlit reads Output/[job_id]_analysis.json",
    ]:
        draw.text((x0 + 20, y), line, font=font(FONT_SANS, 40), fill=TEXT)
        y += 74

    y += 14
    draw.text((x0 + 20, y), "Try it:", font=font(FONT_SANS_BOLD, 40), fill=TEXT)
    draw.text((x0 + 140, y), "huggingface.co/spaces/ChengdongPeter/Clinical-Whisper", font=font(FONT_SANS, 29), fill=TEXT)

    step_overlay(draw, "FINAL SUMMARY")


def draw_frame(index: int, report_canvas: Image.Image) -> Image.Image:
    img = base_frame()

    if index < TITLE_FRAMES:
        draw_title_scene(img)
        return img

    i = index - TITLE_FRAMES
    if i < SCENE1_FRAMES:
        draw_upload_scene(img, i)
        return img

    i -= SCENE1_FRAMES
    if i < SCENE2_FRAMES:
        draw_processing_scene(img, i)
        return img

    i -= SCENE2_FRAMES
    if i < SCENE3_FRAMES:
        draw_report_scene(img, i, report_canvas)
        return img

    draw_summary_scene(img)
    return img


def render_frames() -> None:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAME_DIR.glob("frame_*.png"):
        old.unlink()

    report_canvas = build_report_canvas()

    for i in range(TOTAL_FRAMES):
        frame = draw_frame(i, report_canvas)
        frame.convert("RGB").save(FRAME_DIR / f"frame_{i:04d}.png", format="PNG", optimize=True)
        if i % 25 == 0 or i == TOTAL_FRAMES - 1:
            print(f"Rendered frame {i + 1}/{TOTAL_FRAMES}")


def encode_gif() -> None:
    def encode_variant(raw_out: Path, out_fps: int, duration_sec: int | None = None) -> None:
        t_arg = f"-t {duration_sec} " if duration_sec is not None else ""
        os.system(
            "ffmpeg -y "
            f"-framerate {FPS} "
            f"-i '{FRAME_DIR}/frame_%04d.png' "
            f"{t_arg}"
            f"-vf \"fps={out_fps},scale=1080:1080:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=128:stats_mode=single[p];"
            "[s1][p]paletteuse=dither=sierra2_4a\" "
            f"'{raw_out}'"
        )

    def optimize_variant(src: Path, dst: Path) -> None:
        if os.system("command -v gifsicle >/dev/null 2>&1") == 0:
            os.system(f"gifsicle -O3 --lossy=80 --colors 128 '{src}' -o '{dst}'")
        else:
            # Fallback copy if gifsicle is unavailable.
            os.system(f"cp '{src}' '{dst}'")

    # Primary encode at 10 fps.
    encode_variant(OUT_GIF, out_fps=10)
    optimize_variant(OUT_GIF, OUT_GIF_OPT)

    # If still too large, use the requested fallback to 8 fps.
    if OUT_GIF_OPT.exists() and OUT_GIF_OPT.stat().st_size > TARGET_MAX_BYTES:
        print("LinkedIn GIF is above 8MB; applying 8fps fallback...")
        fallback_raw = ROOT / "assets" / "clinical-whisper-showcase-30s-8fps.gif"
        encode_variant(fallback_raw, out_fps=8)
        optimize_variant(fallback_raw, OUT_GIF_OPT)
        if fallback_raw.exists():
            fallback_raw.unlink()

    # If still too large, apply the requested duration trim fallback to 20 seconds.
    if OUT_GIF_OPT.exists() and OUT_GIF_OPT.stat().st_size > TARGET_MAX_BYTES:
        print("Still above 8MB; applying 20-second trim fallback...")
        trim_raw = ROOT / "assets" / "clinical-whisper-showcase-20s-8fps.gif"
        encode_variant(trim_raw, out_fps=8, duration_sec=20)
        optimize_variant(trim_raw, OUT_GIF_OPT)
        if trim_raw.exists():
            trim_raw.unlink()


def main() -> None:
    print("Generating frames...")
    render_frames()
    print("Encoding GIF...")
    encode_gif()
    print(f"Done: {OUT_GIF}")
    if OUT_GIF_OPT.exists():
        print(f"Done: {OUT_GIF_OPT}")


if __name__ == "__main__":
    main()
