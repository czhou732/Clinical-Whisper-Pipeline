#!/usr/bin/env python3
"""
Generate a 5-slide "Case Study" PDF in an Academic Brutalism style.

Technical constraints:
- Matplotlib-based rendering
- Uses matplotlib.patches for browser/terminal framing
- Exports high DPI PDF as Clinical_Whisper_Blueprint.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


ASSETS = ROOT / "assets"
HERO_FRAME = ASSETS / "showcase_frames" / "frame_0140.png"
OUT_PDF = ROOT / "Clinical_Whisper_Blueprint.pdf"

BG = "#F5F5F7"
BLUE = "#002244"
RED = "#FF3B30"
INK = "#121417"
MUTED = "#4E5968"
GRID = "#9FB2C9"

FIG_W = 16
FIG_H = 9
DPI = 300


def new_canvas() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    draw_texture(ax)
    return fig, ax


def draw_texture(ax: plt.Axes) -> None:
    for x in np.arange(0.0, 1.001, 0.05):
        ax.plot([x, x], [0, 1], color=GRID, alpha=0.08, lw=0.5, zorder=0)
    for y in np.arange(0.0, 1.001, 0.05):
        ax.plot([0, 1], [y, y], color=GRID, alpha=0.08, lw=0.5, zorder=0)

    corners = [(0.035, 0.965), (0.965, 0.965), (0.035, 0.035), (0.965, 0.035)]
    for cx, cy in corners:
        ax.plot([cx - 0.02, cx + 0.02], [cy, cy], color=BLUE, alpha=0.22, lw=1.2, zorder=0)
        ax.plot([cx, cx], [cy - 0.02, cy + 0.02], color=BLUE, alpha=0.22, lw=1.2, zorder=0)
        ax.add_patch(Circle((cx, cy), 0.004, fc=BLUE, ec="none", alpha=0.24, zorder=0))


def slide_1_cover(ax: plt.Axes) -> None:
    # Faint wireframe sphere in the background.
    cx, cy, r = 0.76, 0.56, 0.44
    theta = np.linspace(0, 2 * np.pi, 500)
    for lat in np.linspace(-1.2, 1.2, 13):
        scale = np.cos(lat * np.pi / 3.4)
        x = cx + r * np.cos(theta) * max(0.15, scale) * 0.95
        y = cy + (r * np.sin(theta) * 0.20) + lat * 0.05
        ax.plot(x, y, color=BLUE, alpha=0.08, lw=0.9, zorder=1)
    for lon in np.linspace(-1.2, 1.2, 17):
        t = np.linspace(-1.0, 1.0, 350)
        x = cx + (r * 0.95 * np.sin(lon)) * np.cos(t * np.pi / 2)
        y = cy + (r * 0.45 * np.sin(t * np.pi / 2))
        ax.plot(x, y, color=BLUE, alpha=0.06, lw=0.9, zorder=1)

    ax.text(
        0.97,
        0.955,
        "CONFIDENTIAL // NEURO-SYSTEMS // v1.0",
        ha="right",
        va="top",
        color=MUTED,
        fontsize=12,
        fontfamily="monospace",
        fontweight="bold",
        zorder=3,
    )

    ax.text(
        0.07,
        0.18,
        "THE DATA\nBOTTLENECK",
        ha="left",
        va="bottom",
        color=BLUE,
        fontsize=86,
        fontfamily="DejaVu Sans",
        fontweight="heavy",
        linespacing=0.88,
        zorder=3,
    )
    ax.text(
        0.07,
        0.075,
        "Why offline-first infrastructure matters for reproducible psychiatry",
        ha="left",
        va="bottom",
        color=INK,
        fontsize=20,
        fontfamily="DejaVu Sans",
        fontweight="medium",
        zorder=3,
    )
    ax.plot([0.07, 0.31], [0.068, 0.068], color=RED, lw=2.4, zorder=3)


def slide_2_stack(ax: plt.Axes) -> None:
    ax.text(
        0.07,
        0.90,
        "SYSTEM STACK",
        ha="left",
        va="top",
        color=BLUE,
        fontsize=54,
        fontfamily="DejaVu Sans",
        fontweight="heavy",
    )
    ax.text(
        0.07,
        0.84,
        "Zero data egress architecture for clinical audio intelligence",
        ha="left",
        va="top",
        color=MUTED,
        fontsize=19,
        fontfamily="DejaVu Sans",
        fontweight="medium",
    )

    labels = [
        "INPUT: Raw Audio",
        "ENGINE: Whisper (Local)",
        "LOGIC: Pyannote (Diarization)",
        "OUTPUT: Clinical Report",
    ]
    ys = [0.68, 0.54, 0.40, 0.26]
    widths = [0.79, 0.74, 0.83, 0.70]
    fills = ["#DCE7F7", "#D8EBDD", "#E3DAF4", "#F0E0D8"]

    for idx, (label, y, w, fill) in enumerate(zip(labels, ys, widths, fills), start=1):
        x0 = 0.12
        bar = Rectangle((x0, y), w, 0.095, facecolor=fill, edgecolor=BLUE, lw=1.6, zorder=2)
        ax.add_patch(bar)
        ax.text(
            x0 + 0.018,
            y + 0.048,
            f"{idx:02d}",
            ha="left",
            va="center",
            color=RED,
            fontsize=18,
            fontfamily="DejaVu Sans Mono",
            fontweight="bold",
            zorder=3,
        )
        ax.text(
            x0 + 0.075,
            y + 0.048,
            label,
            ha="left",
            va="center",
            color=INK,
            fontsize=24,
            fontfamily="DejaVu Sans",
            fontweight="bold",
            zorder=3,
        )

    # Circuit-style side traces.
    x_trace_l = 0.07
    x_trace_r = 0.93
    ax.plot([x_trace_l, x_trace_l], [0.24, 0.78], color=BLUE, lw=1.2, alpha=0.55, zorder=1)
    ax.plot([x_trace_r, x_trace_r], [0.24, 0.78], color=BLUE, lw=1.2, alpha=0.55, zorder=1)
    for y in [0.727, 0.587, 0.447, 0.307]:
        ax.plot([x_trace_l, 0.12], [y, y], color=BLUE, lw=1.1, alpha=0.55, zorder=1)
        ax.plot([0.12, 0.12], [y - 0.032, y], color=BLUE, lw=1.1, alpha=0.55, zorder=1)
        ax.add_patch(Circle((x_trace_l, y), 0.0045, fc=RED, ec="none", zorder=2))
        ax.plot([0.12 + 0.70, x_trace_r], [y, y], color=BLUE, lw=1.1, alpha=0.55, zorder=1)
        ax.add_patch(Circle((x_trace_r, y), 0.0045, fc=RED, ec="none", zorder=2))


def slide_3_interface(ax: plt.Axes) -> None:
    ax.text(
        0.07,
        0.90,
        "THE INTERFACE",
        ha="left",
        va="top",
        color=BLUE,
        fontsize=54,
        fontfamily="DejaVu Sans",
        fontweight="heavy",
    )
    ax.text(
        0.07,
        0.84,
        "Annotated clinical intelligence view",
        ha="left",
        va="top",
        color=MUTED,
        fontsize=19,
        fontfamily="DejaVu Sans",
        fontweight="medium",
    )

    x0, y0, w, h = 0.08, 0.12, 0.78, 0.66
    frame = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.01",
        linewidth=1.5,
        edgecolor=BLUE,
        facecolor="#FFFFFF",
        zorder=2,
    )
    ax.add_patch(frame)

    toolbar_h = 0.055
    ax.add_patch(Rectangle((x0, y0 + h - toolbar_h), w, toolbar_h, facecolor="#DDE2E8", edgecolor="none", zorder=3))
    for i, dot_c in enumerate([RED, "#FFCC00", "#34C759"]):
        ax.add_patch(Circle((x0 + 0.022 + i * 0.022, y0 + h - toolbar_h / 2), 0.0075, fc=dot_c, ec="none", zorder=4))

    img_x0, img_x1 = x0 + 0.012, x0 + w - 0.012
    img_y0, img_y1 = y0 + 0.016, y0 + h - toolbar_h - 0.012
    if HERO_FRAME.exists():
        image = plt.imread(str(HERO_FRAME))
        ax.imshow(image, extent=(img_x0, img_x1, img_y0, img_y1), aspect="auto", zorder=2.5)
    else:
        ax.add_patch(Rectangle((img_x0, img_y0), img_x1 - img_x0, img_y1 - img_y0, facecolor="#ECEFF3", edgecolor="none", zorder=2.5))
        ax.text(
            img_x0 + 0.02,
            (img_y0 + img_y1) / 2,
            "Screenshot Placeholder",
            ha="left",
            va="center",
            color=MUTED,
            fontsize=16,
            fontfamily="DejaVu Sans Mono",
            fontweight="bold",
            zorder=3,
        )

    # Callouts with annotation lines + terminal dots.
    callouts = [
        ((x0 + 0.22, y0 + 0.56), (0.90, 0.69), "Sentiment Score"),
        ((x0 + 0.22, y0 + 0.51), (0.90, 0.59), "Risk Flag Proxy"),
        ((x0 + 0.47, y0 + 0.33), (0.90, 0.41), "Speaker-Tagged Transcript"),
    ]
    for (px, py), (lx, ly), label in callouts:
        ax.plot([px, lx - 0.01], [py, ly], color=INK, lw=1.0, zorder=5)
        ax.add_patch(Circle((px, py), 0.0042, fc=INK, ec="none", zorder=5))
        ax.text(
            lx,
            ly,
            label,
            ha="left",
            va="center",
            color=INK,
            fontsize=15,
            fontfamily="DejaVu Sans",
            fontweight="bold",
            zorder=5,
        )


def draw_code_segments(
    ax: plt.Axes,
    x: float,
    y: float,
    segments: list[tuple[str, str]],
    *,
    fontsize: int = 14,
    char_w: float = 0.0072,
) -> None:
    cursor = x
    for text, color in segments:
        ax.text(
            cursor,
            y,
            text,
            ha="left",
            va="center",
            color=color,
            fontsize=fontsize,
            fontfamily="DejaVu Sans Mono",
            fontweight="medium",
            zorder=4,
        )
        cursor += len(text) * char_w


def slide_4_terminal(ax: plt.Axes) -> None:
    ax.text(
        0.07,
        0.90,
        "THE TERMINAL",
        ha="left",
        va="top",
        color=BLUE,
        fontsize=54,
        fontfamily="DejaVu Sans",
        fontweight="heavy",
    )
    ax.text(
        0.07,
        0.84,
        "Open Source & Reproducible",
        ha="left",
        va="top",
        color=MUTED,
        fontsize=19,
        fontfamily="DejaVu Sans",
        fontweight="medium",
    )

    tx0, ty0, tw, th = 0.08, 0.14, 0.84, 0.64
    term = FancyBboxPatch(
        (tx0, ty0),
        tw,
        th,
        boxstyle="round,pad=0.005,rounding_size=0.018",
        linewidth=1.0,
        edgecolor="#2E3340",
        facecolor="#1E1E1E",
        zorder=2,
    )
    ax.add_patch(term)

    tab_h = 0.08
    ax.add_patch(Rectangle((tx0, ty0 + th - tab_h), tw, tab_h, facecolor="#252526", edgecolor="none", zorder=3))
    tab = FancyBboxPatch(
        (tx0 + 0.02, ty0 + th - tab_h + 0.015),
        0.16,
        0.046,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        linewidth=0,
        edgecolor="none",
        facecolor="#333333",
        zorder=4,
    )
    ax.add_patch(tab)
    ax.text(
        tx0 + 0.034,
        ty0 + th - tab_h / 2 + 0.002,
        "pipeline.py",
        ha="left",
        va="center",
        color="#D4D4D4",
        fontsize=13,
        fontfamily="DejaVu Sans Mono",
        fontweight="bold",
        zorder=5,
    )

    kw = "#C586C0"
    mod = "#4FC1FF"
    fn = "#DCDCAA"
    txt = "#D4D4D4"
    st = "#CE9178"
    cm = "#6A9955"
    num = "#B5CEA8"
    ln_color = "#6E7681"

    code = [
        [("import ", kw), ("whisper", mod)],
        [("from ", kw), ("pyannote.audio ", mod), ("import ", kw), ("Pipeline", mod)],
        [("", txt)],
        [("def ", kw), ("run_pipeline", fn), ("(audio_path: str):", txt)],
        [("    ", txt), ("model", txt), (" = whisper.load_model(", txt), ('"base"', st), (")", txt)],
        [("    ", txt), ("result", txt), (" = model.transcribe(audio_path)", txt)],
        [("    ", txt), ("dia", txt), (" = Pipeline.from_pretrained(", txt), ('"pyannote/speaker-diarization"', st), (")", txt)],
        [("    ", txt), ("segments", txt), (" = dia(audio_path)", txt)],
        [("    ", txt), ("print", fn), ("(", txt), ('"Sentiment score:", ', st), ("score", txt), (")", txt)],
        [("    ", txt), ("return ", kw), ("result, segments", txt)],
        [("", txt)],
        [("# Fully offline clinical workflow", cm)],
        [("if ", kw), ("__name__", txt), (" == ", txt), ('"__main__"', st), (":", txt)],
        [("    ", txt), ("run_pipeline", fn), ("(", txt), ('"session.m4a"', st), (")", txt)],
    ]

    y_start = ty0 + th - 0.12
    line_h = 0.039
    x_ln = tx0 + 0.03
    x_code = tx0 + 0.08
    for i, segs in enumerate(code, start=1):
        y = y_start - i * line_h
        ax.text(
            x_ln,
            y,
            f"{i:>2}",
            ha="right",
            va="center",
            color=ln_color,
            fontsize=12,
            fontfamily="DejaVu Sans Mono",
            fontweight="medium",
            zorder=4,
        )
        draw_code_segments(ax, x_code, y, segs, fontsize=14, char_w=0.0070)

    ax.text(
        tx0 + 0.03,
        ty0 + 0.03,
        "Built on OpenAI Whisper + Pyannote + Python",
        ha="left",
        va="bottom",
        color=num,
        fontsize=12,
        fontfamily="DejaVu Sans Mono",
        fontweight="bold",
        zorder=4,
    )


def draw_underlined_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    fontsize: int,
    weight: str = "medium",
    char_w: float = 0.0070,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="baseline",
        color=color,
        fontsize=fontsize,
        fontfamily="DejaVu Sans",
        fontweight=weight,
        zorder=3,
    )
    approx_w = len(text) * char_w
    ax.plot([x, x + approx_w], [y - 0.008, y - 0.008], color=color, lw=1.4, zorder=3)


def slide_5_cta(ax: plt.Axes) -> None:
    ax.text(
        0.07,
        0.12,
        "DEPLOY THE INFRASTRUCTURE.",
        ha="left",
        va="bottom",
        color=BLUE,
        fontsize=52,
        fontfamily="DejaVu Sans",
        fontweight="heavy",
        zorder=3,
    )

    # QR placeholder on center-right.
    qx, qy, qs = 0.64, 0.40, 0.23
    ax.add_patch(Rectangle((qx, qy), qs, qs, facecolor="#FFFFFF", edgecolor=BLUE, lw=2.0, zorder=2))
    ax.add_patch(Rectangle((qx + 0.03, qy + 0.03), qs - 0.06, qs - 0.06, facecolor=BG, edgecolor=BLUE, lw=1.0, zorder=2))
    ax.plot([qx + 0.03, qx + qs - 0.03], [qy + 0.03, qy + qs - 0.03], color=BLUE, lw=0.9, alpha=0.35, zorder=2)
    ax.plot([qx + 0.03, qx + qs - 0.03], [qy + qs - 0.03, qy + 0.03], color=BLUE, lw=0.9, alpha=0.35, zorder=2)
    ax.text(
        qx + qs / 2,
        qy - 0.035,
        "QR PLACEHOLDER",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=11,
        fontfamily="DejaVu Sans Mono",
        fontweight="bold",
        zorder=3,
    )

    draw_underlined_text(
        ax,
        0.07,
        0.085,
        "huggingface.co/spaces/ChengdongPeter/Clinical-Whisper",
        color=BLUE,
        fontsize=16,
        weight="bold",
        char_w=0.0082,
    )
    draw_underlined_text(
        ax,
        0.07,
        0.055,
        "github.com/czhou732/Clinical-Whisper-Pipeline",
        color=BLUE,
        fontsize=16,
        weight="bold",
        char_w=0.0080,
    )

    ax.text(
        0.07,
        0.028,
        "Chengdong (Peter) Zhou | USC",
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=13,
        fontfamily="DejaVu Sans",
        fontweight="medium",
        zorder=3,
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for painter in [slide_1_cover, slide_2_stack, slide_3_interface, slide_4_terminal, slide_5_cta]:
            fig, ax = new_canvas()
            painter(ax)
            pdf.savefig(fig, dpi=DPI, facecolor=BG)
            plt.close(fig)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
