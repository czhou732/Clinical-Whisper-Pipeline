"""Draw Figure 1 of the JOSS paper: the ClinicalWhisper processing pipeline.

The figure is a schematic, not a data plot. It shows the core steps, the two
optional add-ons, and the points where a job stops or waits. Every connector
is straight or turns at right angles. Run from the repo root:

    python paper/figures/make_pipeline_figure.py

The PNG is written next to this script.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).with_name("pipeline.png")
DPI = 300

INK = "#1f2933"
CORE_FILL = "#e8eef5"
CORE_EDGE = "#3d5a80"
ADDON_FILL = "#ffffff"
STOP_FILL = "#fbeaea"
STOP_EDGE = "#a4161a"
OUT_FILL = "#eef5ea"
OUT_EDGE = "#3a6b35"

Point = tuple[float, float]


@dataclass(frozen=True)
class Box:
    """A labelled box centred at (x, y) in axis units (0-100 by 0-62)."""

    x: float
    y: float
    w: float
    h: float
    title: str
    detail: str = ""
    title_dy: float = 1.8
    detail_dy: float = -1.9

    @property
    def left(self) -> Point:
        return (self.x - self.w / 2, self.y)

    @property
    def right(self) -> Point:
        return (self.x + self.w / 2, self.y)

    @property
    def top(self) -> Point:
        return (self.x, self.y + self.h / 2)

    @property
    def bottom(self) -> Point:
        return (self.x, self.y - self.h / 2)


def draw_box(ax: plt.Axes, box: Box, fill: str, edge: str, dashed: bool = False) -> None:
    """Draw one box with a bold title line and an optional detail line."""
    ax.add_patch(FancyBboxPatch(
        (box.x - box.w / 2, box.y - box.h / 2), box.w, box.h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        facecolor=fill, edgecolor=edge, linewidth=1.2,
        linestyle=(0, (4, 2)) if dashed else "solid",
    ))
    if not box.detail:
        ax.text(box.x, box.y, box.title, ha="center", va="center",
                fontsize=8, fontweight="bold", color=INK)
        return
    ax.text(box.x, box.y + box.title_dy, box.title, ha="center", va="center",
            fontsize=8, fontweight="bold", color=INK)
    ax.text(box.x, box.y + box.detail_dy, box.detail, ha="center", va="center",
            fontsize=6.6, color=INK, linespacing=1.25)


def connect(ax: plt.Axes, points: list[Point], dashed: bool = False) -> None:
    """Draw a connector through ``points`` with an arrowhead on the last segment.

    Pass only horizontal and vertical segments to keep the figure orthogonal.
    """
    style = (0, (4, 2)) if dashed else "solid"
    if len(points) > 2:
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color=INK, linewidth=1.0, linestyle=style,
                solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=9, color=INK,
        linewidth=1.0, linestyle=style, shrinkA=0, shrinkB=1.5,
    ))


def label(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """Write a small italic connector label."""
    ax.text(x, y, text, ha="center", va="bottom", fontsize=6.4,
            style="italic", color=INK, linespacing=1.15)


def stop_note(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """Draw a red-outlined note for a point where the job stops or waits."""
    ax.text(x, y, text, ha="center", va="center", fontsize=6.4, color=STOP_EDGE,
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=STOP_FILL,
                      edgecolor=STOP_EDGE, linewidth=0.9))


def build_figure() -> plt.Figure:
    """Lay out the pipeline schematic and return the figure."""
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    row = 37.0
    recording = Box(5.5, row, 9.5, 9, "Recording", "WAV, MP3,\nM4A, FLAC")
    decode = Box(17.5, row, 11.5, 9, "Decode", "PyAV, 16 kHz\nmono, in blocks")
    moss = Box(38, row, 15, 9, "Transcribe", "MOSS-Transcribe-Diarize:\ntimestamped speakers")
    mask = Box(55, row, 15, 9, "Mask identifiers", "OpenMed model, rules,\nname sweep")
    roles = Box(72, row, 15, 9, "Label roles", "Interviewer, Subject\nor Other")
    output = Box(91, row, 14, 14, "Results",
                 "JSON record and CSV:\ntranscript, measures,\nscores, failures,\nmodel versions",
                 title_dy=4.6, detail_dy=-1.6)
    silence = Box(55, 54, 15, 9, "Silence names", "Add-on: wav2vec 2.0\nalignment, English")
    scores = Box(72, 54, 15, 9, "Clinical scores", "Add-on: Llama 3 8B,\nfour 0-3 scores")
    smile = Box(38, 10, 15, 9, "Voice measures", "openSMILE eGeMAPSv02,\nper speaker")

    for box in (recording, decode, moss, mask, roles, smile):
        draw_box(ax, box, CORE_FILL, CORE_EDGE)
    draw_box(ax, output, OUT_FILL, OUT_EDGE)
    for box in (silence, scores):
        draw_box(ax, box, ADDON_FILL, CORE_EDGE, dashed=True)

    # Main row: one straight line from the recording to the results.
    connect(ax, [recording.right, decode.left])
    connect(ax, [decode.right, moss.left])
    label(ax, (decode.right[0] + moss.left[0]) / 2, row + 0.6, "normalized\ncopy")
    connect(ax, [moss.right, mask.left])
    connect(ax, [mask.right, roles.left])
    connect(ax, [roles.right, output.left])

    # Voice branch: down from Decode, across to openSMILE, across and up into Results.
    connect(ax, [decode.bottom, (decode.x, smile.y), smile.left])
    label(ax, (decode.x + smile.left[0]) / 2 + 0.6, smile.y + 0.6, "original-gain copy")
    connect(ax, [smile.right, (output.x, smile.y), output.bottom])

    # Add-ons: straight up from the step they build on.
    connect(ax, [mask.top, silence.bottom], dashed=True)
    connect(ax, [roles.top, scores.bottom], dashed=True)
    connect(ax, [scores.right, (output.x, scores.y), output.top], dashed=True)

    # Stop and wait points: straight down from the step that triggers them.
    note_y = 22.5
    stop_note(ax, mask.x, note_y, "Stops if the masking model\nis missing, fails (strict\n"
                                  "mode) or can't read\nthe language")
    stop_note(ax, roles.x, note_y, "Unclear roles: flagged;\nparticipant measures and\n"
                                   "scores wait until roles\nare confirmed")
    connect(ax, [mask.bottom, (mask.x, note_y + 3.1)])
    connect(ax, [roles.bottom, (roles.x, note_y + 3.1)])

    ax.text(1, 1.5, "Solid boxes: core app.  Dashed boxes: optional add-ons.  "
                    "Red notes: where a job stops or waits.  Everything runs on the device.",
            fontsize=6.6, color=INK, ha="left")
    return fig


def main() -> None:
    """Render the figure to OUT_PATH."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fig = build_figure()
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
