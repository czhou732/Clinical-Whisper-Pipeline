#!/usr/bin/env python3
"""
Render a Mermaid .mmd file into an image (PNG/SVG/PDF) via Mermaid CLI.

Examples:
  python scripts/render_mermaid.py
  python scripts/render_mermaid.py --format svg
  python scripts/render_mermaid.py -i assets/diagrams/one_photon_pipeline.mmd -o assets/diagrams/pipeline.svg
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
ROOT = THIS_FILE.parents[1]
_candidate_default_inputs = [
    SCRIPT_DIR / "one_photon_pipeline.mmd",
    ROOT / "assets" / "diagrams" / "one_photon_pipeline.mmd",
]
DEFAULT_INPUT = next(
    (candidate for candidate in _candidate_default_inputs if candidate.exists()),
    _candidate_default_inputs[0],
)
VALID_FORMATS = {"png", "svg", "pdf"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Mermaid diagrams using Mermaid CLI (mmdc)."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to input Mermaid file (.mmd). Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image/document path. If omitted, uses <input_stem>_output.<format>.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=sorted(VALID_FORMATS),
        help="Output format. If omitted and --output has .png/.svg/.pdf, inferred from output extension.",
    )
    parser.add_argument(
        "--theme",
        choices=["default", "dark", "forest", "neutral", "base"],
        help="Mermaid theme.",
    )
    parser.add_argument(
        "--background",
        default="white",
        help="Background color (e.g. white, transparent, #ffffff). Default: white",
    )
    parser.add_argument("--width", type=int, help="Canvas width in pixels.")
    parser.add_argument("--height", type=int, help="Canvas height in pixels.")
    parser.add_argument(
        "--scale", type=float, help="Scale factor for the rendered output."
    )
    parser.add_argument(
        "--mmdc",
        help="Path to Mermaid CLI executable. If omitted, auto-detects 'mmdc'.",
    )
    parser.add_argument(
        "--puppeteer-config",
        help="Path to a Puppeteer config JSON file passed to mmdc via -p.",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Run Chromium with --no-sandbox and --disable-setuid-sandbox.",
    )
    parser.add_argument(
        "--no-npx-fallback",
        action="store_true",
        help="Do not use npx fallback when mmdc is not found.",
    )
    return parser


def infer_format_from_output(path: Path) -> str | None:
    suffix = path.suffix.lower().lstrip(".")
    return suffix if suffix in VALID_FORMATS else None


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = (ROOT / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input Mermaid file not found: {input_path}")

    output_path: Path
    chosen_format = args.format

    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = (ROOT / output_path).resolve()
        inferred = infer_format_from_output(output_path)
        if chosen_format is None:
            chosen_format = inferred or "png"
            if inferred is None:
                output_path = output_path.with_suffix(".png")
    else:
        chosen_format = chosen_format or "png"
        output_path = input_path.with_name(f"{input_path.stem}_output.{chosen_format}")

    if chosen_format is None:
        chosen_format = "png"

    if output_path.suffix.lower().lstrip(".") not in VALID_FORMATS:
        output_path = output_path.with_suffix(f".{chosen_format}")

    return input_path, output_path, chosen_format


def resolve_mermaid_cli(
    explicit_mmdc: str | None, allow_npx_fallback: bool
) -> tuple[list[str], str]:
    if explicit_mmdc:
        return [explicit_mmdc], explicit_mmdc

    mmdc_path = shutil.which("mmdc")
    if mmdc_path:
        return [mmdc_path], mmdc_path

    if allow_npx_fallback:
        npx_path = shutil.which("npx")
        if npx_path:
            return [npx_path, "-y", "@mermaid-js/mermaid-cli"], "npx @mermaid-js/mermaid-cli"

    raise RuntimeError(
        "Mermaid CLI not found.\n"
        "Install with: npm install -g @mermaid-js/mermaid-cli\n"
        "Or pass --mmdc /path/to/mmdc"
    )


def run_render(
    base_cmd: list[str],
    input_path: Path,
    output_path: Path,
    output_format: str,
    args: argparse.Namespace,
) -> int:
    cmd = base_cmd + [
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-e",
        output_format,
        "-b",
        args.background,
    ]

    if args.theme:
        cmd.extend(["-t", args.theme])
    if args.width is not None:
        cmd.extend(["-w", str(args.width)])
    if args.height is not None:
        cmd.extend(["-H", str(args.height)])
    if args.scale is not None:
        cmd.extend(["-s", str(args.scale)])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_config_path: Path | None = None
    if args.puppeteer_config:
        cmd.extend(["-p", str(Path(args.puppeteer_config).expanduser().resolve())])
    elif args.no_sandbox:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(
                {
                    "args": [
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                    ]
                },
                tmp,
            )
            temp_config_path = Path(tmp.name)
        cmd.extend(["-p", str(temp_config_path)])

    print("Rendering Mermaid diagram...")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        if temp_config_path and temp_config_path.exists():
            temp_config_path.unlink(missing_ok=True)

    if result.returncode != 0:
        if result.stdout.strip():
            print(result.stdout.strip(), file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        return result.returncode

    print(f"Done: {output_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        input_path, output_path, output_format = resolve_paths(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        base_cmd, cli_desc = resolve_mermaid_cli(
            explicit_mmdc=args.mmdc,
            allow_npx_fallback=not args.no_npx_fallback,
        )
        print(f"Renderer: {cli_desc}")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    return run_render(base_cmd, input_path, output_path, output_format, args)


if __name__ == "__main__":
    raise SystemExit(main())
