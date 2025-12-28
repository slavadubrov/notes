#!/usr/bin/env python3
"""
Convert SVG diagrams to high-quality PNG for Substack publishing.

Usage:
    # Convert only missing PNGs (incremental)
    uv run --with playwright python scripts/svg_to_png.py

    # Force regenerate all PNGs
    uv run --with playwright python scripts/svg_to_png.py --force

    # Convert specific SVG file
    uv run --with playwright python scripts/svg_to_png.py path/to/diagram.svg

First-time setup (downloads Chromium ~160MB):
    uv run --with playwright playwright install chromium
"""

import argparse
import re
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: playwright not installed. Run: uv run --with playwright ...")
    sys.exit(1)


# Default paths
BLOG_ASSETS_DIR = Path(__file__).parent.parent / "docs" / "blog" / "assets"


def get_svg_dimensions(svg_content: str) -> tuple[int, int]:
    """Extract dimensions from SVG viewBox or width/height attributes."""
    viewbox_match = re.search(
        r'viewBox="[^"]*\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"', svg_content
    )
    if viewbox_match:
        return int(float(viewbox_match.group(1))), int(float(viewbox_match.group(2)))

    width_match = re.search(r'width="(\d+(?:\.\d+)?)', svg_content)
    height_match = re.search(r'height="(\d+(?:\.\d+)?)', svg_content)
    if width_match and height_match:
        return int(float(width_match.group(1))), int(float(height_match.group(1)))

    return 800, 600


def convert_svg_to_png(
    svg_path: Path,
    scale: int = 2,
    background_color: str = "white",
    playwright_context=None,
) -> Path:
    """Convert a single SVG file to PNG (placed next to the SVG)."""
    svg_path = Path(svg_path).resolve()
    png_path = svg_path.with_suffix(".png")

    if not svg_path.exists():
        raise FileNotFoundError(f"SVG file not found: {svg_path}")

    png_path.parent.mkdir(parents=True, exist_ok=True)

    svg_content = svg_path.read_text()
    width, height = get_svg_dimensions(svg_content)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: {background_color};
            width: {width}px;
            height: {height}px;
        }}
        svg {{
            display: block;
            width: {width}px;
            height: {height}px;
        }}
    </style>
</head>
<body>
{svg_content}
</body>
</html>"""

    should_close = playwright_context is None
    if playwright_context is None:
        pw = sync_playwright().start()
        browser = pw.chromium.launch()
        playwright_context = browser.new_context(
            device_scale_factor=scale, viewport={"width": width, "height": height}
        )

    page = playwright_context.new_page()
    page.set_viewport_size({"width": width, "height": height})
    page.set_content(html_content)

    page.screenshot(
        path=str(png_path),
        omit_background=(background_color == "transparent"),
        full_page=True,
    )

    page.close()

    if should_close:
        browser.close()
        pw.stop()

    return png_path


def convert_all_svgs(
    source_dir: Path = BLOG_ASSETS_DIR,
    scale: int = 2,
    background_color: str = "white",
    force: bool = False,
) -> tuple[list[Path], list[Path]]:
    """
    Convert SVG files to PNG (only missing ones unless force=True).

    Returns:
        Tuple of (converted_paths, skipped_paths)
    """
    source_dir = Path(source_dir)
    svg_files = list(source_dir.rglob("*.svg"))

    if not svg_files:
        print(f"No SVG files found in {source_dir}")
        return [], []

    # Filter to only missing PNGs unless force mode
    to_convert = []
    skipped = []

    for svg_file in svg_files:
        png_file = svg_file.with_suffix(".png")
        if force or not png_file.exists():
            to_convert.append(svg_file)
        else:
            skipped.append(svg_file)

    if not to_convert:
        print(f"All {len(svg_files)} SVGs already have PNGs. Use --force to regenerate.")
        return [], skipped

    mode = "force" if force else "incremental"
    print(f"Converting {len(to_convert)} SVG files ({mode} mode)...")
    if skipped:
        print(f"Skipping {len(skipped)} existing PNGs")
    print(f"Scale: {scale}x | Background: {background_color}")
    print("-" * 50)

    converted = []
    errors = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()

        for svg_file in sorted(to_convert):
            try:
                svg_content = svg_file.read_text()
                width, height = get_svg_dimensions(svg_content)
                context = browser.new_context(
                    device_scale_factor=scale, viewport={"width": width, "height": height}
                )

                png_path = convert_svg_to_png(
                    svg_file,
                    scale=scale,
                    background_color=background_color,
                    playwright_context=context,
                )
                converted.append(png_path)
                print(f"✓ {svg_file.name} → {png_path.name} ({width*scale}x{height*scale}px)")
                context.close()

            except Exception as e:
                errors.append((svg_file, str(e)))
                print(f"✗ {svg_file.name}: {e}")

        browser.close()

    print("-" * 50)
    print(f"✓ Converted: {len(converted)}/{len(to_convert)}")

    if errors:
        print(f"✗ Errors: {len(errors)}")
        sys.exit(1)

    return converted, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert SVG diagrams to high-quality PNG for Substack"
    )
    parser.add_argument(
        "svg_path",
        nargs="?",
        help="Path to specific SVG file (converts all if not specified)",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=int,
        default=2,
        help="Scale factor for resolution (default: 2 for retina)",
    )
    parser.add_argument(
        "--background",
        "-b",
        type=str,
        default="white",
        help="Background color (default: white)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force regenerate all PNGs even if they exist",
    )

    args = parser.parse_args()

    if args.svg_path:
        png_path = convert_svg_to_png(
            Path(args.svg_path),
            scale=args.scale,
            background_color=args.background,
        )
        print(f"✓ {args.svg_path} → {png_path}")
    else:
        convert_all_svgs(
            scale=args.scale,
            background_color=args.background,
            force=args.force,
        )


if __name__ == "__main__":
    main()
