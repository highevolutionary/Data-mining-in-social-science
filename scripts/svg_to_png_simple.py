from __future__ import annotations

import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def num(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(str(value).replace("px", ""))


def color(value: str | None, opacity: float = 1.0) -> tuple[int, int, int, int] | None:
    if not value or value == "none":
        return None
    value = value.strip()
    if value.startswith("#") and len(value) == 7:
        return (
            int(value[1:3], 16),
            int(value[3:5], 16),
            int(value[5:7], 16),
            int(max(0, min(255, round(255 * opacity)))),
        )
    return (0, 0, 0, int(max(0, min(255, round(255 * opacity)))))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, elem: ET.Element) -> None:
    text = "".join(elem.itertext())
    text = html.unescape(text)
    x = num(elem.get("x"))
    y = num(elem.get("y"))
    size = int(num(elem.get("font-size"), 12))
    bold = elem.get("font-weight") in {"700", "bold"}
    fill = color(elem.get("fill"))
    if fill is None:
        return
    fnt = font(size, bold)
    anchor = elem.get("text-anchor", "start")
    bbox = draw.textbbox((0, 0), text, font=fnt)
    width = bbox[2] - bbox[0]
    if anchor == "middle":
        x -= width / 2
    elif anchor == "end":
        x -= width
    draw.text((x, y - size), text, fill=fill, font=fnt)


def render(svg_path: Path, png_path: Path, scale: int = 2) -> None:
    raw = svg_path.read_text(encoding="utf-8")
    root = ET.fromstring(raw)
    width = int(num(root.get("width"), 920))
    height = int(num(root.get("height"), 520))
    img = Image.new("RGBA", (width * scale, height * scale), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    def s(v: float) -> float:
        return v * scale

    for elem in root.iter():
        tag = re.sub(r"^\{.*\}", "", elem.tag)
        opacity = num(elem.get("opacity"), 1.0)
        if tag == "rect":
            x = s(num(elem.get("x")))
            y = s(num(elem.get("y")))
            w = s(num(elem.get("width")))
            h = s(num(elem.get("height")))
            fill = color(elem.get("fill"), opacity)
            outline = color(elem.get("stroke"), opacity)
            sw = max(1, int(s(num(elem.get("stroke-width"), 1))))
            draw.rectangle([x, y, x + w, y + h], fill=fill, outline=outline, width=sw if outline else 1)
        elif tag == "line":
            stroke = color(elem.get("stroke"), opacity) or (0, 0, 0, 255)
            sw = max(1, int(s(num(elem.get("stroke-width"), 1))))
            draw.line([s(num(elem.get("x1"))), s(num(elem.get("y1"))), s(num(elem.get("x2"))), s(num(elem.get("y2")))], fill=stroke, width=sw)
        elif tag == "circle":
            cx = s(num(elem.get("cx")))
            cy = s(num(elem.get("cy")))
            r = s(num(elem.get("r")))
            fill = color(elem.get("fill"), opacity)
            outline = color(elem.get("stroke"), opacity)
            sw = max(1, int(s(num(elem.get("stroke-width"), 1))))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=sw if outline else 1)

    # Draw text after shapes so labels stay on top.
    text_layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_layer)
    for elem in root.iter():
        tag = re.sub(r"^\{.*\}", "", elem.tag)
        if tag == "text":
            draw_text(text_draw, elem)
    text_layer = text_layer.resize((width * scale, height * scale), Image.Resampling.LANCZOS)
    img.alpha_composite(text_layer)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(png_path, "PNG", optimize=True)


def main() -> None:
    for arg in sys.argv[1:]:
        src = Path(arg)
        render(src, src.with_suffix(".png"))


if __name__ == "__main__":
    main()
