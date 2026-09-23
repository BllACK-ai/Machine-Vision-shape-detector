"""Generate A3 printable cutting templates for all supported shapes."""

from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


A3_WIDTH_MM = 297
A3_HEIGHT_MM = 420
MARGIN_MM = 18
OUTPUT_DIR = Path("templates")
OUTPUT_FILE = OUTPUT_DIR / "shapes_templates.pdf"

LINE_WIDTH_PT = 2
METADATA_FONT_SIZE = 9
SCALE_FONT_SIZE = 9


SHAPES = [
    ("Triangle", "polygon", 3),
    ("Square", "square", None),
    ("Rectangle", "rectangle", None),
    ("Pentagon", "polygon", 5),
    ("Hexagon", "polygon", 6),
    ("Heptagon", "polygon", 7),
    ("Octagon", "polygon", 8),
    ("Nonagon", "polygon", 9),
    ("Decagon", "polygon", 10),
    ("Circle", "circle", None),
    ("Ellipse", "ellipse", None),
]


def polygon_unit_points(sides: int) -> list[tuple[float, float]]:
    """Return regular polygon vertices on a unit circle, one vertex facing up."""
    points = []
    rotation = math.pi / 2

    for index in range(sides):
        angle = rotation + (2 * math.pi * index / sides)
        points.append((math.cos(angle), math.sin(angle)))

    return points


def bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def draw_metadata(
    pdf: canvas.Canvas,
    shape_name: str,
    width_mm: float,
    height_mm: float,
) -> None:
    """Draw record text and a 5cm scale marker outside the shape area."""
    left = MARGIN_MM * mm
    top = (A3_HEIGHT_MM - MARGIN_MM) * mm
    bottom = MARGIN_MM * mm

    pdf.setFont("Helvetica", METADATA_FONT_SIZE)
    pdf.drawString(left, top, f"{shape_name}: {width_mm:.1f} mm x {height_mm:.1f} mm")
    pdf.drawString(left, top - 4.5 * mm, f"({width_mm / 10:.1f} cm x {height_mm / 10:.1f} cm)")

    scale_x = left
    scale_y = bottom
    pdf.setLineWidth(1)
    pdf.line(scale_x, scale_y, scale_x + 50 * mm, scale_y)
    pdf.line(scale_x, scale_y - 2 * mm, scale_x, scale_y + 2 * mm)
    pdf.line(scale_x + 50 * mm, scale_y - 2 * mm, scale_x + 50 * mm, scale_y + 2 * mm)
    pdf.setFont("Helvetica", SCALE_FONT_SIZE)
    pdf.drawString(scale_x + 17 * mm, scale_y + 3 * mm, "5cm")


def draw_polygon_template(pdf: canvas.Canvas, shape_name: str, sides: int) -> None:
    """Draw a mathematically regular polygon as a centered outline."""
    available_width = A3_WIDTH_MM - 2 * MARGIN_MM
    available_height = A3_HEIGHT_MM - 2 * MARGIN_MM
    unit_points = polygon_unit_points(sides)
    min_x, min_y, max_x, max_y = bounds(unit_points)
    unit_width = max_x - min_x
    unit_height = max_y - min_y
    radius_mm = min(available_width / unit_width, available_height / unit_height)
    width_mm = unit_width * radius_mm
    height_mm = unit_height * radius_mm

    # Shift by the polygon bbox center so the visible shape is centered on A3.
    bbox_center_x = (min_x + max_x) / 2
    bbox_center_y = (min_y + max_y) / 2
    center_x = (A3_WIDTH_MM / 2) * mm
    center_y = (A3_HEIGHT_MM / 2) * mm

    path = pdf.beginPath()
    for index, (unit_x, unit_y) in enumerate(unit_points):
        x = center_x + (unit_x - bbox_center_x) * radius_mm * mm
        y = center_y + (unit_y - bbox_center_y) * radius_mm * mm
        if index == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    path.close()
    pdf.drawPath(path, stroke=1, fill=0)
    draw_metadata(pdf, shape_name, width_mm, height_mm)


def draw_box_template(
    pdf: canvas.Canvas,
    shape_name: str,
    width_mm: float,
    height_mm: float,
) -> None:
    """Draw a centered square or rectangle outline."""
    x = ((A3_WIDTH_MM - width_mm) / 2) * mm
    y = ((A3_HEIGHT_MM - height_mm) / 2) * mm
    pdf.rect(x, y, width_mm * mm, height_mm * mm, stroke=1, fill=0)
    draw_metadata(pdf, shape_name, width_mm, height_mm)


def draw_ellipse_template(
    pdf: canvas.Canvas,
    shape_name: str,
    width_mm: float,
    height_mm: float,
) -> None:
    """Draw a centered circle or ellipse outline."""
    left = ((A3_WIDTH_MM - width_mm) / 2) * mm
    bottom = ((A3_HEIGHT_MM - height_mm) / 2) * mm
    right = left + width_mm * mm
    top = bottom + height_mm * mm
    pdf.ellipse(left, bottom, right, top, stroke=1, fill=0)
    draw_metadata(pdf, shape_name, width_mm, height_mm)


def draw_shape(pdf: canvas.Canvas, name: str, shape_type: str, sides: int | None) -> None:
    """Draw one shape template on the current PDF page."""
    pdf.setLineWidth(LINE_WIDTH_PT)
    pdf.setStrokeColorRGB(0, 0, 0)

    available_width = A3_WIDTH_MM - 2 * MARGIN_MM
    if shape_type == "polygon" and sides is not None:
        draw_polygon_template(pdf, name, sides)
    elif shape_type == "square":
        side_mm = available_width
        draw_box_template(pdf, name, side_mm, side_mm)
    elif shape_type == "rectangle":
        width_mm = available_width
        height_mm = width_mm / 1.6
        draw_box_template(pdf, name, width_mm, height_mm)
    elif shape_type == "circle":
        diameter_mm = available_width
        draw_ellipse_template(pdf, name, diameter_mm, diameter_mm)
    elif shape_type == "ellipse":
        width_mm = available_width
        height_mm = width_mm / 1.6
        draw_ellipse_template(pdf, name, width_mm, height_mm)
    else:
        raise ValueError(f"Unknown shape definition for {name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    page_size = (A3_WIDTH_MM * mm, A3_HEIGHT_MM * mm)
    pdf = canvas.Canvas(str(OUTPUT_FILE), pagesize=page_size)

    for name, shape_type, sides in SHAPES:
        draw_shape(pdf, name, shape_type, sides)
        pdf.showPage()

    pdf.save()
    print(f"Saved {OUTPUT_FILE} with {len(SHAPES)} pages")


if __name__ == "__main__":
    main()
