"""Generate realistic composite test photos by placing shapes on backgrounds.

The script takes real background photos, places all supported shapes onto each
photo with random non-overlapping positions, and writes matching ground-truth
JSON files for later detector validation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter


# Adjustable rendering constants.
DEFAULT_BACKGROUND_DIR = Path("backgrounds")
DEFAULT_OUTPUT_DIR = Path("composite_input")
COMPOSITES_PER_BACKGROUND = 1

SHAPE_COLOR = (220, 74, 32, 255)  # painted cardboard orange-red, RGBA
EDGE_BLUR_RADIUS = 0.8

SHADOW_COLOR = (0, 0, 0)
SHADOW_OFFSET = (8, 10)
SHADOW_BLUR_RADIUS = 5.0
SHADOW_OPACITY = 85  # 0-255

SCALE_VARIATION_RANGE = (0.80, 1.20)
BASE_SHAPE_SIZE_FRACTION = 0.12  # fraction of the shorter image edge
MIN_SHAPE_SIZE_PX = 36
MAX_SHAPE_SIZE_PX = 150
MIN_SPACING_PX = 18

MAX_PLACEMENT_ATTEMPTS_PER_SHAPE = 3000
MAX_LAYOUT_RETRIES = 8
LAYOUT_SHRINK_FACTOR = 0.90

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ShapeDefinition:
    name: str
    kind: str
    sides: int | None = None
    aspect_ratio: float = 1.0


@dataclass
class RenderedShape:
    definition: ShapeDefinition
    image: Image.Image
    alpha: Image.Image
    rotation_degrees: float
    scale: float
    nominal_size_px: int
    placement_radius: float


@dataclass
class PlacedShape:
    rendered: RenderedShape
    center: tuple[int, int]


SHAPES = [
    ShapeDefinition("Triangle", "polygon", sides=3),
    ShapeDefinition("Square", "box"),
    ShapeDefinition("Rectangle", "box", aspect_ratio=1.6),
    ShapeDefinition("Pentagon", "polygon", sides=5),
    ShapeDefinition("Hexagon", "polygon", sides=6),
    ShapeDefinition("Heptagon", "polygon", sides=7),
    ShapeDefinition("Octagon", "polygon", sides=8),
    ShapeDefinition("Nonagon", "polygon", sides=9),
    ShapeDefinition("Decagon", "polygon", sides=10),
    ShapeDefinition("Circle", "ellipse"),
    ShapeDefinition("Ellipse", "ellipse", aspect_ratio=1.6),
]


def polygon_points(
    center: tuple[float, float],
    radius: float,
    sides: int,
    rotation_degrees: float = -90.0,
) -> list[tuple[float, float]]:
    """Return regular polygon vertices around a center point."""
    cx, cy = center
    start_angle = math.radians(rotation_degrees)
    points = []

    for index in range(sides):
        angle = start_angle + (2 * math.pi * index / sides)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    return points


def draw_shape_mask(definition: ShapeDefinition, nominal_size_px: int) -> Image.Image:
    """Draw a single unrotated shape mask before rotation and edge softening."""
    width = max(1, round(nominal_size_px * definition.aspect_ratio))
    height = max(1, nominal_size_px)
    padding = max(8, round(nominal_size_px * 0.10))
    mask = Image.new("L", (width + padding * 2, height + padding * 2), 0)
    draw = ImageDraw.Draw(mask)

    left = padding
    top = padding
    right = padding + width
    bottom = padding + height

    if definition.kind == "polygon" and definition.sides is not None:
        radius = min(width, height) / 2
        center = (padding + width / 2, padding + height / 2)
        draw.polygon(polygon_points(center, radius, definition.sides), fill=255)
    elif definition.kind == "box":
        draw.rectangle((left, top, right, bottom), fill=255)
    elif definition.kind == "ellipse":
        draw.ellipse((left, top, right, bottom), fill=255)
    else:
        raise ValueError(f"Unknown shape definition: {definition}")

    return mask


def render_shape(
    definition: ShapeDefinition,
    base_shape_size_px: int,
    rng: random.Random,
) -> RenderedShape:
    """Create a rotated RGBA shape layer and its softened alpha mask."""
    scale = rng.uniform(*SCALE_VARIATION_RANGE)
    rotation = rng.uniform(0, 359.999)
    nominal_size_px = max(1, round(base_shape_size_px * scale))

    mask = draw_shape_mask(definition, nominal_size_px)
    resampling = Image.Resampling.BICUBIC
    rotated_alpha = mask.rotate(rotation, resample=resampling, expand=True, fillcolor=0)
    if EDGE_BLUR_RADIUS > 0:
        rotated_alpha = rotated_alpha.filter(ImageFilter.GaussianBlur(EDGE_BLUR_RADIUS))

    shape = Image.new("RGBA", rotated_alpha.size, SHAPE_COLOR)
    shape.putalpha(rotated_alpha)

    shadow_extra = max(abs(SHADOW_OFFSET[0]), abs(SHADOW_OFFSET[1])) + SHADOW_BLUR_RADIUS
    placement_radius = math.hypot(*rotated_alpha.size) / 2 + shadow_extra

    return RenderedShape(
        definition=definition,
        image=shape,
        alpha=rotated_alpha,
        rotation_degrees=rotation,
        scale=scale,
        nominal_size_px=nominal_size_px,
        placement_radius=placement_radius,
    )


def generate_rendered_shapes(
    base_shape_size_px: int,
    rng: random.Random,
) -> list[RenderedShape]:
    """Render every supported shape with independent random rotation and scale."""
    return [render_shape(definition, base_shape_size_px, rng) for definition in SHAPES]


def random_center(
    image_size: tuple[int, int],
    radius: float,
    rng: random.Random,
) -> tuple[int, int] | None:
    """Pick a random center that keeps a rendered shape inside the image."""
    width, height = image_size
    margin = math.ceil(radius)

    if width <= margin * 2 or height <= margin * 2:
        return None

    return (
        rng.randint(margin, width - margin),
        rng.randint(margin, height - margin),
    )


def distance(first: tuple[int, int], second: tuple[int, int]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def place_shapes(
    rendered_shapes: list[RenderedShape],
    image_size: tuple[int, int],
    rng: random.Random,
) -> list[PlacedShape] | None:
    """Place all shapes using conservative radius-based spacing checks."""
    placed: list[PlacedShape] = []
    shapes_to_place = rendered_shapes[:]
    rng.shuffle(shapes_to_place)

    # Large shapes are hardest to fit, so place them first after shuffling ties.
    shapes_to_place.sort(key=lambda item: item.placement_radius, reverse=True)

    for rendered in shapes_to_place:
        for _ in range(MAX_PLACEMENT_ATTEMPTS_PER_SHAPE):
            center = random_center(image_size, rendered.placement_radius, rng)
            if center is None:
                return None

            overlaps = any(
                distance(center, existing.center)
                < rendered.placement_radius
                + existing.rendered.placement_radius
                + MIN_SPACING_PX
                for existing in placed
            )
            if not overlaps:
                placed.append(PlacedShape(rendered=rendered, center=center))
                break
        else:
            return None

    placed.sort(key=lambda item: item.rendered.definition.name)
    return placed


def alpha_from_shadow(alpha: Image.Image) -> Image.Image:
    """Create a blurred translucent shadow mask from a shape alpha mask."""
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
    return shadow_alpha.point(lambda value: round(value * SHADOW_OPACITY / 255))


def paste_layer(
    background: Image.Image,
    layer: Image.Image,
    center: tuple[int, int],
    offset: tuple[int, int] = (0, 0),
) -> tuple[int, int]:
    """Alpha-composite a layer centered on the requested point."""
    x = round(center[0] - layer.width / 2 + offset[0])
    y = round(center[1] - layer.height / 2 + offset[1])
    background.alpha_composite(layer, dest=(x, y))
    return x, y


def composite_shapes(background: Image.Image, placed_shapes: list[PlacedShape]) -> Image.Image:
    """Overlay shadows and shapes onto a background photo."""
    result = background.convert("RGBA")

    for placed in placed_shapes:
        shadow_alpha = alpha_from_shadow(placed.rendered.alpha)
        shadow = Image.new("RGBA", shadow_alpha.size, (*SHADOW_COLOR, 0))
        shadow.putalpha(shadow_alpha)
        paste_layer(result, shadow, placed.center, SHADOW_OFFSET)
        paste_layer(result, placed.rendered.image, placed.center)

    return result.convert("RGB")


def groundtruth_records(placed_shapes: list[PlacedShape]) -> list[dict[str, object]]:
    """Build JSON-serializable records for each placed shape."""
    records = []

    for placed in placed_shapes:
        rendered = placed.rendered
        left = round(placed.center[0] - rendered.image.width / 2)
        top = round(placed.center[1] - rendered.image.height / 2)
        records.append(
            {
                "name": rendered.definition.name,
                "center": {"x": placed.center[0], "y": placed.center[1]},
                "rotation_degrees": round(rendered.rotation_degrees, 4),
                "scale": round(rendered.scale, 4),
                "nominal_size_px": rendered.nominal_size_px,
                "bbox": {
                    "left": left,
                    "top": top,
                    "width": rendered.image.width,
                    "height": rendered.image.height,
                },
            }
        )

    return records


def find_background_images(background_dir: Path) -> list[Path]:
    """Return supported image files from a background folder."""
    return sorted(
        path
        for path in background_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def fit_layout(
    image_size: tuple[int, int],
    initial_base_size_px: int,
    rng: random.Random,
) -> tuple[int, list[PlacedShape]]:
    """Render and place shapes, shrinking slightly if the photo is crowded."""
    base_size_px = initial_base_size_px

    for _ in range(MAX_LAYOUT_RETRIES):
        rendered_shapes = generate_rendered_shapes(base_size_px, rng)
        placed_shapes = place_shapes(rendered_shapes, image_size, rng)
        if placed_shapes is not None:
            return base_size_px, placed_shapes

        base_size_px = max(MIN_SHAPE_SIZE_PX, round(base_size_px * LAYOUT_SHRINK_FACTOR))

    raise RuntimeError(
        "Could not place all shapes without overlap. Try a larger background image, "
        "lower BASE_SHAPE_SIZE_FRACTION, or lower MIN_SPACING_PX."
    )


def output_stem(background_path: Path, composite_index: int) -> str:
    return f"{background_path.stem}_composite_{composite_index:02d}"


def save_composite(
    background_path: Path,
    output_dir: Path,
    composite_index: int,
    rng: random.Random,
    forced_base_size_px: int | None,
) -> tuple[Path, Path]:
    """Generate one composite image and its ground-truth JSON file."""
    with Image.open(background_path) as opened_background:
        background = opened_background.convert("RGB")

    min_edge = min(background.size)
    default_base_size_px = round(min_edge * BASE_SHAPE_SIZE_FRACTION)
    default_base_size_px = max(MIN_SHAPE_SIZE_PX, min(MAX_SHAPE_SIZE_PX, default_base_size_px))
    initial_base_size_px = forced_base_size_px or default_base_size_px

    used_base_size_px, placed_shapes = fit_layout(background.size, initial_base_size_px, rng)
    composite = composite_shapes(background, placed_shapes)

    stem = output_stem(background_path, composite_index)
    image_path = output_dir / f"{stem}.png"
    groundtruth_path = output_dir / f"{stem}_groundtruth.json"

    composite.save(image_path)
    groundtruth = {
        "source_background": str(background_path),
        "output_image": str(image_path),
        "image_size": {"width": composite.width, "height": composite.height},
        "base_shape_size_px": used_base_size_px,
        "shape_color_rgba": list(SHAPE_COLOR),
        "edge_blur_radius": EDGE_BLUR_RADIUS,
        "shadow": {
            "offset": {"x": SHADOW_OFFSET[0], "y": SHADOW_OFFSET[1]},
            "blur_radius": SHADOW_BLUR_RADIUS,
            "opacity": SHADOW_OPACITY,
        },
        "min_spacing_px": MIN_SPACING_PX,
        "shapes": groundtruth_records(placed_shapes),
    }
    groundtruth_path.write_text(json.dumps(groundtruth, indent=2), encoding="utf-8")

    return image_path, groundtruth_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay all supported shapes onto background photos."
    )
    parser.add_argument(
        "background_dir",
        nargs="?",
        default=str(DEFAULT_BACKGROUND_DIR),
        help="Folder containing real background images. Default: backgrounds/",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder for composite images and ground-truth JSON files.",
    )
    parser.add_argument(
        "--count-per-background",
        type=positive_int,
        default=COMPOSITES_PER_BACKGROUND,
        help="Number of random composites to create per background image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable composites.",
    )
    parser.add_argument(
        "--base-size",
        type=positive_int,
        default=None,
        help="Optional nominal shape height in pixels before random scale variation.",
    )
    return parser.parse_args()


def generate_all(
    background_paths: Iterable[Path],
    output_dir: Path,
    count_per_background: int,
    rng: random.Random,
    forced_base_size_px: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for background_path in background_paths:
        for composite_index in range(1, count_per_background + 1):
            image_path, groundtruth_path = save_composite(
                background_path=background_path,
                output_dir=output_dir,
                composite_index=composite_index,
                rng=rng,
                forced_base_size_px=forced_base_size_px,
            )
            print(f"Saved {image_path} and {groundtruth_path}")


def main() -> None:
    args = parse_args()
    background_dir = Path(args.background_dir)
    output_dir = Path(args.output_dir)

    if not background_dir.exists() or not background_dir.is_dir():
        raise SystemExit(f"Background folder does not exist: {background_dir}")

    background_paths = find_background_images(background_dir)
    if not background_paths:
        raise SystemExit(f"No background images found in: {background_dir}")

    rng = random.Random(args.seed)
    generate_all(
        background_paths=background_paths,
        output_dir=output_dir,
        count_per_background=args.count_per_background,
        rng=rng,
        forced_base_size_px=args.base_size,
    )


if __name__ == "__main__":
    main()
