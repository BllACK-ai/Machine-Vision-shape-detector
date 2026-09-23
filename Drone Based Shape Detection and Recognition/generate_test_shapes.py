"""Generate one input test image per geometric shape."""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np


CANVAS_SIZE = 500
BACKGROUND_COLOR = (170, 210, 170)  # light green, BGR
SHAPE_COLOR = (65, 95, 130)  # dark cardboard-like brown, BGR
OUTPUT_DIR = Path("input")
RANDOM_SEED = 7


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


def polygon_points(
    center: tuple[int, int], radius: int, sides: int, rotation_degrees: float
) -> np.ndarray:
    """Return evenly spaced polygon vertices around a circle."""
    cx, cy = center
    rotation = math.radians(rotation_degrees) - math.pi / 2
    points = []

    for index in range(sides):
        angle = rotation + (2 * math.pi * index / sides)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((round(x), round(y)))

    return np.array(points, dtype=np.int32)


def paste_with_mask(
    canvas: np.ndarray, patch: np.ndarray, mask: np.ndarray, center: tuple[int, int]
) -> None:
    """Paste a rotated shape patch onto the canvas using a mask."""
    x = center[0] - patch.shape[1] // 2
    y = center[1] - patch.shape[0] // 2
    roi = canvas[y : y + patch.shape[0], x : x + patch.shape[1]]
    roi[mask > 0] = patch[mask > 0]


def draw_rotated_rectangle(
    canvas: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    rotation_degrees: float,
) -> None:
    """Draw a filled rectangle using cv2.rectangle, then rotate it."""
    width, height = size
    patch_size = int(math.ceil(math.hypot(width, height))) + 32
    patch_center = (patch_size // 2, patch_size // 2)
    top_left = (patch_center[0] - width // 2, patch_center[1] - height // 2)
    bottom_right = (patch_center[0] + width // 2, patch_center[1] + height // 2)

    patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    cv2.rectangle(patch, top_left, bottom_right, SHAPE_COLOR, thickness=-1)
    cv2.rectangle(mask, top_left, bottom_right, 255, thickness=-1)

    matrix = cv2.getRotationMatrix2D(patch_center, rotation_degrees, 1.0)
    rotated_patch = cv2.warpAffine(
        patch, matrix, (patch_size, patch_size), flags=cv2.INTER_NEAREST
    )
    rotated_mask = cv2.warpAffine(
        mask, matrix, (patch_size, patch_size), flags=cv2.INTER_NEAREST
    )
    paste_with_mask(canvas, rotated_patch, rotated_mask, center)


def draw_shape(name: str, shape_type: str, sides: int | None, rotation: float) -> np.ndarray:
    """Create a single centered shape image without any ground-truth label."""
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BACKGROUND_COLOR, dtype=np.uint8)
    center = (CANVAS_SIZE // 2, CANVAS_SIZE // 2)

    if shape_type == "polygon" and sides is not None:
        points = polygon_points(center, radius=130, sides=sides, rotation_degrees=rotation)
        cv2.fillPoly(canvas, [points], SHAPE_COLOR)
    elif shape_type == "square":
        draw_rotated_rectangle(canvas, center, size=(230, 230), rotation_degrees=rotation)
    elif shape_type == "rectangle":
        draw_rotated_rectangle(canvas, center, size=(285, 170), rotation_degrees=rotation)
    elif shape_type == "circle":
        cv2.circle(canvas, center, radius=130, color=SHAPE_COLOR, thickness=-1)
    elif shape_type == "ellipse":
        cv2.ellipse(
            canvas,
            center,
            axes=(155, 95),
            angle=rotation,
            startAngle=0,
            endAngle=360,
            color=SHAPE_COLOR,
            thickness=-1,
        )
    else:
        raise ValueError(f"Unknown shape definition for {name}")

    return canvas


def main() -> None:
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, shape_type, sides in SHAPES:
        rotation = random.uniform(0, 359)
        image = draw_shape(name, shape_type, sides, rotation)
        output_path = OUTPUT_DIR / f"{name.lower()}.png"
        cv2.imwrite(str(output_path), image)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
