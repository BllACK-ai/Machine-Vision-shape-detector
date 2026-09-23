"""
Detect and classify cardboard geometric shapes in static drone images.

Usage:
    python shape_detector.py
    python shape_detector.py input --output output
    python shape_detector.py input --method adaptive
    python shape_detector.py input --method canny --min-area 2500
    python shape_detector.py input --method hsv --hsv-target cardboard
    python shape_detector.py input --method hsv --hsv-target white

The default method is "auto", which compares adaptive thresholding and Canny
edge detection and uses the one that produces the cleaner contour set.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# -----------------------------
# Tunable project parameters
# -----------------------------

BLUR_KERNEL_SIZE = 9
ADAPTIVE_BLOCK_SIZE = 41
ADAPTIVE_C = 5
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
MORPH_OPEN_KERNEL_SIZE = 5
MIN_CONTOUR_AREA = 1500.0
APPROX_EPSILON_FACTOR = 0.03

# Four-sided shapes are treated as squares when their bounding box is near 1:1.
# The tolerance is intentionally broad because drone images may have mild tilt.
SQUARE_ASPECT_TOLERANCE = 0.22

# High circularity is close to 1.0 for a perfect circle.
CIRCLE_MIN_CIRCULARITY = 0.78
CIRCLE_ASPECT_TOLERANCE = 0.20
CIRCLE_MIN_ENCLOSING_FILL = 0.96
ELLIPSE_MIN_CIRCULARITY = 0.45
ELLIPSE_MIN_ASPECT_RATIO = 1.20
HIGH_VERTEX_ROUND_MIN_CIRCULARITY = ELLIPSE_MIN_CIRCULARITY

# Shape validity gates. These run before a contour is accepted as a known shape.
SHAPE_CONFIDENCE_THRESHOLD = 0.75
CONVEXITY_MIN_RATIO = 0.90
SIDE_LENGTH_CV_MAX = 0.25
ANGLE_CV_MAX = 0.20
AREA_DEVIATION_MAX = 0.30

# Optional HSV fallback. Pick a named target with --hsv-target, or override the
# exact bounds with --hsv-lower/--hsv-upper when a real cutout color needs tuning.
HSV_CARDBOARD_LOWER = (5, 45, 45)
HSV_CARDBOARD_UPPER = (35, 255, 255)
HSV_WHITE_LOWER = (0, 0, 180)
HSV_WHITE_UPPER = (179, 40, 255)
HSV_TARGETS = {
    "cardboard": (HSV_CARDBOARD_LOWER, HSV_CARDBOARD_UPPER),
    "white": (HSV_WHITE_LOWER, HSV_WHITE_UPPER),
}
DEFAULT_HSV_TARGET = "cardboard"
HSV_LOWER = HSV_CARDBOARD_LOWER
HSV_UPPER = HSV_CARDBOARD_UPPER

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 1.0
LABEL_THICKNESS = 2
ANNOTATION_BASE_LONG_EDGE = 1000.0
ANNOTATION_MIN_SCALE = 0.75
ANNOTATION_MAX_SCALE = 3.0
LABEL_MIN_SCALE = 0.75
LABEL_MAX_SCALE = 2.6
LABEL_MIN_THICKNESS = 2
LABEL_MAX_THICKNESS = 8
LABEL_BOX_PADDING_X = 8
LABEL_BOX_PADDING_Y = 6
ANNOTATION_MIN_THICKNESS = 1
ANNOTATION_MAX_THICKNESS = 8
CONTOUR_COLOR = (0, 255, 0)
LABEL_COLOR = (255, 255, 255)
LABEL_BACKGROUND_COLOR = (32, 32, 32)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


VERTEX_SHAPES = {
    3: "Triangle",
    5: "Pentagon",
    6: "Hexagon",
    7: "Heptagon",
    8: "Octagon",
    9: "Nonagon",
    10: "Decagon",
}


@dataclass(frozen=True)
class ShapeConfidence:
    score: float
    passed: bool
    reasons: tuple[str, ...]
    convexity_ratio: float
    side_length_cv: float | None
    angle_cv: float | None
    area_deviation: float | None


@dataclass(frozen=True)
class Detection:
    shape: str
    vertices: int
    area: float
    bbox: tuple[int, int, int, int]
    contour: np.ndarray
    approx: np.ndarray
    circularity: float
    confidence: float


def ensure_odd(value: int, minimum: int = 3) -> int:
    """Return an odd integer accepted by Gaussian/adaptive threshold kernels."""
    value = max(value, minimum)
    return value if value % 2 == 1 else value + 1


def parse_hsv_triplet(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV values must be formatted as H,S,V")

    try:
        hsv = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HSV values must be integers") from exc

    if not all(0 <= item <= 255 for item in hsv):
        raise argparse.ArgumentTypeError("HSV values must be between 0 and 255")

    return hsv  # type: ignore[return-value]


def resolve_hsv_bounds(
    hsv_target: str,
    hsv_lower: tuple[int, int, int] | None,
    hsv_upper: tuple[int, int, int] | None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    target_lower, target_upper = HSV_TARGETS[hsv_target]
    return hsv_lower or target_lower, hsv_upper or target_upper


def preprocess_image(
    image: np.ndarray,
    method: str,
    blur_kernel_size: int = BLUR_KERNEL_SIZE,
    adaptive_block_size: int = ADAPTIVE_BLOCK_SIZE,
    adaptive_c: int = ADAPTIVE_C,
    canny_low: int = CANNY_LOW_THRESHOLD,
    canny_high: int = CANNY_HIGH_THRESHOLD,
    hsv_lower: tuple[int, int, int] = HSV_LOWER,
    hsv_upper: tuple[int, int, int] = HSV_UPPER,
    morph_kernel_size: int = MORPH_OPEN_KERNEL_SIZE,
) -> np.ndarray:
    """Create a binary image suitable for contour extraction."""
    blur_kernel_size = ensure_odd(blur_kernel_size)
    adaptive_block_size = ensure_odd(adaptive_block_size)
    morph_kernel_size = max(1, morph_kernel_size)
    morph_kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    if method == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            adaptive_block_size,
            adaptive_c,
        )
        # Opening removes small texture specks before contour extraction.
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)

    if method == "canny":
        edges = cv2.Canny(blurred, canny_low, canny_high)
        # Closing joins small gaps in shape outlines caused by texture/shadows.
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morph_kernel)

    raise ValueError(f"Unknown preprocessing method: {method}")


def get_contours(binary: np.ndarray, min_area: float) -> list[np.ndarray]:
    """Find external contours and remove small noise components."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) >= min_area]


def contour_score(contours: Iterable[np.ndarray]) -> float:
    """Score contour quality for automatic method selection."""
    score = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = max(w * h, 1)
        fill_ratio = area / bbox_area
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        # Prefer solid, compact contours. Penalize tiny fragmented edge pieces.
        score += area * max(0.15, min(fill_ratio, 1.0)) * max(0.20, min(circularity, 1.0))
    return score


def choose_preprocessing_method(
    image: np.ndarray,
    methods: tuple[str, ...],
    min_area: float,
    **preprocess_kwargs: object,
) -> tuple[str, np.ndarray, list[np.ndarray]]:
    """Pick the thresholding method with the best contour quality score."""
    best_method = methods[0]
    best_binary = preprocess_image(image, best_method, **preprocess_kwargs)
    best_contours = get_contours(best_binary, min_area)
    best_score = contour_score(best_contours)

    for method in methods[1:]:
        binary = preprocess_image(image, method, **preprocess_kwargs)
        contours = get_contours(binary, min_area)
        score = contour_score(contours)
        logging.info(
            "method=%s contours=%d score=%.2f",
            method,
            len(contours),
            score,
        )

        if score > best_score:
            best_method = method
            best_binary = binary
            best_contours = contours
            best_score = score

    logging.info(
        "selected_method=%s contours=%d score=%.2f",
        best_method,
        len(best_contours),
        best_score,
    )
    return best_method, best_binary, best_contours


def coefficient_of_variation(values: np.ndarray) -> float:
    """Return stddev / mean for positive measurements."""
    if values.size == 0:
        return float("inf")

    mean = float(np.mean(values))
    if mean <= 0:
        return float("inf")

    return float(np.std(values) / mean)


def contour_convexity_ratio(contour: np.ndarray, area: float) -> float:
    """Compare contour area with the area of its convex hull."""
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return 0.0

    return float(area / hull_area)


def polygon_points(approx: np.ndarray) -> np.ndarray:
    """Flatten approxPolyDP output to an Nx2 float array."""
    return approx.reshape(-1, 2).astype(np.float64)


def polygon_side_lengths(points: np.ndarray) -> np.ndarray:
    next_points = np.roll(points, -1, axis=0)
    return np.linalg.norm(next_points - points, axis=1)


def polygon_interior_angles(points: np.ndarray) -> np.ndarray:
    angles: list[float] = []

    for index, point in enumerate(points):
        previous_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        vector_a = previous_point - point
        vector_b = next_point - point

        norm_a = float(np.linalg.norm(vector_a))
        norm_b = float(np.linalg.norm(vector_b))
        if norm_a <= 0 or norm_b <= 0:
            angles.append(0.0)
            continue

        cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))

    return np.array(angles, dtype=np.float64)


def regular_polygon_area_from_perimeter(vertices: int, perimeter: float) -> float:
    if vertices < 3 or perimeter <= 0:
        return 0.0

    return float((perimeter * perimeter) / (4.0 * vertices * np.tan(np.pi / vertices)))


def relative_deviation(actual: float, expected: float) -> float:
    if expected <= 0:
        return float("inf")

    return float(abs(actual - expected) / expected)


def score_minimum(value: float, minimum: float) -> float:
    """Score metrics where larger is better and the minimum is the pass line."""
    if minimum <= 0 or not np.isfinite(value):
        return 0.0

    if value >= minimum:
        if minimum < 1.0:
            headroom = max(1.0 - minimum, 1e-9)
            return float(0.75 + 0.25 * np.clip((value - minimum) / headroom, 0.0, 1.0))
        return 1.0

    return float(0.75 * np.clip(value / minimum, 0.0, 1.0))


def score_maximum(value: float, maximum: float) -> float:
    """Score metrics where smaller is better and the maximum is the pass line."""
    if maximum <= 0 or not np.isfinite(value):
        return 0.0

    if value <= maximum:
        return float(0.75 + 0.25 * np.clip(1.0 - (value / maximum), 0.0, 1.0))

    overshoot = (value - maximum) / maximum
    return float(0.75 * np.clip(1.0 - overshoot, 0.0, 1.0))


def round_shape_geometry(
    contour: np.ndarray,
    circularity: float,
    circle_min_circularity: float,
    circle_aspect_tolerance: float,
) -> tuple[str, float, float]:
    """Return the round candidate and the aspect metrics used to judge it."""
    x, y, width, height = cv2.boundingRect(contour)
    bbox_aspect = width / float(height) if height else 0.0

    ellipse_aspect = bbox_aspect
    if len(contour) >= 5:
        (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
        short_axis = max(min(major_axis, minor_axis), 1.0)
        ellipse_aspect = max(major_axis, minor_axis) / short_axis

    near_circle = (
        circularity >= circle_min_circularity
        and abs(bbox_aspect - 1.0) <= circle_aspect_tolerance
        and abs(ellipse_aspect - 1.0) <= circle_aspect_tolerance
    )
    if near_circle:
        return "Circle", bbox_aspect, ellipse_aspect

    elongated = (
        ellipse_aspect >= ELLIPSE_MIN_ASPECT_RATIO
        or abs(bbox_aspect - 1.0) > circle_aspect_tolerance
    )
    if elongated and circularity >= ELLIPSE_MIN_CIRCULARITY:
        return "Ellipse", bbox_aspect, ellipse_aspect

    return "Unknown", bbox_aspect, ellipse_aspect


def rectangle_side_length_cv(points: np.ndarray) -> float:
    """Score rectangles by opposite-side agreement, not all-side equality."""
    side_lengths = polygon_side_lengths(points)
    if side_lengths.size != 4:
        return float("inf")

    first_pair_mean = float(np.mean(side_lengths[[0, 2]]))
    second_pair_mean = float(np.mean(side_lengths[[1, 3]]))
    first_pair_deviation = relative_deviation(float(side_lengths[0]), first_pair_mean)
    second_pair_deviation = relative_deviation(float(side_lengths[1]), second_pair_mean)

    return max(first_pair_deviation, second_pair_deviation)


def calculate_shape_confidence(
    contour: np.ndarray,
    approx: np.ndarray,
    candidate_shape: str,
    circularity: float,
    *,
    circle_min_circularity: float = CIRCLE_MIN_CIRCULARITY,
    circle_aspect_tolerance: float = CIRCLE_ASPECT_TOLERANCE,
    convexity_min_ratio: float = CONVEXITY_MIN_RATIO,
    side_length_cv_max: float = SIDE_LENGTH_CV_MAX,
    angle_cv_max: float = ANGLE_CV_MAX,
    area_deviation_max: float = AREA_DEVIATION_MAX,
    confidence_threshold: float = SHAPE_CONFIDENCE_THRESHOLD,
) -> ShapeConfidence:
    """Calculate shape regularity before accepting a final classification."""
    area = float(cv2.contourArea(contour))
    convexity_ratio = contour_convexity_ratio(contour, area)
    reasons: list[str] = []
    scores = [score_minimum(convexity_ratio, convexity_min_ratio)]
    side_length_cv: float | None = None
    angle_cv: float | None = None
    area_deviation: float | None = None

    if convexity_ratio < convexity_min_ratio:
        reasons.append(f"convexity {convexity_ratio:.3f} < {convexity_min_ratio:.3f}")

    if candidate_shape in {"Circle", "Ellipse"}:
        round_shape, bbox_aspect, ellipse_aspect = round_shape_geometry(
            contour,
            circularity,
            circle_min_circularity,
            circle_aspect_tolerance,
        )
        if round_shape != candidate_shape:
            reasons.append(f"round geometry looks like {round_shape}")

        if candidate_shape == "Circle":
            circle_fill = min_enclosing_circle_fill(contour, area)
            bbox_score = score_maximum(abs(bbox_aspect - 1.0), circle_aspect_tolerance)
            ellipse_score = score_maximum(abs(ellipse_aspect - 1.0), circle_aspect_tolerance)
            scores.extend(
                [
                    score_minimum(circularity, circle_min_circularity),
                    bbox_score,
                    ellipse_score,
                    score_minimum(circle_fill, CIRCLE_MIN_ENCLOSING_FILL),
                ]
            )
            if circle_fill < CIRCLE_MIN_ENCLOSING_FILL:
                reasons.append(
                    f"circle fill {circle_fill:.3f} < {CIRCLE_MIN_ENCLOSING_FILL:.3f}"
                )
        else:
            elongated = (
                ellipse_aspect >= ELLIPSE_MIN_ASPECT_RATIO
                or abs(bbox_aspect - 1.0) > circle_aspect_tolerance
            )
            scores.extend(
                [
                    score_minimum(circularity, ELLIPSE_MIN_CIRCULARITY),
                    score_minimum(ellipse_aspect, ELLIPSE_MIN_ASPECT_RATIO),
                ]
            )
            if not elongated:
                reasons.append("ellipse is not elongated enough")

    else:
        points = polygon_points(approx)
        vertices = len(points)
        angle_cv = coefficient_of_variation(polygon_interior_angles(points))
        scores.append(score_maximum(angle_cv, angle_cv_max))

        if candidate_shape == "Rectangle":
            side_length_cv = rectangle_side_length_cv(points)
            (_, _), (width, height), _ = cv2.minAreaRect(approx)
            expected_area = float(max(width, 1.0) * max(height, 1.0))
        else:
            side_length_cv = coefficient_of_variation(polygon_side_lengths(points))
            expected_area = regular_polygon_area_from_perimeter(
                vertices,
                cv2.arcLength(contour, True),
            )

        area_deviation = relative_deviation(area, expected_area)
        scores.extend(
            [
                score_maximum(side_length_cv, side_length_cv_max),
                score_maximum(area_deviation, area_deviation_max),
            ]
        )

        if side_length_cv > side_length_cv_max:
            reasons.append(
                f"side length variation {side_length_cv:.3f} > {side_length_cv_max:.3f}"
            )
        if angle_cv > angle_cv_max:
            reasons.append(f"angle variation {angle_cv:.3f} > {angle_cv_max:.3f}")
        if area_deviation > area_deviation_max:
            reasons.append(
                f"area deviation {area_deviation:.3f} > {area_deviation_max:.3f}"
            )

    score = float(np.mean(scores)) if scores else 0.0
    if score < confidence_threshold:
        reasons.append(f"confidence {score:.3f} < {confidence_threshold:.3f}")

    return ShapeConfidence(
        score=score,
        passed=not reasons,
        reasons=tuple(reasons),
        convexity_ratio=convexity_ratio,
        side_length_cv=side_length_cv,
        angle_cv=angle_cv,
        area_deviation=area_deviation,
    )


def classify_shape(
    contour: np.ndarray,
    epsilon_factor: float = APPROX_EPSILON_FACTOR,
    square_tolerance: float = SQUARE_ASPECT_TOLERANCE,
    circle_min_circularity: float = CIRCLE_MIN_CIRCULARITY,
    circle_aspect_tolerance: float = CIRCLE_ASPECT_TOLERANCE,
    high_vertex_round_min_circularity: float = HIGH_VERTEX_ROUND_MIN_CIRCULARITY,
    confidence_threshold: float = SHAPE_CONFIDENCE_THRESHOLD,
    convexity_min_ratio: float = CONVEXITY_MIN_RATIO,
    side_length_cv_max: float = SIDE_LENGTH_CV_MAX,
    angle_cv_max: float = ANGLE_CV_MAX,
    area_deviation_max: float = AREA_DEVIATION_MAX,
) -> tuple[str | None, np.ndarray, float, ShapeConfidence]:
    """Classify a contour only after it passes geometric confidence checks."""
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
    fine_approx = cv2.approxPolyDP(contour, min(epsilon_factor, 0.02) * perimeter, True)
    vertices = len(approx)
    fine_vertices = len(fine_approx)

    circularity = 0.0
    if perimeter > 0:
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))

    def validate(candidate_shape: str, candidate_approx: np.ndarray) -> ShapeConfidence:
        return calculate_shape_confidence(
            contour,
            candidate_approx,
            candidate_shape,
            circularity,
            circle_min_circularity=circle_min_circularity,
            circle_aspect_tolerance=circle_aspect_tolerance,
            convexity_min_ratio=convexity_min_ratio,
            side_length_cv_max=side_length_cv_max,
            angle_cv_max=angle_cv_max,
            area_deviation_max=area_deviation_max,
            confidence_threshold=confidence_threshold,
        )

    if vertices == 4:
        (_, _), (width, height), _ = cv2.minAreaRect(approx)
        short_side = max(min(width, height), 1.0)
        aspect_ratio = max(width, height) / short_side

        # A square has a near-equal width and height, but the tolerance allows
        # mild perspective skew from a drone that is not perfectly nadir.
        if abs(aspect_ratio - 1.0) <= square_tolerance:
            confidence = validate("Square", approx)
            return ("Square" if confidence.passed else None), approx, circularity, confidence

        confidence = validate("Rectangle", approx)
        return ("Rectangle" if confidence.passed else None), approx, circularity, confidence

    if vertices in VERTEX_SHAPES and vertices <= 6:
        shape = VERTEX_SHAPES[vertices]
        confidence = validate(shape, approx)
        return (shape if confidence.passed else None), approx, circularity, confidence

    round_shape = classify_round_shape(
        contour,
        circularity,
        circle_min_circularity,
        circle_aspect_tolerance,
    )
    circle_fill = min_enclosing_circle_fill(contour, area)

    if round_shape == "Ellipse":
        confidence = validate(round_shape, approx)
        return (round_shape if confidence.passed else None), approx, circularity, confidence

    if round_shape == "Circle" and circle_fill >= CIRCLE_MIN_ENCLOSING_FILL:
        confidence = validate(round_shape, approx)
        return (round_shape if confidence.passed else None), approx, circularity, confidence

    if (
        fine_vertices in VERTEX_SHAPES
        and fine_vertices >= 7
        and circularity >= high_vertex_round_min_circularity
    ):
        shape = VERTEX_SHAPES[fine_vertices]
        confidence = validate(shape, fine_approx)
        return (shape if confidence.passed else None), fine_approx, circularity, confidence

    if vertices in VERTEX_SHAPES:
        shape = VERTEX_SHAPES[vertices]
        confidence = validate(shape, approx)
        return (shape if confidence.passed else None), approx, circularity, confidence

    confidence = calculate_shape_confidence(
        contour,
        approx,
        "Unknown",
        circularity,
        circle_min_circularity=circle_min_circularity,
        circle_aspect_tolerance=circle_aspect_tolerance,
        confidence_threshold=1.0,
    )
    return None, approx, circularity, confidence


def min_enclosing_circle_fill(contour: np.ndarray, area: float) -> float:
    """Measure how much of the minimum enclosing circle is occupied."""
    _, radius = cv2.minEnclosingCircle(contour)
    enclosing_area = np.pi * radius * radius
    if enclosing_area <= 0:
        return 0.0
    return float(area / enclosing_area)


def classify_round_shape(
    contour: np.ndarray,
    circularity: float,
    circle_min_circularity: float,
    circle_aspect_tolerance: float,
) -> str:
    """Separate circles from ellipses when polygon vertex counts are unclear."""
    shape, _, _ = round_shape_geometry(
        contour,
        circularity,
        circle_min_circularity,
        circle_aspect_tolerance,
    )
    return shape


def detect_shapes(
    image: np.ndarray,
    contours: Iterable[np.ndarray],
    epsilon_factor: float,
    square_tolerance: float,
    circle_min_circularity: float,
    circle_aspect_tolerance: float,
    confidence_threshold: float,
    convexity_min_ratio: float,
    side_length_cv_max: float,
    angle_cv_max: float,
    area_deviation_max: float,
) -> list[Detection]:
    detections: list[Detection] = []

    for contour in contours:
        shape, approx, circularity, confidence = classify_shape(
            contour,
            epsilon_factor=epsilon_factor,
            square_tolerance=square_tolerance,
            circle_min_circularity=circle_min_circularity,
            circle_aspect_tolerance=circle_aspect_tolerance,
            confidence_threshold=confidence_threshold,
            convexity_min_ratio=convexity_min_ratio,
            side_length_cv_max=side_length_cv_max,
            angle_cv_max=angle_cv_max,
            area_deviation_max=area_deviation_max,
        )
        area = float(cv2.contourArea(contour))
        bbox = cv2.boundingRect(contour)

        if shape is None:
            logging.debug(
                "rejected contour area=%.1f vertices=%d confidence=%.3f reasons=%s",
                area,
                len(approx),
                confidence.score,
                "; ".join(confidence.reasons),
            )
            continue

        detections.append(
            Detection(
                shape=shape,
                vertices=len(approx),
                area=area,
                bbox=bbox,
                contour=contour,
                approx=approx,
                circularity=circularity,
                confidence=confidence.score,
            )
        )

    # Report larger objects first so the main cardboard cutouts are easy to see.
    return sorted(detections, key=lambda detection: detection.area, reverse=True)


def annotate_image(image: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    annotated = image.copy()
    image_height, image_width = annotated.shape[:2]
    image_long_edge = max(image_width, image_height)
    annotation_scale = float(
        np.clip(
            image_long_edge / ANNOTATION_BASE_LONG_EDGE,
            ANNOTATION_MIN_SCALE,
            ANNOTATION_MAX_SCALE,
        )
    )
    label_scale = float(
        np.clip(
            LABEL_SCALE * annotation_scale,
            LABEL_MIN_SCALE,
            LABEL_MAX_SCALE,
        )
    )
    line_thickness = int(
        np.clip(
            round(LABEL_THICKNESS * annotation_scale),
            ANNOTATION_MIN_THICKNESS,
            ANNOTATION_MAX_THICKNESS,
        )
    )
    label_thickness = int(
        np.clip(
            round(LABEL_THICKNESS * annotation_scale),
            LABEL_MIN_THICKNESS,
            LABEL_MAX_THICKNESS,
        )
    )
    label_padding_x = max(4, int(round(LABEL_BOX_PADDING_X * annotation_scale)))
    label_padding_y = max(4, int(round(LABEL_BOX_PADDING_Y * annotation_scale)))

    for detection in detections:
        cv2.drawContours(
            annotated, [detection.contour], -1, CONTOUR_COLOR, line_thickness
        )
        cv2.drawContours(
            annotated, [detection.approx], -1, (255, 0, 0), line_thickness
        )

        x, y, width, height = detection.bbox
        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            (0, 180, 255),
            line_thickness,
        )

        label = (
            f"{detection.shape} "
            f"v={detection.vertices} "
            f"c={detection.circularity:.2f} "
            f"q={detection.confidence:.2f}"
        )
        (label_width, label_height), label_baseline = cv2.getTextSize(
            label, LABEL_FONT, label_scale, label_thickness
        )
        label_box_width = label_width + (2 * label_padding_x)
        label_box_height = label_height + label_baseline + (2 * label_padding_y)
        box_x = min(max(x, 0), max(image_width - label_box_width - 1, 0))
        above_y = y - label_box_height
        below_y = y + height
        if above_y >= 0:
            box_y = above_y
        elif below_y + label_box_height < image_height:
            box_y = below_y
        else:
            box_y = min(
                max(0, y),
                max(image_height - label_box_height - 1, 0),
            )
        label_x = box_x + label_padding_x
        label_y = box_y + label_padding_y + label_height

        cv2.rectangle(
            annotated,
            (box_x, box_y),
            (
                min(box_x + label_box_width, image_width - 1),
                min(box_y + label_box_height, image_height - 1),
            ),
            LABEL_BACKGROUND_COLOR,
            cv2.FILLED,
        )

        cv2.putText(
            annotated,
            label,
            (label_x, label_y),
            LABEL_FONT,
            label_scale,
            LABEL_COLOR,
            label_thickness,
            cv2.LINE_AA,
        )

    return annotated


def log_detections(detections: Iterable[Detection]) -> None:
    for index, detection in enumerate(detections, start=1):
        x, y, width, height = detection.bbox
        logging.info(
            "#%d shape=%s vertices=%d area=%.1f bbox=(x=%d, y=%d, w=%d, h=%d) circularity=%.3f confidence=%.3f",
            index,
            detection.shape,
            detection.vertices,
            detection.area,
            x,
            y,
            width,
            height,
            detection.circularity,
            detection.confidence,
        )


def iter_image_paths(input_path: Path) -> list[Path]:
    """Return supported image files from a folder, or a single image path."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_SUFFIXES else []

    if not input_path.is_dir():
        return []

    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def process_image(
    image_path: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[Detection], str] | None:
    """Load one image and run the existing detection/classification pipeline."""
    image = cv2.imread(str(image_path))
    if image is None:
        logging.warning("Could not load image: %s", image_path)
        return None

    hsv_lower, hsv_upper = resolve_hsv_bounds(
        args.hsv_target,
        args.hsv_lower,
        args.hsv_upper,
    )
    preprocess_kwargs = {
        "blur_kernel_size": args.blur_kernel,
        "morph_kernel_size": args.morph_kernel,
        "hsv_lower": hsv_lower,
        "hsv_upper": hsv_upper,
    }

    if args.method == "auto":
        selected_method, binary, contours = choose_preprocessing_method(
            image,
            ("adaptive", "canny"),
            args.min_area,
            **preprocess_kwargs,
        )
    else:
        selected_method = args.method
        binary = preprocess_image(image, selected_method, **preprocess_kwargs)
        contours = get_contours(binary, args.min_area)
        logging.info("selected_method=%s contours=%d", selected_method, len(contours))

    detections = detect_shapes(
        image,
        contours,
        epsilon_factor=args.epsilon,
        square_tolerance=args.square_tolerance,
        circle_min_circularity=args.circle_circularity,
        circle_aspect_tolerance=args.circle_aspect_tolerance,
        confidence_threshold=args.shape_confidence,
        convexity_min_ratio=args.convexity_ratio,
        side_length_cv_max=args.side_length_cv,
        angle_cv_max=args.angle_cv,
        area_deviation_max=args.area_deviation,
    )

    return image, binary, detections, selected_method


def write_results_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    """Append one CSV row per detected shape across the whole batch."""
    fieldnames = [
        "filename",
        "detected_shape",
        "vertices",
        "area",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "circularity",
        "confidence",
    ]
    should_write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerows(rows)


def unique_output_path(output_path: Path) -> Path:
    """Return output_path, or append _2, _3, etc. if that file already exists."""
    if not output_path.exists():
        return output_path

    for suffix in range(2, 10_000):
        candidate = output_path.with_name(
            f"{output_path.stem}_{suffix}{output_path.suffix}"
        )
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an available output filename for {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and classify geometric cardboard cutouts in drone images."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("input"),
        help="Input image file or folder path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Folder where annotated images and results.csv are saved.",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "adaptive", "canny", "hsv"),
        default="auto",
        help="Preprocessing method. 'auto' compares adaptive and Canny.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=MIN_CONTOUR_AREA,
        help="Minimum contour area used to ignore debris and shadows.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=APPROX_EPSILON_FACTOR,
        help="approxPolyDP epsilon as a fraction of contour perimeter.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=BLUR_KERNEL_SIZE,
        help="Gaussian blur kernel size. Even values are rounded up.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=MORPH_OPEN_KERNEL_SIZE,
        help="Morphological cleanup kernel size used after thresholding.",
    )
    parser.add_argument(
        "--square-tolerance",
        type=float,
        default=SQUARE_ASPECT_TOLERANCE,
        help="Allowed distance from 1:1 aspect ratio for square classification.",
    )
    parser.add_argument(
        "--circle-circularity",
        type=float,
        default=CIRCLE_MIN_CIRCULARITY,
        help="Minimum circularity needed for circle classification.",
    )
    parser.add_argument(
        "--circle-aspect-tolerance",
        type=float,
        default=CIRCLE_ASPECT_TOLERANCE,
        help="Allowed distance from 1:1 aspect ratio for circle classification.",
    )
    parser.add_argument(
        "--shape-confidence",
        type=float,
        default=SHAPE_CONFIDENCE_THRESHOLD,
        help="Minimum geometric confidence needed before a contour is labeled.",
    )
    parser.add_argument(
        "--convexity-ratio",
        type=float,
        default=CONVEXITY_MIN_RATIO,
        help="Minimum contour area divided by convex hull area.",
    )
    parser.add_argument(
        "--side-length-cv",
        type=float,
        default=SIDE_LENGTH_CV_MAX,
        help="Maximum side-length coefficient of variation for polygon validity.",
    )
    parser.add_argument(
        "--angle-cv",
        type=float,
        default=ANGLE_CV_MAX,
        help="Maximum interior-angle coefficient of variation for polygon validity.",
    )
    parser.add_argument(
        "--area-deviation",
        type=float,
        default=AREA_DEVIATION_MAX,
        help="Maximum relative deviation from expected geometric area.",
    )
    parser.add_argument(
        "--hsv-target",
        choices=tuple(HSV_TARGETS),
        default=DEFAULT_HSV_TARGET,
        help="Named HSV target preset for --method hsv.",
    )
    parser.add_argument(
        "--hsv-lower",
        type=parse_hsv_triplet,
        default=None,
        help="Manual lower HSV color threshold as H,S,V for --method hsv.",
    )
    parser.add_argument(
        "--hsv-upper",
        type=parse_hsv_triplet,
        default=None,
        help="Manual upper HSV color threshold as H,S,V for --method hsv.",
    )
    parser.add_argument(
        "--debug-mask",
        type=Path,
        help="Optional folder to save the binary mask for each processed image.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log debug details, including rejected contour reasons.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    image_paths = iter_image_paths(args.input)
    if not image_paths:
        logging.error("No supported image files found in: %s", args.input)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    if args.debug_mask:
        args.debug_mask.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        logging.info("processing=%s", image_path)
        result = process_image(image_path, args)
        if result is None:
            continue

        image, binary, detections, selected_method = result
        log_detections(detections)

        annotated = annotate_image(image, detections)
        annotated_path = unique_output_path(args.output / image_path.name)
        cv2.imwrite(str(annotated_path), annotated)
        logging.info("saved_output=%s", annotated_path)

        if args.debug_mask:
            debug_mask_path = unique_output_path(args.debug_mask / image_path.name)
            cv2.imwrite(str(debug_mask_path), binary)
            logging.info("saved_debug_mask=%s", debug_mask_path)

        for detection in detections:
            x, y, width, height = detection.bbox
            rows.append(
                {
                    "filename": image_path.name,
                    "detected_shape": detection.shape,
                    "vertices": detection.vertices,
                    "area": f"{detection.area:.1f}",
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_width": width,
                    "bbox_height": height,
                    "circularity": f"{detection.circularity:.3f}",
                    "confidence": f"{detection.confidence:.3f}",
                }
            )

        logging.info(
            "finished=%s selected_method=%s detections=%d",
            image_path.name,
            selected_method,
            len(detections),
        )

    results_path = args.output / "results.csv"
    write_results_csv(results_path, rows)
    logging.info("saved_results=%s rows=%d", results_path, len(rows))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
