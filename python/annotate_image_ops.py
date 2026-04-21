#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.wcs import FITSFixedWarning
from PIL import Image, ImageOps

try:
    import sep
except ImportError:  # pragma: no cover - optional dependency for source extraction
    sep = None

from annotate_types import CropCandidate, SourceAnalysis, SourceDetection


warnings.filterwarnings("ignore", category=FITSFixedWarning)

# Resource caps against adversarial inputs. A malicious image with tens of
# thousands of tiny blobs can make sep.extract burn minutes of CPU; a
# decompression-bomb PNG can allocate hundreds of MB of luma before we even
# reach source extraction. These bounds keep the worker's worst case roughly
# linear with the legitimate use case and let obvious attacks fail fast.
MAX_INPUT_IMAGE_PIXELS = 50_000_000
MAX_INPUT_IMAGE_DIMENSION = 10_000
MAX_DETECTED_SOURCES = 8_000
Image.MAX_IMAGE_PIXELS = MAX_INPUT_IMAGE_PIXELS


def _reject_oversize_image(width: int, height: int) -> None:
    if width > MAX_INPUT_IMAGE_DIMENSION or height > MAX_INPUT_IMAGE_DIMENSION:
        raise RuntimeError(
            f"image {width}x{height} exceeds dimension cap of {MAX_INPUT_IMAGE_DIMENSION}px per side",
        )
    if width * height > MAX_INPUT_IMAGE_PIXELS:
        raise RuntimeError(
            f"image area {width * height} exceeds pixel cap of {MAX_INPUT_IMAGE_PIXELS}",
        )


def normalize_image(input_path: Path, workdir: Path) -> tuple[Image.Image, Path]:
    with Image.open(input_path) as image:
        _reject_oversize_image(image.width, image.height)
        normalized = ImageOps.exif_transpose(image).convert("RGB")
    _reject_oversize_image(normalized.width, normalized.height)
    normalized_path = workdir / "normalized-input.jpg"
    normalized.save(normalized_path, quality=95)
    return normalized, normalized_path


def save_crop(image: Image.Image, crop: CropCandidate, workdir: Path) -> Path:
    if crop.x == 0 and crop.y == 0 and crop.width == image.width and crop.height == image.height:
        return workdir / "normalized-input.jpg"

    crop_path = workdir / f"{crop.name}.jpg"
    image.crop((crop.x, crop.y, crop.x + crop.width, crop.y + crop.height)).save(crop_path, quality=95)
    return crop_path


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def image_to_luma_array(image: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image.convert("L"), dtype=np.float32))


def score_source_candidate(flux: float, peak: float, major: float, minor: float, npix: int) -> float:
    if flux <= 0.0 or peak <= 0.0 or major <= 0.0 or minor <= 0.0 or npix <= 0:
        return 0.0

    size = max(major, minor)
    elongation = max(major, minor) / max(min(major, minor), 1e-3)
    compactness = peak * max(npix, 1) / max(flux, 1.0)

    size_score = clamp_float(1.0 - abs(size - 1.8) / 3.0, 0.0, 1.0)
    area_score = clamp_float(1.0 - max(0.0, npix - 18.0) / 64.0, 0.0, 1.0)
    elongation_score = clamp_float(1.0 - max(0.0, elongation - 1.2) / 2.2, 0.0, 1.0)
    compactness_score = clamp_float((compactness - 0.18) / 0.9, 0.0, 1.0)
    brightness_score = clamp_float(math.log10(max(flux, 1.0)) / 2.8, 0.1, 1.0)

    score = (
        size_score * 0.34
        + area_score * 0.16
        + elongation_score * 0.24
        + compactness_score * 0.16
        + brightness_score * 0.10
    )

    if size > 8.0 or npix > 90 or elongation > 4.5:
        score *= 0.15

    return clamp_float(score, 0.0, 1.0)


def analyze_sources(
    image: Image.Image,
    sky_mask: np.ndarray | None = None,
) -> SourceAnalysis:
    """Extract candidate star sources from the image.

    `sky_mask`, when provided, is a (H, W) uint8 array with 1 = sky / 0 =
    ground. sep detections whose centroid lands on ground pixels are dropped
    before being fed to solve-field. This dramatically cleans up the source
    list on foreground-heavy images (tree silhouettes, city skylines) and
    keeps the full-image solve's RMS competitive with sub-crop solves. The
    mask is applied "optimistically": if it accidentally filters too much
    (bad mask → few detections left), we fall back to the unfiltered list so
    solve still has something to work with.
    """
    if sep is None:
        return SourceAnalysis(
            mode="fallback-no-sep",
            detections=[],
            tile_scores=np.zeros((1, 1), dtype=np.float32),
            diagnostics={
                "mode": "fallback-no-sep",
                "raw_sources": 0,
                "usable_sources": 0,
                "tile_rows": 1,
                "tile_cols": 1,
                "top_tiles": [],
            },
        )

    luma = image_to_luma_array(image)
    background = sep.Background(luma)
    data = np.ascontiguousarray(luma - background.back(), dtype=np.float32)
    threshold = max(float(background.globalrms) * 2.2, 4.0)
    # deblend_nthresh=16 / deblend_cont=0.01 keep enough deblending for dense
    # clusters while making hostile dot-storm images roughly 2x cheaper to
    # process than the sep defaults.
    objects = sep.extract(
        data,
        thresh=threshold,
        err=background.globalrms,
        minarea=3,
        deblend_nthresh=16,
        deblend_cont=0.01,
    )

    # On adversarial inputs sep.extract can return tens of thousands of blobs.
    # We only feed up to ~900 into solve-field anyway, so truncate by flux
    # here before the O(n) Python loop below.
    if len(objects) > MAX_DETECTED_SOURCES:
        flux_order = np.argsort(np.asarray(objects["flux"], dtype=np.float64))[::-1]
        objects = objects[flux_order[:MAX_DETECTED_SOURCES]]

    raw_detections: list[SourceDetection] = []
    usable_detections: list[SourceDetection] = []
    for obj in objects:
        x_value = float(obj["x"])
        y_value = float(obj["y"])
        major = float(max(obj["a"], obj["b"]))
        minor = float(min(obj["a"], obj["b"]))
        npix = int(obj["npix"])
        elongation = major / max(minor, 1e-3)
        flux = float(obj["flux"])
        peak = float(obj["peak"])
        star_score = score_source_candidate(flux, peak, major, minor, npix)
        sort_flux = flux * (0.55 + 0.9 * star_score)
        detection = SourceDetection(
            x=x_value,
            y=y_value,
            flux=flux,
            peak=peak,
            major=major,
            minor=minor,
            npix=npix,
            elongation=elongation,
            star_score=star_score,
            sort_flux=sort_flux,
        )
        raw_detections.append(detection)
        if star_score >= 0.22:
            usable_detections.append(detection)

    mask_filtered_count = 0
    if sky_mask is not None and sky_mask.size > 0 and usable_detections:
        mask_h, mask_w = sky_mask.shape[:2]
        kept: list[SourceDetection] = []
        for detection in usable_detections:
            xi = max(0, min(mask_w - 1, int(round(detection.x))))
            yi = max(0, min(mask_h - 1, int(round(detection.y))))
            if sky_mask[yi, xi]:
                kept.append(detection)
        # Only apply the mask if enough detections survive to keep solve
        # viable — otherwise a bad (hallucinated) mask could starve the
        # solver of real sources. 30 is a conservative floor; astrometry.net
        # typically needs ~20 matches to lock in a solution.
        if len(kept) >= 30 or len(kept) >= len(usable_detections) * 0.5:
            mask_filtered_count = len(usable_detections) - len(kept)
            usable_detections = kept

    tile_cols = max(4, min(12, round(image.width / 180)))
    tile_rows = max(4, min(12, round(image.height / 180)))
    tile_scores = np.zeros((tile_rows, tile_cols), dtype=np.float32)
    tile_star_weights = np.zeros((tile_rows, tile_cols), dtype=np.float32)
    tile_star_counts = np.zeros((tile_rows, tile_cols), dtype=np.int32)
    tile_extended_penalty = np.zeros((tile_rows, tile_cols), dtype=np.float32)

    grad_x = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
    grad_y = np.abs(np.diff(luma, axis=0, prepend=luma[:1, :]))
    gradient = grad_x + grad_y
    edge_norm = max(float(background.globalrms) * 4.0, 12.0)

    for row in range(tile_rows):
        y0 = int(round(row * image.height / tile_rows))
        y1 = int(round((row + 1) * image.height / tile_rows))
        for col in range(tile_cols):
            x0 = int(round(col * image.width / tile_cols))
            x1 = int(round((col + 1) * image.width / tile_cols))
            tile_luma = luma[y0:y1, x0:x1]
            tile_gradient = gradient[y0:y1, x0:x1]
            if tile_luma.size == 0:
                continue
            bright_fraction = float(np.mean(tile_luma > 228.0))
            saturated_fraction = float(np.mean(tile_luma > 247.0))
            edge_density = float(tile_gradient.mean() / edge_norm)
            tile_scores[row, col] = -edge_density * 0.45 - bright_fraction * 3.5 - saturated_fraction * 7.5

    for detection in usable_detections:
        col = min(tile_cols - 1, max(0, int(detection.x / image.width * tile_cols)))
        row = min(tile_rows - 1, max(0, int(detection.y / image.height * tile_rows)))
        tile_star_weights[row, col] += detection.star_score
        tile_star_counts[row, col] += 1
        if detection.npix > 22 or detection.elongation > 1.9:
            tile_extended_penalty[row, col] += 1.0

    tile_scores += tile_star_weights * 1.8
    tile_scores += tile_star_counts.astype(np.float32) * 0.24
    tile_scores -= tile_extended_penalty * 0.9

    top_tiles: list[dict[str, Any]] = []
    for row in range(tile_rows):
        for col in range(tile_cols):
            top_tiles.append(
                {
                    "row": row,
                    "col": col,
                    "score": float(tile_scores[row, col]),
                    "star_weight": float(tile_star_weights[row, col]),
                    "star_count": int(tile_star_counts[row, col]),
                }
            )
    top_tiles.sort(key=lambda item: item["score"], reverse=True)

    return SourceAnalysis(
        mode="sep",
        detections=sorted(usable_detections, key=lambda item: item.sort_flux, reverse=True),
        tile_scores=tile_scores,
        diagnostics={
            "mode": "sep",
            "raw_sources": len(raw_detections),
            "usable_sources": len(usable_detections),
            "mask_filtered": mask_filtered_count,
            "threshold": threshold,
            "background_rms": float(background.globalrms),
            "tile_rows": tile_rows,
            "tile_cols": tile_cols,
            "tile_scores": [[round(float(value), 3) for value in row] for row in tile_scores.tolist()],
            "top_tiles": top_tiles[:8],
        },
    )


def crop_iou(first: CropCandidate, second: CropCandidate) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / max(union, 1)


def sum_integral_region(integral: np.ndarray, row0: int, col0: int, row1: int, col1: int) -> float:
    return float(
        integral[row1, col1]
        - integral[row0, col1]
        - integral[row1, col0]
        + integral[row0, col0]
    )


def build_crop_candidates(width: int, height: int, source_analysis: SourceAnalysis | None = None) -> list[CropCandidate]:
    full_crop = CropCandidate(name="full", x=0, y=0, width=width, height=height)
    if source_analysis is None or source_analysis.tile_scores.size <= 1:
        return [full_crop]

    tile_scores = source_analysis.tile_scores
    tile_rows, tile_cols = tile_scores.shape
    integral = np.pad(tile_scores, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    fractions = (0.78, 0.9, 0.66)
    candidate_windows: list[tuple[float, int, int, int, int, float]] = []
    for fraction in fractions:
        window_rows = max(2, min(tile_rows, int(round(tile_rows * fraction))))
        window_cols = max(2, min(tile_cols, int(round(tile_cols * fraction))))
        scored_windows: list[tuple[float, int, int, int, int, float]] = []
        for row in range(0, tile_rows - window_rows + 1):
            for col in range(0, tile_cols - window_cols + 1):
                total_score = sum_integral_region(integral, row, col, row + window_rows, col + window_cols)
                mean_score = total_score / max(window_rows * window_cols, 1)
                scored_windows.append((total_score, row, col, window_rows, window_cols, mean_score))
        scored_windows.sort(key=lambda item: (item[0], item[5]), reverse=True)

        chosen_windows: list[tuple[float, int, int, int, int, float]] = []
        for window in scored_windows:
            _, row, col, rows, cols, _ = window
            overlaps = False
            for _, chosen_row, chosen_col, chosen_rows, chosen_cols, _ in chosen_windows:
                row_overlap = min(row + rows, chosen_row + chosen_rows) - max(row, chosen_row)
                col_overlap = min(col + cols, chosen_col + chosen_cols) - max(col, chosen_col)
                if row_overlap > 0 and col_overlap > 0:
                    overlap_area = row_overlap * col_overlap
                    if overlap_area / max(rows * cols, chosen_rows * chosen_cols) > 0.45:
                        overlaps = True
                        break
            if overlaps:
                continue
            chosen_windows.append(window)
            if len(chosen_windows) >= 2:
                break
        candidate_windows.extend(chosen_windows)

    crops: list[CropCandidate] = []
    for index, (total_score, row, col, rows, cols, mean_score) in enumerate(candidate_windows):
        if total_score <= 0.0 and mean_score <= 0.0:
            continue
        x0 = int(math.floor(col * width / tile_cols))
        x1 = int(math.ceil((col + cols) * width / tile_cols))
        y0 = int(math.floor(row * height / tile_rows))
        y1 = int(math.ceil((row + rows) * height / tile_rows))

        margin_x = max(18, int(round(width / tile_cols * 0.35)))
        margin_y = max(18, int(round(height / tile_rows * 0.35)))
        crop = CropCandidate(
            name=f"score-{index + 1}",
            x=max(0, x0 - margin_x),
            y=max(0, y0 - margin_y),
            width=min(width, x1 + margin_x) - max(0, x0 - margin_x),
            height=min(height, y1 + margin_y) - max(0, y0 - margin_y),
        )
        if crop.width < max(240, int(width * 0.4)) or crop.height < max(240, int(height * 0.4)):
            continue
        if any(crop_iou(crop, existing) > 0.82 for existing in crops):
            continue
        crops.append(crop)

    ordered = [*crops[:3], full_crop]
    deduped: list[CropCandidate] = []
    for crop in ordered:
        if any(crop_iou(crop, existing) > 0.9 for existing in deduped):
            continue
        deduped.append(crop)
    return deduped or [full_crop]
