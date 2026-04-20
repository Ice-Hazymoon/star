#!/usr/bin/env python3
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from annotate_types import CropCandidate


def load_wcs(wcs_path: Path) -> WCS:
    with fits.open(wcs_path) as hdul:
        return WCS(hdul[0].header)


def is_point_visible(x: float, y: float, width: int, height: int, margin: float = 0.0) -> bool:
    return -margin <= x <= width + margin and -margin <= y <= height + margin


def is_point_inside_crop(x: float, y: float, crop: CropCandidate, margin: float = 0.0) -> bool:
    return (
        crop.x - margin <= x <= crop.x + crop.width + margin
        and crop.y - margin <= y <= crop.y + crop.height + margin
    )


def crop_bounds(crop: CropCandidate, margin: float = 0.0) -> tuple[float, float, float, float]:
    return (
        crop.x - margin,
        crop.x + crop.width + margin,
        crop.y - margin,
        crop.y + crop.height + margin,
    )


def project_points(wcs: WCS, ra_values: np.ndarray, dec_values: np.ndarray, crop: CropCandidate) -> tuple[np.ndarray, np.ndarray]:
    if len(ra_values) == 0:
        return np.asarray(ra_values, dtype=np.float64), np.asarray(dec_values, dtype=np.float64)
    # all_world2pix applies the SIP / distortion polynomial that solve-field writes
    # by default; wcs_world2pix would only do the linear TAN step and miss tens of
    # pixels of correction near the edge of wide-angle frames.
    #
    # The iterative SIP solver inside astropy calls np.nanmax on its residual
    # array, which warns "All-NaN slice encountered" when every input point falls
    # outside the projection (common: we batch-project whole catalogs that
    # include stars far from the field center). Downstream code already filters
    # with math.isfinite, so silence that specific warning here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
        x_values, y_values = wcs.all_world2pix(ra_values, dec_values, 0, quiet=True)
    return x_values + crop.x, y_values + crop.y


def compute_field_metrics(wcs: WCS, crop: CropCandidate) -> dict[str, Any]:
    width = crop.width
    height = crop.height

    center_ra, center_dec = wcs.all_pix2world(width / 2.0, height / 2.0, 0)
    left_ra, left_dec = wcs.all_pix2world(0.0, height / 2.0, 0)
    right_ra, right_dec = wcs.all_pix2world(width - 1.0, height / 2.0, 0)
    top_ra, top_dec = wcs.all_pix2world(width / 2.0, 0.0, 0)
    bottom_ra, bottom_dec = wcs.all_pix2world(width / 2.0, height - 1.0, 0)

    center = SkyCoord(center_ra, center_dec, unit="deg")
    left = SkyCoord(left_ra, left_dec, unit="deg")
    right = SkyCoord(right_ra, right_dec, unit="deg")
    top = SkyCoord(top_ra, top_dec, unit="deg")
    bottom = SkyCoord(bottom_ra, bottom_dec, unit="deg")

    return {
        "center_ra_deg": float(center_ra),
        "center_dec_deg": float(center_dec),
        "field_width_deg": float(left.separation(right).deg),
        "field_height_deg": float(top.separation(bottom).deg),
        "crop": {
            "name": crop.name,
            "x": crop.x,
            "y": crop.y,
            "width": crop.width,
            "height": crop.height,
        },
    }


def compute_field_center_and_radius(wcs: WCS, crop: CropCandidate) -> tuple[SkyCoord, float]:
    width = crop.width
    height = crop.height
    sample_pixels = np.array(
        [
            (width / 2.0, height / 2.0),
            (0.0, 0.0),
            (width - 1.0, 0.0),
            (0.0, height - 1.0),
            (width - 1.0, height - 1.0),
            (0.0, height / 2.0),
            (width - 1.0, height / 2.0),
            (width / 2.0, 0.0),
            (width / 2.0, height - 1.0),
        ],
        dtype=np.float64,
    )
    ra_values, dec_values = wcs.all_pix2world(sample_pixels[:, 0], sample_pixels[:, 1], 0)
    coords = SkyCoord(ra_values, dec_values, unit="deg")
    center = coords[0]
    edge_separations = center.separation(coords[1:]).deg
    radius = float(np.nanmax(edge_separations)) if len(edge_separations) else 0.0
    return center, radius


def skycoord_separation_degrees(center: SkyCoord, ra_values: np.ndarray, dec_values: np.ndarray) -> np.ndarray:
    if len(ra_values) == 0:
        return np.asarray(ra_values, dtype=np.float64)
    coords = SkyCoord(ra_values, dec_values, unit="deg")
    return center.separation(coords).deg


def compute_out_code(x: float, y: float, min_x: float, max_x: float, min_y: float, max_y: float) -> int:
    code = 0
    if x < min_x:
        code |= 1
    elif x > max_x:
        code |= 2
    if y < min_y:
        code |= 4
    elif y > max_y:
        code |= 8
    return code


def segment_intersects_rect(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    width: int,
    height: int,
    margin: float = 0.0,
) -> bool:
    min_x = -margin
    max_x = width + margin
    min_y = -margin
    max_y = height + margin

    x1, y1 = start_x, start_y
    x2, y2 = end_x, end_y
    out_code_1 = compute_out_code(x1, y1, min_x, max_x, min_y, max_y)
    out_code_2 = compute_out_code(x2, y2, min_x, max_x, min_y, max_y)

    while True:
        if not (out_code_1 | out_code_2):
            return True
        if out_code_1 & out_code_2:
            return False

        out_code_out = out_code_1 or out_code_2
        if out_code_out & 8:
            if y2 == y1:
                return False
            x_value = x1 + (x2 - x1) * (max_y - y1) / (y2 - y1)
            y_value = max_y
        elif out_code_out & 4:
            if y2 == y1:
                return False
            x_value = x1 + (x2 - x1) * (min_y - y1) / (y2 - y1)
            y_value = min_y
        elif out_code_out & 2:
            if x2 == x1:
                return False
            y_value = y1 + (y2 - y1) * (max_x - x1) / (x2 - x1)
            x_value = max_x
        else:
            if x2 == x1:
                return False
            y_value = y1 + (y2 - y1) * (min_x - x1) / (x2 - x1)
            x_value = min_x

        if out_code_out == out_code_1:
            x1, y1 = x_value, y_value
            out_code_1 = compute_out_code(x1, y1, min_x, max_x, min_y, max_y)
        else:
            x2, y2 = x_value, y_value
            out_code_2 = compute_out_code(x2, y2, min_x, max_x, min_y, max_y)


def segment_intersects_crop(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    crop: CropCandidate,
    margin: float = 0.0,
) -> bool:
    min_x, max_x, min_y, max_y = crop_bounds(crop, margin=margin)
    return segment_intersects_rect_with_bounds(start_x, start_y, end_x, end_y, min_x, max_x, min_y, max_y)


def segment_intersects_rect_with_bounds(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> bool:
    x1, y1 = start_x, start_y
    x2, y2 = end_x, end_y
    out_code_1 = compute_out_code(x1, y1, min_x, max_x, min_y, max_y)
    out_code_2 = compute_out_code(x2, y2, min_x, max_x, min_y, max_y)

    while True:
        if not (out_code_1 | out_code_2):
            return True
        if out_code_1 & out_code_2:
            return False

        out_code_out = out_code_1 or out_code_2
        if out_code_out & 8:
            if y2 == y1:
                return False
            x_value = x1 + (x2 - x1) * (max_y - y1) / (y2 - y1)
            y_value = max_y
        elif out_code_out & 4:
            if y2 == y1:
                return False
            x_value = x1 + (x2 - x1) * (min_y - y1) / (y2 - y1)
            y_value = min_y
        elif out_code_out & 2:
            if x2 == x1:
                return False
            y_value = y1 + (y2 - y1) * (max_x - x1) / (x2 - x1)
            x_value = max_x
        else:
            if x2 == x1:
                return False
            y_value = y1 + (y2 - y1) * (min_x - x1) / (x2 - x1)
            x_value = min_x

        if out_code_out == out_code_1:
            x1, y1 = x_value, y_value
            out_code_1 = compute_out_code(x1, y1, min_x, max_x, min_y, max_y)
        else:
            x2, y2 = x_value, y_value
            out_code_2 = compute_out_code(x2, y2, min_x, max_x, min_y, max_y)


def clip_segment_to_bounds(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> tuple[float, float, float, float] | None:
    x1, y1 = start_x, start_y
    x2, y2 = end_x, end_y
    out_code_1 = compute_out_code(x1, y1, min_x, max_x, min_y, max_y)
    out_code_2 = compute_out_code(x2, y2, min_x, max_x, min_y, max_y)

    while True:
        if not (out_code_1 | out_code_2):
            return x1, y1, x2, y2
        if out_code_1 & out_code_2:
            return None

        out_code_out = out_code_1 or out_code_2
        if out_code_out & 8:
            if y2 == y1:
                return None
            x_value = x1 + (x2 - x1) * (max_y - y1) / (y2 - y1)
            y_value = max_y
        elif out_code_out & 4:
            if y2 == y1:
                return None
            x_value = x1 + (x2 - x1) * (min_y - y1) / (y2 - y1)
            y_value = min_y
        elif out_code_out & 2:
            if x2 == x1:
                return None
            y_value = y1 + (y2 - y1) * (max_x - x1) / (x2 - x1)
            x_value = max_x
        else:
            if x2 == x1:
                return None
            y_value = y1 + (y2 - y1) * (min_x - x1) / (x2 - x1)
            x_value = min_x

        if out_code_out == out_code_1:
            x1, y1 = x_value, y_value
            out_code_1 = compute_out_code(x1, y1, min_x, max_x, min_y, max_y)
        else:
            x2, y2 = x_value, y_value
            out_code_2 = compute_out_code(x2, y2, min_x, max_x, min_y, max_y)


def build_segment_key(
    start_ra: float,
    start_dec: float,
    end_ra: float,
    end_dec: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    first = (round(start_ra, 2), round(start_dec, 2))
    second = (round(end_ra, 2), round(end_dec, 2))
    return tuple(sorted((first, second)))


def point_distance_squared(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax - bx) ** 2 + (ay - by) ** 2


def is_projected_segment_duplicate(
    segments: list[dict[str, Any]],
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    tolerance: float,
) -> bool:
    tolerance_squared = tolerance**2
    for segment in segments:
        existing_start = segment["start"]
        existing_end = segment["end"]
        direct_match = (
            point_distance_squared(start_x, start_y, existing_start["x"], existing_start["y"]) <= tolerance_squared
            and point_distance_squared(end_x, end_y, existing_end["x"], existing_end["y"]) <= tolerance_squared
        )
        reverse_match = (
            point_distance_squared(start_x, start_y, existing_end["x"], existing_end["y"]) <= tolerance_squared
            and point_distance_squared(end_x, end_y, existing_start["x"], existing_start["y"]) <= tolerance_squared
        )
        if direct_match or reverse_match:
            return True
    return False
