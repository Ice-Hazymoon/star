#!/usr/bin/env python3
from __future__ import annotations

import math
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from PIL import Image, ImageDraw, ImageFont

from annotate_geometry import (
    build_segment_key,
    clip_segment_to_bounds,
    compute_display_field_center_and_radius,
    compute_field_center_and_radius,
    crop_bounds,
    is_point_inside_crop,
    is_point_visible,
    is_projected_segment_duplicate,
    point_distance_squared,
    project_points,
    segment_intersects_crop,
    skycoord_separation_degrees,
)
from annotate_localization import normalize_lookup_key
from annotate_options import overlay_detail_value, overlay_layer_enabled
from annotate_types import CropCandidate


try:
    LANCZOS_RESAMPLING = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - Pillow compatibility
    LANCZOS_RESAMPLING = Image.LANCZOS


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def clamp_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: float,
    y: float,
    image_width: int,
    image_height: int,
    font: ImageFont.ImageFont,
    stroke_width: int = 2,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[float, float]:
    left, top, right, bottom = draw.textbbox((x, y), text, font=font, stroke_width=stroke_width)
    width = right - left
    height = bottom - top
    if bounds is None:
        min_x = 2.0
        max_x = image_width - width - 2.0
        min_y = 2.0
        max_y = image_height - height - 2.0
    else:
        min_x = bounds[0] + 2.0
        max_x = bounds[1] - width - 2.0
        min_y = bounds[2] + 2.0
        max_y = bounds[3] - height - 2.0
    clamped_x = min(max(min_x, x), max(min_x, max_x))
    clamped_y = min(max(min_y, y), max(min_y, max_y))
    return clamped_x, clamped_y


def boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], padding: float = 6.0) -> bool:
    return not (a[2] + padding < b[0] or b[2] + padding < a[0] or a[3] + padding < b[1] or b[3] + padding < a[1])


def place_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    anchor_x: float,
    anchor_y: float,
    image_width: int,
    image_height: int,
    font: ImageFont.ImageFont,
    occupied_boxes: list[tuple[float, float, float, float]],
    offsets: list[tuple[float, float]],
    stroke_width: int = 2,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[float, float] | None:
    for dx, dy in offsets:
        x_value, y_value = clamp_text(
            draw,
            text,
            anchor_x + dx,
            anchor_y + dy,
            image_width,
            image_height,
            font,
            stroke_width=stroke_width,
            bounds=bounds,
        )
        bbox = draw.textbbox((x_value, y_value), text, font=font, stroke_width=stroke_width)
        normalized_box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if any(boxes_overlap(normalized_box, existing) for existing in occupied_boxes):
            continue
        occupied_boxes.append(normalized_box)
        return x_value, y_value
    return None


def dso_category(item: dict[str, Any]) -> str:
    normalized_type = normalize_lookup_key(item.get("type"))
    if normalized_type in {"ocl", "opencluster"}:
        return "open_cluster"
    if normalized_type in {"gcl", "globularcluster"}:
        return "globular_cluster"
    if normalized_type in {"pn", "planetarynebula"}:
        return "planetary_nebula"
    if normalized_type in {"snr", "supernovaremnant"}:
        return "supernova_remnant"
    if normalized_type in {"neb", "diffusenebula", "emissionnebula", "reflectionnebula", "hii", "cln", "gneb", "rfn"}:
        return "diffuse_nebula"
    if normalized_type in {"g", "galaxy", "spiralgalaxy", "ellipticalgalaxy", "lenticulargalaxy", "irregulargalaxy", "ultrafaintdwarfgalaxy"}:
        return "galaxy"
    return "other"


def dso_style(item: dict[str, Any]) -> tuple[str, tuple[int, int, int, int]]:
    category = dso_category(item)
    styles = {
        "open_cluster": ("square", (145, 228, 255, 235)),
        "globular_cluster": ("crossed_circle", (160, 245, 198, 235)),
        "planetary_nebula": ("ring", (116, 220, 255, 235)),
        "supernova_remnant": ("x_circle", (255, 178, 138, 235)),
        "diffuse_nebula": ("hexagon", (118, 202, 255, 235)),
        "galaxy": ("diamond", (214, 206, 255, 235)),
        "other": ("circle", (140, 235, 255, 230)),
    }
    return styles[category]


def compute_label_leader_segment(
    draw: ImageDraw.ImageDraw,
    anchor_x: float,
    anchor_y: float,
    label_position: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    stroke_width: int = 2,
) -> tuple[float, float, float, float] | None:
    bbox = draw.textbbox(label_position, text, font=font, stroke_width=stroke_width)
    target_x = min(max(anchor_x, bbox[0]), bbox[2])
    target_y = min(max(anchor_y, bbox[1]), bbox[3])
    if point_distance_squared(anchor_x, anchor_y, target_x, target_y) < 24.0**2:
        return None
    return float(anchor_x), float(anchor_y), float(target_x), float(target_y)


def overlay_supersample_scale(width: int, height: int) -> int:
    pixels = width * height
    if pixels <= 4_000_000:
        return 3
    if pixels <= 12_000_000:
        return 2
    return 1


def scale_crop_candidate(crop: CropCandidate, scale: int) -> CropCandidate:
    return CropCandidate(
        name=crop.name,
        x=int(round(crop.x * scale)),
        y=int(round(crop.y * scale)),
        width=int(round(crop.width * scale)),
        height=int(round(crop.height * scale)),
    )


def scale_positioned_overlay_items(items: list[dict[str, Any]], scale: int) -> list[dict[str, Any]]:
    scaled_items = deepcopy(items)
    for item in scaled_items:
        if "x" in item:
            item["x"] = float(item["x"]) * scale
        if "y" in item:
            item["y"] = float(item["y"]) * scale
    return scaled_items


def scale_constellation_overlays(constellations: list[dict[str, Any]], scale: int) -> list[dict[str, Any]]:
    scaled_constellations = deepcopy(constellations)
    for constellation in scaled_constellations:
        constellation["label_x"] = float(constellation["label_x"]) * scale
        constellation["label_y"] = float(constellation["label_y"]) * scale
        for segment in constellation["segments"]:
            for endpoint in ("start", "end"):
                segment[endpoint]["x"] = float(segment[endpoint]["x"]) * scale
                segment[endpoint]["y"] = float(segment[endpoint]["y"]) * scale
    return scaled_constellations


def rgba_to_list(color: tuple[int, int, int, int]) -> list[int]:
    return [int(value) for value in color]


def clip_constellation_segment_to_crop(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    crop: CropCandidate,
) -> tuple[float, float, float, float] | None:
    min_x, max_x, min_y, max_y = crop_bounds(crop)
    clipped_segment = clip_segment_to_bounds(
        start_x,
        start_y,
        end_x,
        end_y,
        min_x,
        max_x,
        min_y,
        max_y,
    )
    if clipped_segment is None:
        return None
    if point_distance_squared(*clipped_segment) < 1.0:
        return None
    return clipped_segment


def constellation_segment_min_separation_degrees(
    field_center: Any,
    start_ra_degrees: float,
    start_dec_degrees: float,
    end_ra_degrees: float,
    end_dec_degrees: float,
) -> float:
    endpoint_separations = skycoord_separation_degrees(
        field_center,
        np.array([start_ra_degrees, end_ra_degrees]),
        np.array([start_dec_degrees, end_dec_degrees]),
    )
    return float(np.min(endpoint_separations))


def collect_named_stars(
    catalog: pd.DataFrame,
    star_names: dict[int, str],
    wcs: WCS,
    crop: CropCandidate,
    image_width: int,
    image_height: int,
    overlay_options: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_hips = [hip for hip in star_names if hip in catalog.index]
    if not candidate_hips:
        return []
    # Guard radius covers the full image — when a sub-crop wins the solve we
    # still want to surface objects anywhere in the visible frame, not just
    # inside the solved rectangle.
    field_center, field_radius_degrees = compute_display_field_center_and_radius(
        wcs, crop, image_width, image_height
    )
    star_guard_radius_degrees = field_radius_degrees + 6.0

    magnitude_limit = float(overlay_detail_value(overlay_options, "star_magnitude_limit"))
    max_labels = int(overlay_detail_value(overlay_options, "star_label_limit"))
    bright_star_separation = float(overlay_detail_value(overlay_options, "star_bright_separation"))
    dim_star_separation = float(overlay_detail_value(overlay_options, "star_dim_separation"))

    subset = catalog.loc[candidate_hips].copy()
    subset = subset[subset["magnitude"] <= magnitude_limit].sort_values("magnitude")
    separations = skycoord_separation_degrees(
        field_center,
        subset["ra_degrees"].to_numpy(),
        subset["dec_degrees"].to_numpy(),
    )
    subset = subset.iloc[separations <= star_guard_radius_degrees]
    if subset.empty:
        return []
    x_values, y_values = project_points(
        wcs,
        subset["ra_degrees"].to_numpy(),
        subset["dec_degrees"].to_numpy(),
        crop,
    )

    visible: list[dict[str, Any]] = []
    for (hip, row), x_value, y_value in zip(subset.iterrows(), x_values, y_values, strict=True):
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        if not is_point_visible(float(x_value), float(y_value), image_width, image_height, margin=12.0):
            continue
        visible.append(
            {
                "hip": int(hip),
                "name": star_names[int(hip)],
                "magnitude": float(row["magnitude"]),
                "x": float(x_value),
                "y": float(y_value),
            }
        )

    selected: list[dict[str, Any]] = []
    anchors: list[tuple[float, float]] = []
    for star in visible:
        if len(selected) >= max_labels:
            break
        separation = bright_star_separation if star["magnitude"] <= 2.0 else dim_star_separation
        if any((star["x"] - px) ** 2 + (star["y"] - py) ** 2 < separation**2 for px, py in anchors):
            continue
        anchors.append((star["x"], star["y"]))
        selected.append(star)
    return selected


def collect_constellations(
    catalog: pd.DataFrame,
    constellations: list[dict[str, Any]],
    wcs: WCS,
    crop: CropCandidate,
    image_width: int,
    image_height: int,
    overlay_options: dict[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    duplicate_tolerance = max(6.0, min(image_width, image_height) / 180.0)
    # `display_crop` spans the full image so visibility / clipping covers the
    # whole frame even when a sub-crop won the solve. `crop` is kept as-is for
    # project_points (where its x/y offset un-offsets the crop-local WCS).
    display_crop = CropCandidate(
        name="display", x=0, y=0, width=image_width, height=image_height
    )
    min_x, max_x, min_y, max_y = crop_bounds(display_crop)
    field_center, field_radius_degrees = compute_display_field_center_and_radius(
        wcs, crop, image_width, image_height
    )
    segment_guard_radius_degrees = field_radius_degrees + 12.0
    label_guard_radius_degrees = field_radius_degrees + 8.0

    # Every constellation segment endpoint is a HIP star in `catalog`. Project
    # the whole catalog once with a single astropy call and build HIP-keyed
    # lookup tables — the inner loops then do dict access instead of calling
    # all_world2pix / SkyCoord.separation per segment.
    catalog_hips = catalog.index.to_numpy()
    catalog_ra = catalog["ra_degrees"].to_numpy()
    catalog_dec = catalog["dec_degrees"].to_numpy()
    catalog_xs, catalog_ys = project_points(wcs, catalog_ra, catalog_dec, crop)
    catalog_seps = skycoord_separation_degrees(field_center, catalog_ra, catalog_dec)
    catalog_ra_by_hip: dict[int, float] = {}
    catalog_dec_by_hip: dict[int, float] = {}
    catalog_xy_by_hip: dict[int, tuple[float, float]] = {}
    catalog_sep_by_hip: dict[int, float] = {}
    for hip, ra, dec, x, y, sep in zip(
        catalog_hips.tolist(),
        catalog_ra.tolist(),
        catalog_dec.tolist(),
        catalog_xs.tolist(),
        catalog_ys.tolist(),
        catalog_seps.tolist(),
        strict=True,
    ):
        hip_int = int(hip)
        catalog_ra_by_hip[hip_int] = float(ra)
        catalog_dec_by_hip[hip_int] = float(dec)
        catalog_xy_by_hip[hip_int] = (float(x), float(y))
        catalog_sep_by_hip[hip_int] = float(sep)

    # Batch-project the explicit label anchors across all constellations (most
    # have label_ra/dec_degrees set). The recognition loop below only uses the
    # lookup for constellations that actually make it to the label phase, so
    # projecting the full set up front is cheaper than per-constellation calls.
    label_abbrs: list[str] = []
    label_ra_list: list[float] = []
    label_dec_list: list[float] = []
    for item in constellations:
        if item.get("label_ra_degrees") is None or item.get("label_dec_degrees") is None:
            continue
        label_abbrs.append(item["abbr"])
        label_ra_list.append(float(item["label_ra_degrees"]))
        label_dec_list.append(float(item["label_dec_degrees"]))
    label_xy_by_abbr: dict[str, tuple[float, float]] = {}
    label_sep_by_abbr: dict[str, float] = {}
    if label_abbrs:
        lbl_ra = np.asarray(label_ra_list, dtype=np.float64)
        lbl_dec = np.asarray(label_dec_list, dtype=np.float64)
        lbl_seps = skycoord_separation_degrees(field_center, lbl_ra, lbl_dec)
        lbl_xs, lbl_ys = project_points(wcs, lbl_ra, lbl_dec, crop)
        for abbr, x, y, sep in zip(
            label_abbrs, lbl_xs.tolist(), lbl_ys.tolist(), lbl_seps.tolist(), strict=True
        ):
            label_xy_by_abbr[abbr] = (float(x), float(y))
            label_sep_by_abbr[abbr] = float(sep)

    for constellation in constellations:
        visible_segments: list[dict[str, Any]] = []
        label_points: list[tuple[float, float]] = []
        onscreen_segments = 0
        segment_keys: set[tuple[tuple[float, float], tuple[float, float]]] = set()
        recognized = False

        for polyline in constellation.get("lines", []):
            for start_hip, end_hip in pairwise(polyline):
                start_xy = catalog_xy_by_hip.get(start_hip)
                end_xy = catalog_xy_by_hip.get(end_hip)
                if start_xy is None or end_xy is None:
                    continue
                if min(catalog_sep_by_hip[start_hip], catalog_sep_by_hip[end_hip]) > segment_guard_radius_degrees:
                    continue
                start_x, start_y = start_xy
                end_x, end_y = end_xy
                if not all(math.isfinite(value) for value in (start_x, start_y, end_x, end_y)):
                    continue
                if not segment_intersects_crop(start_x, start_y, end_x, end_y, display_crop, margin=36.0):
                    continue
                recognized = True
                break
            if recognized:
                break

        if not recognized:
            continue

        for polyline in constellation.get("lines", []):
            for start_hip, end_hip in pairwise(polyline):
                start_xy = catalog_xy_by_hip.get(start_hip)
                end_xy = catalog_xy_by_hip.get(end_hip)
                if start_xy is None or end_xy is None:
                    continue
                segment_key = build_segment_key(
                    catalog_ra_by_hip[start_hip],
                    catalog_dec_by_hip[start_hip],
                    catalog_ra_by_hip[end_hip],
                    catalog_dec_by_hip[end_hip],
                )
                if segment_key in segment_keys:
                    continue
                start_x, start_y = start_xy
                end_x, end_y = end_xy
                if not all(math.isfinite(value) for value in (start_x, start_y, end_x, end_y)):
                    continue
                if not segment_intersects_crop(start_x, start_y, end_x, end_y, display_crop, margin=36.0):
                    continue
                clipped_segment = clip_constellation_segment_to_crop(start_x, start_y, end_x, end_y, display_crop)
                if clipped_segment is None:
                    continue
                clipped_start_x, clipped_start_y, clipped_end_x, clipped_end_y = clipped_segment
                if is_projected_segment_duplicate(
                    visible_segments,
                    clipped_start_x,
                    clipped_start_y,
                    clipped_end_x,
                    clipped_end_y,
                    duplicate_tolerance,
                ):
                    continue

                segment_keys.add(segment_key)
                onscreen_segments += 1

                start_payload: dict[str, Any] = {"x": clipped_start_x, "y": clipped_start_y}
                end_payload: dict[str, Any] = {"x": clipped_end_x, "y": clipped_end_y}
                if point_distance_squared(clipped_start_x, clipped_start_y, start_x, start_y) < 1.0:
                    start_payload["hip"] = int(start_hip)
                if point_distance_squared(clipped_end_x, clipped_end_y, end_x, end_y) < 1.0:
                    end_payload["hip"] = int(end_hip)

                visible_segments.append({"start": start_payload, "end": end_payload})
                label_points.append((clipped_start_x, clipped_start_y))
                label_points.append((clipped_end_x, clipped_end_y))

        if not visible_segments:
            continue
        if len(label_points) < 2 and onscreen_segments == 0:
            continue

        label_xy = label_xy_by_abbr.get(constellation["abbr"])
        if label_xy is not None:
            explicit_label_x, explicit_label_y = label_xy
            label_separation = label_sep_by_abbr[constellation["abbr"]]
            if (
                label_separation <= label_guard_radius_degrees
                and math.isfinite(explicit_label_x)
                and math.isfinite(explicit_label_y)
                and is_point_visible(
                explicit_label_x,
                explicit_label_y,
                image_width,
                image_height,
                margin=48.0,
                )
            ):
                if is_point_inside_crop(explicit_label_x, explicit_label_y, display_crop, margin=48.0):
                    label_x = explicit_label_x
                    label_y = explicit_label_y
                elif label_points:
                    label_x = sum(point[0] for point in label_points) / len(label_points)
                    label_y = sum(point[1] for point in label_points) / len(label_points)
                else:
                    first_segment = visible_segments[0]
                    label_x = (first_segment["start"]["x"] + first_segment["end"]["x"]) / 2.0
                    label_y = (first_segment["start"]["y"] + first_segment["end"]["y"]) / 2.0
            elif label_points:
                label_x = sum(point[0] for point in label_points) / len(label_points)
                label_y = sum(point[1] for point in label_points) / len(label_points)
            else:
                first_segment = visible_segments[0]
                label_x = (first_segment["start"]["x"] + first_segment["end"]["x"]) / 2.0
                label_y = (first_segment["start"]["y"] + first_segment["end"]["y"]) / 2.0
        elif label_points:
            label_x = sum(point[0] for point in label_points) / len(label_points)
            label_y = sum(point[1] for point in label_points) / len(label_points)
        else:
            first_segment = visible_segments[0]
            label_x = (first_segment["start"]["x"] + first_segment["end"]["x"]) / 2.0
            label_y = (first_segment["start"]["y"] + first_segment["end"]["y"]) / 2.0

        label_x = min(max(float(label_x), min_x), max_x)
        label_y = min(max(float(label_y), min_y), max_y)

        result.append(
            {
                "abbr": constellation["abbr"],
                "english_name": constellation["english_name"],
                "native_name": constellation["native_name"],
                "display_name": constellation["display_name"],
                "label_x": float(label_x),
                "label_y": float(label_y),
                "segments": visible_segments,
                "show_label": False,
            }
        )

    if bool(overlay_detail_value(overlay_options, "show_all_constellation_labels")):
        for constellation in result:
            constellation["show_label"] = True
    else:
        anchors: list[tuple[float, float]] = []
        for constellation in sorted(result, key=lambda item: len(item["segments"]), reverse=True):
            if any((constellation["label_x"] - px) ** 2 + (constellation["label_y"] - py) ** 2 < 72.0**2 for px, py in anchors):
                continue
            anchors.append((constellation["label_x"], constellation["label_y"]))
            constellation["show_label"] = True

    return sorted(result, key=lambda item: len(item["segments"]), reverse=True)


def is_interesting_dso(item: dict[str, Any], overlay_options: dict[str, Any]) -> bool:
    object_type = item["type"]
    if object_type.startswith("*") or object_type in {"Dup", "NonEx", "Other"}:
        return False
    if item["common_name"] in {"Alnilam", "Orion B", "Gem A", "Browning", "Flame Nebula", "Great Bird Cluster", "Lower Sword", "Upper Sword"}:
        return False
    if item["messier"] or item["common_name"]:
        return True
    if normalize_lookup_key(item.get("label")) not in {
        normalize_lookup_key(item.get("name")),
        normalize_lookup_key(item.get("catalog_id")),
    }:
        return True
    if bool(overlay_detail_value(overlay_options, "include_catalog_dsos")) and item.get("catalog_id"):
        return True
    return False


def dso_importance(item: dict[str, Any]) -> float:
    score = 0.0
    if item.get("curated"):
        score += 4.0
    if item["messier"]:
        score += 8.0
    if item["common_name"]:
        score += 5.0
    if normalize_lookup_key(item.get("label")) not in {
        normalize_lookup_key(item.get("name")),
        normalize_lookup_key(item.get("messier")),
        normalize_lookup_key(item.get("common_name")),
        normalize_lookup_key(item.get("catalog_id")),
    }:
        score += 2.5
    if item["major_axis_arcmin"] is not None:
        score += min(item["major_axis_arcmin"], 180.0) / 30.0
    if item["magnitude"] is not None:
        score += max(0.0, 10.5 - item["magnitude"]) / 2.5
    return score


def compose_dso_display_label(item: dict[str, Any]) -> str:
    base_label = item.get("label") or item.get("name") or ""
    messier = item.get("messier")
    catalog_id = item.get("catalog_id")

    if messier and normalize_lookup_key(base_label) != normalize_lookup_key(messier):
        return f"{messier} {base_label}"
    if messier:
        return messier
    if catalog_id and catalog_id.upper().startswith(("NGC", "IC")) and normalize_lookup_key(base_label) != normalize_lookup_key(catalog_id):
        return f"{catalog_id} {base_label}"
    return base_label


def collect_deep_sky_objects(
    deep_sky_objects: list[dict[str, Any]],
    wcs: WCS,
    crop: CropCandidate,
    image_width: int,
    image_height: int,
    overlay_options: dict[str, Any],
) -> list[dict[str, Any]]:
    dso_magnitude_limit = float(overlay_detail_value(overlay_options, "dso_magnitude_limit"))
    dso_label_limit = int(overlay_detail_value(overlay_options, "dso_label_limit"))
    dso_spacing_scale = float(overlay_detail_value(overlay_options, "dso_spacing_scale"))
    detailed_labels = bool(overlay_detail_value(overlay_options, "detailed_dso_labels"))
    field_center, field_radius_degrees = compute_display_field_center_and_radius(
        wcs, crop, image_width, image_height
    )
    dso_guard_radius_degrees = field_radius_degrees + 6.0

    # Phase 1 — cheap Python prefilter (same predicates as before, just lifted
    # out of the per-object astropy loop).
    prefiltered: list[dict[str, Any]] = []
    for item in deep_sky_objects:
        if not is_interesting_dso(item, overlay_options):
            continue
        if item["magnitude"] is not None and item["magnitude"] > dso_magnitude_limit and not item["messier"] and not item["common_name"]:
            continue
        prefiltered.append(item)

    candidates: list[dict[str, Any]] = []
    if prefiltered:
        # Phase 2 — batch separation check against the field center.
        pre_ra = np.fromiter(
            (float(item["ra_degrees"]) for item in prefiltered),
            dtype=np.float64,
            count=len(prefiltered),
        )
        pre_dec = np.fromiter(
            (float(item["dec_degrees"]) for item in prefiltered),
            dtype=np.float64,
            count=len(prefiltered),
        )
        separations = skycoord_separation_degrees(field_center, pre_ra, pre_dec)
        within_guard = separations <= dso_guard_radius_degrees

        shortlist_ra = pre_ra[within_guard]
        shortlist_dec = pre_dec[within_guard]
        shortlist_items = [
            prefiltered[i] for i in range(len(prefiltered)) if within_guard[i]
        ]

        if shortlist_items:
            # Phase 3 — one astropy all_world2pix call for everything that made it.
            xs, ys = project_points(wcs, shortlist_ra, shortlist_dec, crop)

            # Phase 4 — per-item finite + visibility filter; cheap Python.
            for item, x_raw, y_raw in zip(shortlist_items, xs, ys, strict=True):
                x_value = float(x_raw)
                y_value = float(y_raw)
                if not (math.isfinite(x_value) and math.isfinite(y_value)):
                    continue
                if not is_point_visible(x_value, y_value, image_width, image_height, margin=28.0):
                    continue
                candidate = dict(item)
                candidate.update(
                    {
                        "x": x_value,
                        "y": y_value,
                        "display_label": compose_dso_display_label(item) if detailed_labels else (item.get("label") or item.get("name") or ""),
                    }
                )
                candidates.append(candidate)

    selected: list[dict[str, Any]] = []
    anchors: list[tuple[float, float]] = []
    labels: list[tuple[str, float, float]] = []
    for item in sorted(
        candidates,
        key=lambda entry: (-dso_importance(entry), entry["magnitude"] if entry["magnitude"] is not None else 99.0, -(entry["major_axis_arcmin"] or 0.0)),
    ):
        if len(selected) >= dso_label_limit:
            break
        spacing = max(34.0, min(104.0, (item["major_axis_arcmin"] or 28.0) * dso_spacing_scale))
        if any((item["x"] - px) ** 2 + (item["y"] - py) ** 2 < spacing**2 for px, py in anchors):
            continue
        if any(label == item["display_label"] and (item["x"] - px) ** 2 + (item["y"] - py) ** 2 < 180.0**2 for label, px, py in labels):
            continue
        anchors.append((item["x"], item["y"]))
        labels.append((item["display_label"], item["x"], item["y"]))
        selected.append(item)
    return selected


def add_contextual_constellation_labels(
    constellations: list[dict[str, Any]],
    deep_sky_objects: list[dict[str, Any]],
    constellation_catalog: dict[str, dict[str, Any]],
    overlay_options: dict[str, Any],
) -> list[dict[str, Any]]:
    if not overlay_layer_enabled(overlay_options, "contextual_constellation_labels"):
        return constellations
    known_abbrs = {item["abbr"] for item in constellations}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in deep_sky_objects:
        abbr = (item.get("const") or "").strip()
        if not abbr or abbr in known_abbrs or abbr not in constellation_catalog:
            continue
        grouped.setdefault(abbr, []).append(item)

    extras: list[dict[str, Any]] = []
    for abbr, items in grouped.items():
        reference = constellation_catalog[abbr]
        label_x = sum(item["x"] for item in items) / len(items)
        label_y = sum(item["y"] for item in items) / len(items)
        extras.append(
            {
                "abbr": abbr,
                "english_name": reference["english_name"],
                "native_name": reference["native_name"],
                "display_name": reference["display_name"],
                "label_x": float(label_x),
                "label_y": float(label_y),
                "segments": [],
                "show_label": True,
            }
        )
    return [*constellations, *extras]


def build_overlay_scene(
    image_size: tuple[int, int],
    constellations: list[dict[str, Any]],
    named_stars: list[dict[str, Any]],
    deep_sky_objects: list[dict[str, Any]],
    crop: CropCandidate,
    overlay_options: dict[str, Any],
) -> dict[str, Any]:
    image_width, image_height = image_size
    layout_surface = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layout_surface)
    occupied_boxes: list[tuple[float, float, float, float]] = []

    min_dimension = min(image_width, image_height)
    line_width = max(1, min_dimension // 600)
    render_bounds = crop_bounds(crop)

    constellation_font_size = max(18, min_dimension // 52)
    dso_font_size = max(14, min_dimension // 74)
    star_font_size = max(12, min_dimension // 84)
    constellation_font = load_font(constellation_font_size)
    dso_font = load_font(dso_font_size)
    star_font = load_font(star_font_size)
    dso_radius = max(4, min_dimension // 250)
    star_radius = max(2, min_dimension // 320)

    show_dso_markers = overlay_layer_enabled(overlay_options, "deep_sky_markers")
    show_dso_labels = overlay_layer_enabled(overlay_options, "deep_sky_labels")
    show_constellation_labels = overlay_layer_enabled(overlay_options, "constellation_labels")
    show_contextual_labels = overlay_layer_enabled(overlay_options, "contextual_constellation_labels")
    show_star_markers = overlay_layer_enabled(overlay_options, "star_markers")
    show_star_labels = overlay_layer_enabled(overlay_options, "star_labels")
    show_label_leaders = overlay_layer_enabled(overlay_options, "label_leaders")

    scene: dict[str, Any] = {
        "image_width": image_width,
        "image_height": image_height,
        "crop": {
            "name": crop.name,
            "x": crop.x,
            "y": crop.y,
            "width": crop.width,
            "height": crop.height,
        },
        "bounds": {
            "left": float(render_bounds[0]),
            "top": float(render_bounds[2]),
            "right": float(render_bounds[1]),
            "bottom": float(render_bounds[3]),
        },
        "constellation_lines": [],
        "constellation_labels": [],
        "deep_sky_markers": [],
        "deep_sky_labels": [],
        "star_markers": [],
        "star_labels": [],
    }

    if overlay_layer_enabled(overlay_options, "constellation_lines"):
        for constellation in constellations:
            line_color = (212, 222, 236, 135 if constellation["show_label"] else 92)
            for segment in constellation["segments"]:
                clipped_segment = clip_segment_to_bounds(
                    segment["start"]["x"],
                    segment["start"]["y"],
                    segment["end"]["x"],
                    segment["end"]["y"],
                    render_bounds[0],
                    render_bounds[1],
                    render_bounds[2],
                    render_bounds[3],
                )
                if clipped_segment is None:
                    continue
                if point_distance_squared(*clipped_segment) < 1.0:
                    continue
                scene["constellation_lines"].append(
                    {
                        "x1": float(clipped_segment[0]),
                        "y1": float(clipped_segment[1]),
                        "x2": float(clipped_segment[2]),
                        "y2": float(clipped_segment[3]),
                        "line_width": line_width,
                        "rgba": rgba_to_list(line_color),
                    }
                )

    for item in deep_sky_objects:
        if show_dso_markers:
            marker, marker_color = dso_style(item)
            scene["deep_sky_markers"].append(
                {
                    "marker": marker,
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "radius": dso_radius,
                    "line_width": line_width,
                    "rgba": rgba_to_list(marker_color),
                }
            )
        if not show_dso_labels:
            continue
        position = place_label(
            draw,
            item["display_label"],
            item["x"],
            item["y"],
            image_width,
            image_height,
            dso_font,
            occupied_boxes,
            offsets=[
                (10.0, -26.0),
                (10.0, 10.0),
                (-112.0, -26.0),
                (-112.0, 10.0),
                (14.0, -42.0),
                (14.0, 24.0),
                (-128.0, -42.0),
                (-128.0, 24.0),
                (8.0, -22.0),
                (8.0, 8.0),
                (-86.0, -22.0),
                (-86.0, 8.0),
            ],
            stroke_width=2,
            bounds=render_bounds,
        )
        if not position:
            continue
        leader_segment = None
        if show_label_leaders:
            leader_segment = compute_label_leader_segment(
                draw,
                item["x"],
                item["y"],
                position,
                item["display_label"],
                dso_font,
                stroke_width=2,
            )
        scene["deep_sky_labels"].append(
            {
                "text": item["display_label"],
                "x": float(position[0]),
                "y": float(position[1]),
                "font_size": dso_font_size,
                "stroke_width": 2,
                "text_rgba": rgba_to_list((242, 246, 255, 255)),
                "stroke_rgba": rgba_to_list((0, 0, 0, 220)),
                "leader": (
                    {
                        "x1": leader_segment[0],
                        "y1": leader_segment[1],
                        "x2": leader_segment[2],
                        "y2": leader_segment[3],
                        "line_width": 1,
                        "rgba": rgba_to_list((165, 220, 255, 190)),
                    }
                    if leader_segment is not None
                    else None
                ),
            }
        )

    if show_constellation_labels:
        for constellation in constellations:
            if not constellation["show_label"]:
                continue
            if not constellation["segments"] and not show_contextual_labels:
                continue
            position = place_label(
                draw,
                constellation["display_name"],
                constellation["label_x"],
                constellation["label_y"],
                image_width,
                image_height,
                constellation_font,
                occupied_boxes,
                offsets=[
                    (10.0, 10.0),
                    (10.0, -34.0),
                    (-56.0, 10.0),
                    (-56.0, -34.0),
                    (12.0, 28.0),
                    (-74.0, 28.0),
                ],
                stroke_width=3,
                bounds=render_bounds,
            )
            if not position:
                continue
            scene["constellation_labels"].append(
                {
                    "text": constellation["display_name"],
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "font_size": constellation_font_size,
                    "stroke_width": 3,
                    "text_rgba": rgba_to_list((225, 232, 245, 255)),
                    "stroke_rgba": rgba_to_list((0, 0, 0, 230)),
                }
            )

    for star in named_stars:
        if show_star_markers:
            scene["star_markers"].append(
                {
                    "x": float(star["x"]),
                    "y": float(star["y"]),
                    "radius": star_radius,
                    "fill_rgba": rgba_to_list((255, 210, 150, 215)),
                    "outline_rgba": rgba_to_list((255, 255, 255, 210)),
                }
            )
        if not show_star_labels:
            continue
        position = place_label(
            draw,
            star["name"],
            star["x"],
            star["y"],
            image_width,
            image_height,
            star_font,
            occupied_boxes,
            offsets=[
                (8.0, -20.0),
                (8.0, 10.0),
                (-86.0, -20.0),
                (-86.0, 10.0),
                (10.0, -34.0),
                (-96.0, -34.0),
                (8.0, -18.0),
                (8.0, 8.0),
                (-74.0, -18.0),
                (-74.0, 8.0),
            ],
            stroke_width=2,
            bounds=render_bounds,
        )
        if not position:
            continue
        leader_segment = None
        if show_label_leaders:
            leader_segment = compute_label_leader_segment(
                draw,
                star["x"],
                star["y"],
                position,
                star["name"],
                star_font,
                stroke_width=2,
            )
        scene["star_labels"].append(
            {
                "text": star["name"],
                "x": float(position[0]),
                "y": float(position[1]),
                "font_size": star_font_size,
                "stroke_width": 2,
                "text_rgba": rgba_to_list((250, 244, 236, 255)),
                "stroke_rgba": rgba_to_list((0, 0, 0, 220)),
                "leader": (
                    {
                        "x1": leader_segment[0],
                        "y1": leader_segment[1],
                        "x2": leader_segment[2],
                        "y2": leader_segment[3],
                        "line_width": 1,
                        "rgba": rgba_to_list((255, 233, 188, 176)),
                    }
                    if leader_segment is not None
                    else None
                ),
            }
        )

    return scene
