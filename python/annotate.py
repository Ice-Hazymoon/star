#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import tempfile
import time
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
from astroquery.vizier import Vizier
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skyfield.data import hipparcos, stellarium

try:
    import sep
except ImportError:  # pragma: no cover - optional dependency for source extraction
    sep = None

warnings.filterwarnings("ignore", category=FITSFixedWarning)

RESOURCE_KEY_PREFIXES = ("the_", "great_")
RESOURCE_KEY_SUFFIXES = (
    "_globular_cluster",
    "_open_cluster",
    "_supernova_remnant",
    "_planetary_nebula",
    "_star_cluster",
    "_galaxy_cluster",
    "_cluster",
    "_nebula",
    "_galaxy",
    "_remnant",
)
CONSTELLATION_RESOURCE_OVERRIDES = {
    "Ser": ("serpens_caput", "serpens_cauda"),
}
SUPPLEMENTAL_CONSTELLATION_ABBR_OVERRIDES = {
    "serpens_caput": "Ser",
    "serpens_cauda": "Ser",
}

DEFAULT_OVERLAY_OPTIONS = {
    "preset": "max",
    "layers": {
        "constellation_lines": True,
        "constellation_labels": True,
        "contextual_constellation_labels": True,
        "star_markers": True,
        "star_labels": True,
        "deep_sky_markers": True,
        "deep_sky_labels": True,
        "label_leaders": True,
    },
    "detail": {
        "star_label_limit": 36,
        "star_magnitude_limit": 4.8,
        "star_bright_separation": 82.0,
        "star_dim_separation": 60.0,
        "dso_label_limit": 48,
        "dso_magnitude_limit": 13.0,
        "dso_spacing_scale": 0.58,
        "show_all_constellation_labels": True,
        "detailed_dso_labels": True,
        "include_catalog_dsos": True,
    },
}

OVERLAY_PRESETS = {
    "balanced": {
        "detail": {
            "star_label_limit": 18,
            "star_magnitude_limit": 3.8,
            "star_bright_separation": 105.0,
            "star_dim_separation": 82.0,
            "dso_label_limit": 24,
            "dso_magnitude_limit": 11.0,
            "dso_spacing_scale": 0.7,
            "show_all_constellation_labels": False,
            "detailed_dso_labels": False,
            "include_catalog_dsos": False,
        },
    },
    "detailed": {
        "detail": {
            "star_label_limit": 28,
            "star_magnitude_limit": 4.4,
            "star_bright_separation": 92.0,
            "star_dim_separation": 72.0,
            "dso_label_limit": 36,
            "dso_magnitude_limit": 12.2,
            "dso_spacing_scale": 0.64,
            "show_all_constellation_labels": False,
            "detailed_dso_labels": True,
            "include_catalog_dsos": True,
        },
    },
    "max": {
        "detail": {
            "star_label_limit": 36,
            "star_magnitude_limit": 4.8,
            "star_bright_separation": 82.0,
            "star_dim_separation": 60.0,
            "dso_label_limit": 48,
            "dso_magnitude_limit": 13.0,
            "dso_spacing_scale": 0.58,
            "show_all_constellation_labels": True,
            "detailed_dso_labels": True,
            "include_catalog_dsos": True,
        },
    },
}


@dataclass(frozen=True)
class CropCandidate:
    name: str
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class SourceDetection:
    x: float
    y: float
    flux: float
    peak: float
    major: float
    minor: float
    npix: int
    elongation: float
    star_score: float
    sort_flux: float


@dataclass
class SourceAnalysis:
    mode: str
    detections: list[SourceDetection]
    tile_scores: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class SolveResult:
    crop: CropCandidate
    downsample: int
    scale_low: float
    scale_high: float
    input_mode: str
    wcs_path: Path
    stdout: str
    stderr: str
    corr_path: Path | None = None
    verification: dict[str, Any] | None = None


def verification_score(verification: dict[str, Any], crop: CropCandidate, image_width: int, image_height: int) -> float:
    area_ratio = (crop.width * crop.height) / max(image_width * image_height, 1)
    match_count = float(verification.get("match_count", 0))
    rms_px = float(verification.get("rms_px", 99.0))
    max_px = float(verification.get("max_px", 99.0))
    spread = float(verification.get("spread_x", 0.0)) + float(verification.get("spread_y", 0.0))
    covered_quadrants = float(verification.get("covered_quadrants", 0))
    alignment_mean = float(verification.get("alignment_mean_px", 30.0))
    alignment_p75 = float(verification.get("alignment_p75_px", 45.0))
    log_matches = math.log1p(max(match_count, 0.0))
    return (
        log_matches * 32.0
        + spread * 40.0
        + area_ratio * 20.0
        + covered_quadrants * 4.0
        - rms_px * 8.0
        - max_px * 1.1
        - alignment_mean * 1.2
        - alignment_p75 * 0.6
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate a star-field image with constellation lines and star names.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-image", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--constellations", action="append", required=True)
    parser.add_argument("--star-names", required=True)
    parser.add_argument("--dso-catalog", action="append", default=[])
    parser.add_argument("--localization", action="append", default=[])
    parser.add_argument("--locale", default="en")
    parser.add_argument("--supplemental-dso", default="")
    parser.add_argument("--options-json", default="")
    return parser.parse_args()


def batched(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def merge_nested_dict(target: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_nested_dict(target[key], value)
        else:
            target[key] = value
    return target


def coerce_int(value: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, numeric))


def coerce_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, numeric))


def overlay_layer_enabled(overlay_options: dict[str, Any], key: str) -> bool:
    return bool(overlay_options.get("layers", {}).get(key, False))


def overlay_detail_value(overlay_options: dict[str, Any], key: str) -> Any:
    return overlay_options.get("detail", {}).get(key)


def parse_overlay_options(raw_options: str) -> dict[str, Any]:
    options = deepcopy(DEFAULT_OVERLAY_OPTIONS)
    payload: dict[str, Any] = {}
    if raw_options:
        try:
            parsed = json.loads(raw_options)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid overlay options JSON: {exc}") from exc
        if isinstance(parsed, dict):
            payload = parsed

    preset_name = str(payload.get("preset") or options["preset"]).strip().lower()
    preset = OVERLAY_PRESETS.get(preset_name, OVERLAY_PRESETS[options["preset"]])
    options["preset"] = preset_name if preset_name in OVERLAY_PRESETS else options["preset"]
    merge_nested_dict(options, deepcopy(preset))
    payload_without_preset = {key: value for key, value in payload.items() if key != "preset"}
    merge_nested_dict(options, payload_without_preset)

    detail = options["detail"]
    detail["star_label_limit"] = coerce_int(detail.get("star_label_limit"), 36, 0, 80)
    detail["star_magnitude_limit"] = coerce_float(detail.get("star_magnitude_limit"), 4.8, 0.0, 8.0)
    detail["star_bright_separation"] = coerce_float(detail.get("star_bright_separation"), 82.0, 20.0, 180.0)
    detail["star_dim_separation"] = coerce_float(detail.get("star_dim_separation"), 60.0, 12.0, 150.0)
    detail["dso_label_limit"] = coerce_int(detail.get("dso_label_limit"), 48, 0, 120)
    detail["dso_magnitude_limit"] = coerce_float(detail.get("dso_magnitude_limit"), 13.0, 0.0, 20.0)
    detail["dso_spacing_scale"] = coerce_float(detail.get("dso_spacing_scale"), 0.58, 0.1, 1.5)
    detail["show_all_constellation_labels"] = bool(detail.get("show_all_constellation_labels"))
    detail["detailed_dso_labels"] = bool(detail.get("detailed_dso_labels"))
    detail["include_catalog_dsos"] = bool(detail.get("include_catalog_dsos"))
    return options


def normalize_catalog_frame(frame: pd.DataFrame) -> pd.DataFrame:
    catalog = frame.copy()
    if catalog.empty:
        return pd.DataFrame(columns=["magnitude", "ra_degrees", "dec_degrees"]).rename_axis("hip")
    if "hip" not in catalog.columns:
        catalog = catalog.reset_index()
    catalog = catalog.rename(columns={"HIP": "hip", "RAICRS": "ra_degrees", "DEICRS": "dec_degrees", "Vmag": "magnitude"})
    catalog["hip"] = catalog["hip"].astype(int)
    for column in ("magnitude", "ra_degrees", "dec_degrees"):
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    catalog = catalog.replace([np.inf, -np.inf], np.nan).dropna(subset=["ra_degrees", "dec_degrees", "magnitude"])
    catalog = catalog.drop_duplicates(subset="hip").set_index("hip").sort_index()
    return catalog


def fetch_minimal_catalog(
    catalog_path: Path,
    required_hips: set[int],
    existing_catalog: pd.DataFrame | None = None,
    prefer_full_catalog: bool = True,
) -> pd.DataFrame:
    Vizier.ROW_LIMIT = -1
    rows: list[dict[str, float | int]] = []

    if prefer_full_catalog:
        try:
            table = Vizier(columns=["HIP", "RAICRS", "DEICRS", "Vmag"]).get_catalogs("I/239/hip_main")[0]
            frame = table.to_pandas()[["HIP", "RAICRS", "DEICRS", "Vmag"]]
            frame["HIP"] = frame["HIP"].astype(int)
            frame = frame[frame["HIP"].isin(sorted(required_hips))]
            rows.extend(frame.to_dict("records"))
        except Exception:
            pass

    if not rows:
        for chunk in batched(sorted(required_hips), 20):
            result = Vizier(columns=["HIP", "RAICRS", "DEICRS", "Vmag"]).query_constraints(
                catalog="I/239/hip_main",
                HIP=",".join(str(hip) for hip in chunk),
            )
            if not result:
                continue
            for entry in result[0]:
                ra_value = entry["RAICRS"]
                dec_value = entry["DEICRS"]
                magnitude_value = entry["Vmag"]
                if np.ma.is_masked(ra_value) or np.ma.is_masked(dec_value) or np.ma.is_masked(magnitude_value):
                    continue
                rows.append(
                    {
                        "hip": int(entry["HIP"]),
                        "magnitude": float(magnitude_value),
                        "ra_degrees": float(ra_value),
                        "dec_degrees": float(dec_value),
                    }
                )

    fetched_catalog = normalize_catalog_frame(pd.DataFrame(rows))
    if existing_catalog is not None and not existing_catalog.empty:
        fetched_catalog = normalize_catalog_frame(pd.concat([existing_catalog.reset_index(), fetched_catalog.reset_index()], ignore_index=True))

    if fetched_catalog.empty:
        raise RuntimeError("failed to build minimal Hipparcos cache from VizieR")
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    fetched_catalog.to_csv(catalog_path)
    return fetched_catalog


def load_catalog(catalog_path: Path, required_hips: set[int]) -> pd.DataFrame:
    if catalog_path.exists():
        if catalog_path.suffix.lower() in {".dat", ".gz"}:
            with catalog_path.open("rb") as handle:
                catalog = hipparcos.load_dataframe(handle)
            return catalog[catalog["magnitude"].notna()].copy()

        catalog = normalize_catalog_frame(pd.read_csv(catalog_path))
        missing = required_hips - set(int(value) for value in catalog.index)
        if not missing:
            return catalog
        return fetch_minimal_catalog(catalog_path, missing, existing_catalog=catalog, prefer_full_catalog=False)

    return fetch_minimal_catalog(catalog_path, required_hips, prefer_full_catalog=False)


def strip_diacritics(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(character for character in normalized if not unicodedata.combining(character))


def normalize_constellation_key(value: str | None) -> str:
    text = strip_diacritics((value or "").strip().lower())
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if normalized:
        return normalized
    return re.sub(r"\s+", "", (value or "").strip().lower())


def normalize_lookup_key(value: str | None) -> str:
    return normalize_constellation_key(value).replace("_", "")


def normalize_human_alias(value: str | None) -> str | None:
    text = (value or "").replace("_", " ").strip()
    if not text:
        return None
    return re.sub(r"\s+", " ", text)


def resource_key_candidates(*values: str | None) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()

    def append_candidate(candidate: str | None) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            results.append(candidate)

    queue: list[str] = []
    queued: set[str] = set()

    def enqueue(candidate: str | None) -> None:
        if candidate and candidate not in queued:
            queued.add(candidate)
            queue.append(candidate)

    for value in values:
        enqueue(normalize_constellation_key(value))

    while queue:
        candidate = queue.pop(0)
        append_candidate(candidate)
        for prefix in RESOURCE_KEY_PREFIXES:
            if candidate.startswith(prefix):
                enqueue(candidate[len(prefix) :])
        for suffix in RESOURCE_KEY_SUFFIXES:
            if candidate.endswith(suffix) and len(candidate) > len(suffix):
                enqueue(candidate[: -len(suffix)])

    return results


@dataclass(frozen=True)
class LocalizationBundle:
    requested_locale: str
    resolved_locale: str
    available_locales: list[str]
    strings: dict[str, str]


def canonicalize_locale_tag(value: str | None) -> str:
    text = (value or "").replace("_", "-").strip()
    if not text:
        return "en"
    parts = [part for part in text.split("-") if part]
    if not parts:
        return "en"

    normalized: list[str] = []
    for index, part in enumerate(parts):
        if index == 0:
            normalized.append(part.lower())
        elif len(part) == 4 and part.isalpha():
            normalized.append(part.title())
        elif len(part) in {2, 3} and part.isalnum():
            normalized.append(part.upper())
        else:
            normalized.append(part)
    return "-".join(normalized)


def locale_candidates(requested_locale: str) -> list[str]:
    requested = canonicalize_locale_tag(requested_locale)
    candidates: list[str] = []

    def append_candidate(candidate: str | None) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    append_candidate(requested)

    parts = requested.split("-")
    if parts and parts[0] == "zh":
        regions = {part for part in parts[1:] if len(part) in {2, 3} and part.isupper()}
        if "Hans" in parts or regions.intersection({"CN", "SG", "MY"}):
            append_candidate("zh-Hans")
        if "Hant" in parts or regions.intersection({"TW", "HK", "MO"}):
            append_candidate("zh-Hant")

    while len(parts) > 1:
        parts = parts[:-1]
        append_candidate("-".join(parts))

    append_candidate("en")
    return candidates


def android_values_directory_to_locale(values_dir_name: str) -> str:
    if values_dir_name == "values":
        return "en"
    if values_dir_name.startswith("values-b+"):
        return canonicalize_locale_tag(values_dir_name[len("values-b+") :].replace("+", "-"))
    if values_dir_name.startswith("values-"):
        return canonicalize_locale_tag(values_dir_name[len("values-") :])
    return "en"


def load_localized_names(localization_paths: list[Path], locale: str | None = None) -> LocalizationBundle:
    localized_catalogs: dict[str, dict[str, str]] = {}
    for localization_path in localization_paths:
        if not localization_path.exists():
            continue
        locale_tag = android_values_directory_to_locale(localization_path.parent.name)
        localized_names = localized_catalogs.setdefault(locale_tag, {})
        root = ET.parse(localization_path).getroot()
        for node in root.findall("./string"):
            key = normalize_constellation_key(node.attrib.get("name"))
            value = "".join(node.itertext()).strip()
            if key and value and key not in localized_names:
                localized_names[key] = value

    available_locales = sorted(localized_catalogs)
    resolved_locale = "en" if "en" in localized_catalogs else (available_locales[0] if available_locales else "en")
    strings = dict(localized_catalogs.get("en", {}))
    requested_locale = canonicalize_locale_tag(locale)

    for candidate in locale_candidates(requested_locale):
        if candidate not in localized_catalogs:
            continue
        resolved_locale = candidate
        strings = dict(localized_catalogs.get("en", {}))
        strings.update(localized_catalogs[candidate])
        break

    return LocalizationBundle(
        requested_locale=requested_locale,
        resolved_locale=resolved_locale,
        available_locales=available_locales,
        strings=strings,
    )


def resolve_localized_name(localized_names: dict[str, str], *values: str | None) -> str | None:
    for key in resource_key_candidates(*values):
        translated = localized_names.get(key)
        if translated:
            return translated.strip()
    return None


def resolve_constellation_display_name(
    abbr: str,
    english_name: str,
    native_name: str | None,
    localized_names: dict[str, str],
) -> str:
    overrides = CONSTELLATION_RESOURCE_OVERRIDES.get(abbr, ())
    translated = resolve_localized_name(localized_names, native_name, english_name, *overrides)
    if translated:
        if abbr == "Ser":
            return re.sub(r"[（(].*?[）)]", "", translated).strip()
        return translated
    return native_name or english_name or abbr


def load_star_names(star_names_path: Path, localized_names: dict[str, str]) -> dict[int, str]:
    _ = localized_names
    with star_names_path.open("rb") as handle:
        entries = stellarium.parse_star_names(handle)
    return {entry.hip: entry.name for entry in entries}


def build_constellation_entry(
    abbr: str,
    english_name: str,
    native_name: str | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    resolved_native_name = native_name or english_name or abbr
    return {
        "abbr": abbr,
        "english_name": english_name or abbr,
        "native_name": resolved_native_name,
        "display_name": display_name or resolved_native_name,
        "lines": [],
        "coord_lines": [],
        "label_ra_degrees": None,
        "label_dec_degrees": None,
    }


def merge_constellation_entries(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    if not target["english_name"] and incoming["english_name"]:
        target["english_name"] = incoming["english_name"]
    if not target["native_name"] and incoming["native_name"]:
        target["native_name"] = incoming["native_name"]
    if not target["display_name"] and incoming["display_name"]:
        target["display_name"] = incoming["display_name"]

    for polyline in incoming.get("lines", []):
        if len(polyline) >= 2:
            target["lines"].append([int(hip) for hip in polyline])

    for polyline in incoming.get("coord_lines", []):
        if len(polyline) >= 2:
            target["coord_lines"].append(polyline)

    if target["label_ra_degrees"] is None and incoming.get("label_ra_degrees") is not None:
        target["label_ra_degrees"] = float(incoming["label_ra_degrees"])
    if target["label_dec_degrees"] is None and incoming.get("label_dec_degrees") is not None:
        target["label_dec_degrees"] = float(incoming["label_dec_degrees"])


def build_constellation_name_map(constellations: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for constellation in constellations:
        for candidate in (
            constellation["abbr"],
            constellation["english_name"],
            constellation["native_name"],
            constellation["display_name"],
        ):
            key = normalize_constellation_key(candidate)
            if key:
                mapping[key] = constellation["abbr"]
    return mapping


def parse_proto_scalar(value_text: str) -> str | float | int:
    value_text = value_text.strip()
    if value_text.startswith('"') and value_text.endswith('"'):
        return value_text[1:-1]
    try:
        if any(marker in value_text for marker in (".", "e", "E")):
            return float(value_text)
        return int(value_text)
    except ValueError:
        return value_text


def parse_stardroid_constellations(constellation_path: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    stack: list[tuple[str, dict[str, Any]]] = []

    for raw_line in constellation_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.endswith("{"):
            block_name = line[:-1].strip()
            parent_type, parent_target = stack[-1] if stack else ("", {})
            if block_name == "source":
                target = {"labels": [], "lines": [], "name_keys": []}
                sources.append(target)
            elif block_name == "search_location":
                target = {}
                parent_target["search_location"] = target
            elif block_name == "label":
                target = {}
                parent_target["labels"].append(target)
            elif block_name == "location":
                if parent_type != "label":
                    continue
                target = {}
                parent_target["location"] = target
            elif block_name == "line":
                target = {"vertices": []}
                parent_target["lines"].append(target)
            elif block_name == "vertex":
                target = {}
                parent_target["vertices"].append(target)
            else:
                continue
            stack.append((block_name, target))
            continue

        if line == "}":
            if stack:
                stack.pop()
            continue

        if ":" not in line or not stack:
            continue

        key, value_text = line.split(":", 1)
        key = key.strip()
        block_type, target = stack[-1]
        parsed_value = parse_proto_scalar(value_text)

        if block_type == "source" and key == "name_str_ids":
            target.setdefault("name_keys", []).append(str(parsed_value))
            continue

        target[key] = parsed_value

    parsed_sources: list[dict[str, Any]] = []
    for source in sources:
        coord_lines: list[list[dict[str, float]]] = []
        for raw_line in source.get("lines", []):
            vertices: list[dict[str, float]] = []
            for vertex in raw_line.get("vertices", []):
                ra_value = vertex.get("right_ascension")
                dec_value = vertex.get("declination")
                if ra_value is None or dec_value is None:
                    continue
                vertices.append(
                    {
                        "ra_degrees": float(ra_value),
                        "dec_degrees": float(dec_value),
                    }
                )
            if len(vertices) >= 2:
                coord_lines.append(vertices)

        if not coord_lines:
            continue

        label_block = next((label for label in source.get("labels", []) if label.get("strings_str_id")), None)
        if label_block is None:
            label_block = source.get("labels", [{}])[0] if source.get("labels") else {}
        label_location = label_block.get("location", {}) if isinstance(label_block, dict) else {}
        search_location = source.get("search_location", {})

        location = label_location or search_location
        raw_name_key = (
            (label_block.get("strings_str_id") if isinstance(label_block, dict) else None)
            or next((item for item in source.get("name_keys", []) if item), "")
        )
        english_name = normalize_human_alias(raw_name_key) or "Unknown"
        if english_name == "Unknown":
            continue

        parsed_sources.append(
            {
                "name_key": raw_name_key,
                "english_name": english_name.title(),
                "coord_lines": coord_lines,
                "label_ra_degrees": float(location["right_ascension"]) if "right_ascension" in location else None,
                "label_dec_degrees": float(location["declination"]) if "declination" in location else None,
            }
        )
    return parsed_sources


def load_constellations(constellation_paths: list[Path], localized_names: dict[str, str]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    json_paths = [path for path in constellation_paths if path.suffix.lower() == ".json"]
    supplemental_paths = [path for path in constellation_paths if path.suffix.lower() != ".json"]

    for constellation_path in json_paths:
        data = json.loads(constellation_path.read_text())
        for item in data["constellations"]:
            common_name = item.get("common_name", {})
            constellation_id = item["id"].split()[-1]
            fallback_name = common_name.get("native", common_name.get("english", constellation_id))
            incoming = build_constellation_entry(
                constellation_id,
                common_name.get("english", constellation_id),
                fallback_name,
                resolve_constellation_display_name(
                    constellation_id,
                    common_name.get("english", constellation_id),
                    fallback_name,
                    localized_names,
                ),
            )
            incoming["lines"] = [[int(hip) for hip in polyline] for polyline in item["lines"] if len(polyline) >= 2]
            target = merged.setdefault(
                constellation_id,
                build_constellation_entry(
                    constellation_id,
                    incoming["english_name"],
                    incoming["native_name"],
                    incoming["display_name"],
                ),
            )
            merge_constellation_entries(target, incoming)

    name_map = build_constellation_name_map(list(merged.values()))

    for constellation_path in supplemental_paths:
        for item in parse_stardroid_constellations(constellation_path):
            abbr = name_map.get(normalize_constellation_key(item["name_key"]))
            if not abbr:
                abbr = name_map.get(normalize_constellation_key(item["english_name"]))
            if not abbr:
                abbr = SUPPLEMENTAL_CONSTELLATION_ABBR_OVERRIDES.get(normalize_constellation_key(item["name_key"]))
            english_name = item["english_name"]
            if not abbr:
                continue
            resolved_abbr = abbr

            existing = merged.get(resolved_abbr)
            incoming = build_constellation_entry(
                resolved_abbr,
                existing["english_name"] if existing else english_name,
                existing["native_name"] if existing else english_name,
                existing["display_name"]
                if existing
                else resolve_constellation_display_name(resolved_abbr, english_name, english_name, localized_names),
            )
            incoming["coord_lines"] = item["coord_lines"]
            incoming["label_ra_degrees"] = item["label_ra_degrees"]
            incoming["label_dec_degrees"] = item["label_dec_degrees"]

            target = merged.setdefault(
                resolved_abbr,
                build_constellation_entry(
                    resolved_abbr,
                    incoming["english_name"],
                    incoming["native_name"],
                    incoming["display_name"],
                ),
            )
            merge_constellation_entries(target, incoming)
            name_map[normalize_constellation_key(english_name)] = resolved_abbr

    return sorted(merged.values(), key=lambda item: item["abbr"])


def collect_required_hips(constellations: list[dict[str, Any]], star_names: dict[int, str]) -> set[int]:
    hips = set(star_names)
    for constellation in constellations:
        for line in constellation["lines"]:
            hips.update(line)
    return hips


def parse_optional_float(value: str | None) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_messier_label(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return f"M{int(text)}"
    except ValueError:
        normalized = text.lstrip("0") or text
        return f"M{normalized}"


def choose_common_name(names: list[str]) -> str | None:
    cleaned = [name.strip() for name in names if name and name.strip()]
    if not cleaned:
        return None
    for candidate in cleaned:
        if any(character.isupper() for character in candidate):
            return candidate
    for candidate in cleaned:
        if " " in candidate:
            return candidate
    for candidate in cleaned:
        if any(character.isalpha() for character in candidate):
            return candidate
    return cleaned[0]


def dedupe_aliases(names: list[str]) -> list[str]:
    unique_names: list[str] = []
    seen: set[str] = set()
    for name in names:
        candidate = name.strip()
        key = candidate.casefold()
        if candidate and key not in seen:
            seen.add(key)
            unique_names.append(candidate)
    return unique_names


def strip_catalog_prefix(label: str, *prefixes: str | None) -> str:
    resolved_label = label.strip()
    for prefix in prefixes:
        prefix_text = (prefix or "").strip()
        if not prefix_text:
            continue
        stripped = re.sub(
            rf"^{re.escape(prefix_text)}(?:\s+|[：:：-]\s*)?",
            "",
            resolved_label,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        if stripped and stripped != resolved_label:
            return stripped
    return resolved_label


def resolve_dso_label(
    name: str,
    messier: str | None,
    common_names: list[str],
    localized_names: dict[str, str],
    catalog_id: str | None = None,
) -> str:
    translated = resolve_localized_name(localized_names, messier, *common_names, name, catalog_id)
    if translated:
        return strip_catalog_prefix(translated, messier, catalog_id)
    if messier:
        return messier
    common_name = choose_common_name(common_names)
    if common_name:
        return common_name
    return name


def normalize_constellation_abbr(value: str | None, constellation_name_map: dict[str, str]) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    return constellation_name_map.get(normalize_constellation_key(text), text)


def build_dso_key(
    name: str,
    messier: str | None,
    common_names: list[str],
    catalog_id: str | None = None,
) -> str:
    if messier:
        return normalize_lookup_key(messier)
    if catalog_id:
        return normalize_lookup_key(catalog_id)
    if name:
        return normalize_lookup_key(name)
    common_name = choose_common_name(common_names)
    return normalize_lookup_key(common_name)


def merge_dso_entry(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    target["common_names"] = dedupe_aliases([*target.get("common_names", []), *incoming.get("common_names", [])])
    if incoming.get("curated"):
        target["curated"] = True

    for key in ("type", "const", "messier", "catalog_id"):
        if not target.get(key) and incoming.get(key):
            target[key] = incoming[key]

    for key in ("ra_degrees", "dec_degrees", "magnitude"):
        if target.get(key) is None and incoming.get(key) is not None:
            target[key] = incoming[key]

    if incoming.get("major_axis_arcmin") is not None:
        target["major_axis_arcmin"] = max(target.get("major_axis_arcmin") or 0.0, incoming["major_axis_arcmin"])

    current_common_name = target.get("common_name")
    incoming_common_name = incoming.get("common_name")
    if not current_common_name and incoming_common_name:
        target["common_name"] = incoming_common_name

    current_label = target.get("label")
    incoming_label = incoming.get("label")
    if incoming_label and (not current_label or current_label == target.get("name") or incoming.get("curated")):
        target["label"] = incoming_label


def load_openngc_objects(
    dso_path: Path,
    constellation_name_map: dict[str, str],
    localized_names: dict[str, str],
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    with dso_path.open(encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            name = (row.get("Name") or "").strip()
            ra_text = (row.get("RA") or "").strip()
            dec_text = (row.get("Dec") or "").strip()
            if not name or not ra_text or not dec_text:
                continue
            try:
                coord = SkyCoord(ra_text, dec_text, unit=(u.hourangle, u.deg))
            except ValueError:
                continue

            common_names = dedupe_aliases([item.strip() for item in (row.get("Common names") or "").split(",") if item.strip()])
            messier = format_messier_label(row.get("M"))
            catalog_id = name if name.upper().startswith(("NGC", "IC", "SH2", "B", "C")) else None
            objects.append(
                {
                    "name": name,
                    "type": (row.get("Type") or "").strip(),
                    "const": normalize_constellation_abbr(row.get("Const"), constellation_name_map),
                    "ra_degrees": float(coord.ra.deg),
                    "dec_degrees": float(coord.dec.deg),
                    "major_axis_arcmin": parse_optional_float(row.get("MajAx")),
                    "magnitude": parse_optional_float(row.get("V-Mag")) or parse_optional_float(row.get("B-Mag")),
                    "messier": messier,
                    "catalog_id": catalog_id,
                    "common_name": choose_common_name(common_names),
                    "common_names": common_names,
                    "label": resolve_dso_label(name, messier, common_names, localized_names, catalog_id),
                    "curated": False,
                }
            )
    return objects


def load_stardroid_dso_objects(
    dso_path: Path,
    constellation_name_map: dict[str, str],
    localized_names: dict[str, str],
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    with dso_path.open(encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return []
        for row in reader:
            if not row:
                continue
            if len(row) < 10:
                continue
            if len(row) > 10:
                row = [*row[:9], ",".join(part.strip() for part in row[9:] if part.strip())]

            aliases = [normalize_human_alias(value) for value in row[0].split("|")]
            aliases = [alias for alias in aliases if alias]
            if not aliases:
                continue

            primary_name = aliases[0]
            common_name = normalize_human_alias(row[9])
            common_names = dedupe_aliases([*aliases[1:], *([common_name] if common_name else [])])

            try:
                ra_degrees = float(row[2]) * 15.0
                dec_degrees = float(row[3])
            except ValueError:
                continue

            messier = None
            if primary_name.upper().startswith("M") and primary_name[1:].replace(".", "", 1).isdigit():
                messier = format_messier_label(primary_name[1:])

            catalog_id = normalize_human_alias(row[6])
            objects.append(
                {
                    "name": primary_name,
                    "type": normalize_human_alias(row[1]) or "",
                    "const": normalize_constellation_abbr(row[7], constellation_name_map),
                    "ra_degrees": ra_degrees,
                    "dec_degrees": dec_degrees,
                    "major_axis_arcmin": parse_optional_float(row[5]),
                    "magnitude": parse_optional_float(row[4]),
                    "messier": messier,
                    "catalog_id": catalog_id,
                    "common_name": choose_common_name(common_names),
                    "common_names": common_names,
                    "label": resolve_dso_label(primary_name, messier, common_names, localized_names, catalog_id),
                    "curated": True,
                }
            )
    return objects


def load_supplemental_deep_sky_objects(
    supplemental_dso_path: Path | None,
    constellation_name_map: dict[str, str],
    localized_names: dict[str, str],
) -> list[dict[str, Any]]:
    if supplemental_dso_path is None or not supplemental_dso_path.exists():
        return []

    items = json.loads(supplemental_dso_path.read_text(encoding="utf-8"))
    objects: list[dict[str, Any]] = []
    for item in items:
        name = str(item.get("name") or "").strip()
        ra_text = str(item.get("ra") or "").strip()
        dec_text = str(item.get("dec") or "").strip()
        if not name or not ra_text or not dec_text:
            continue

        coord = SkyCoord(ra_text, dec_text, unit=(u.hourangle, u.deg))
        common_names = dedupe_aliases([str(value).strip() for value in item.get("common_names", []) if str(value).strip()])
        common_name = choose_common_name(common_names)
        messier = format_messier_label(item.get("messier"))
        catalog_id = str(item.get("catalog_id") or name).strip() or name
        label = resolve_localized_name(
            localized_names,
            str(item.get("label_key") or ""),
            messier,
            *common_names,
            name,
            catalog_id,
        )
        objects.append(
            {
                "name": name,
                "type": str(item.get("type") or "").strip(),
                "const": normalize_constellation_abbr(item.get("const"), constellation_name_map),
                "ra_degrees": float(coord.ra.deg),
                "dec_degrees": float(coord.dec.deg),
                "major_axis_arcmin": parse_optional_float(str(item.get("major_axis_arcmin") or "")),
                "magnitude": parse_optional_float(str(item.get("magnitude") or "")),
                "messier": messier,
                "catalog_id": catalog_id,
                "common_name": common_name,
                "common_names": common_names,
                "label": strip_catalog_prefix(label, messier, catalog_id) if label else (common_name or name),
                "curated": True,
            }
        )

    return objects


def load_deep_sky_objects(
    dso_paths: list[Path],
    constellation_name_map: dict[str, str],
    localized_names: dict[str, str],
    supplemental_dso_path: Path | None = None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for dso_path in dso_paths:
        if not dso_path.exists():
            continue

        with dso_path.open(encoding="utf-8", errors="ignore") as handle:
            header = handle.readline()

        if header.startswith("Object,Type,RA (h),DEC (deg)"):
            loaded_objects = load_stardroid_dso_objects(dso_path, constellation_name_map, localized_names)
        else:
            loaded_objects = load_openngc_objects(dso_path, constellation_name_map, localized_names)

        for item in loaded_objects:
            key = build_dso_key(item["name"], item.get("messier"), item.get("common_names", []), item.get("catalog_id"))
            if not key:
                continue
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(item)
                continue
            merge_dso_entry(existing, item)

    for item in load_supplemental_deep_sky_objects(supplemental_dso_path, constellation_name_map, localized_names):
        key = build_dso_key(item["name"], item.get("messier"), item.get("common_names", []), item.get("catalog_id"))
        existing = merged.get(key)
        if existing is None:
            merged[key] = item
        else:
            merge_dso_entry(existing, item)

    return list(merged.values())


def normalize_image(input_path: Path, workdir: Path) -> tuple[Image.Image, Path]:
    with Image.open(input_path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
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


def analyze_sources(image: Image.Image) -> SourceAnalysis:
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
    objects = sep.extract(data, thresh=threshold, err=background.globalrms, minarea=3)

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


def select_sources_for_crop(source_analysis: SourceAnalysis, crop: CropCandidate) -> list[SourceDetection]:
    selected: list[SourceDetection] = []
    secondary: list[SourceDetection] = []
    for detection in source_analysis.detections:
        if not (crop.x <= detection.x <= crop.x + crop.width and crop.y <= detection.y <= crop.y + crop.height):
            continue
        if detection.star_score >= 0.4:
            selected.append(detection)
        elif detection.star_score >= 0.28:
            secondary.append(detection)

    limit = max(90, min(900, int(crop.width * crop.height / 7000)))
    selected = selected[:limit]
    if len(selected) < min(60, limit):
        selected.extend(secondary[: min(limit - len(selected), 80)])
    return sorted(selected, key=lambda item: item.sort_flux, reverse=True)


def write_xylist(
    source_analysis: SourceAnalysis,
    crop: CropCandidate,
    workdir: Path,
) -> tuple[Path | None, int]:
    selected_sources = select_sources_for_crop(source_analysis, crop)
    if len(selected_sources) < 12:
        return None, len(selected_sources)

    x_values = np.array([source.x - crop.x + 1.0 for source in selected_sources], dtype=np.float64)
    y_values = np.array([source.y - crop.y + 1.0 for source in selected_sources], dtype=np.float64)
    flux_values = np.array([source.sort_flux for source in selected_sources], dtype=np.float32)

    columns = [
        fits.Column(name="X", format="D", array=x_values),
        fits.Column(name="Y", format="D", array=y_values),
        fits.Column(name="FLUX", format="E", array=flux_values),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    table.header["IMAGEW"] = crop.width
    table.header["IMAGEH"] = crop.height
    xyls_path = workdir / f"{crop.name}.xyls"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(xyls_path, overwrite=True)
    return xyls_path, len(selected_sources)


def run_solve_on_xylist(
    xylist_path: Path,
    crop: CropCandidate,
    scale_low: float,
    scale_high: float,
    workdir: Path,
    index_dir: Path,
) -> SolveResult | None:
    base_name = f"solve-{crop.name}-xyls-s{int(scale_low)}-{int(scale_high)}"
    command = [
        "solve-field",
        "--overwrite",
        "--no-plots",
        "--dir",
        str(workdir),
        "--out",
        base_name,
        "--index-dir",
        str(index_dir),
        "--cpulimit",
        "45",
        "--scale-units",
        "degwidth",
        "--scale-low",
        str(scale_low),
        "--scale-high",
        str(scale_high),
        "--width",
        str(crop.width),
        "--height",
        str(crop.height),
        "--x-column",
        "X",
        "--y-column",
        "Y",
        "--sort-column",
        "FLUX",
        "--no-verify",
        "--depth",
        "20,40,80,160",
        "--new-fits",
        "none",
        str(xylist_path),
    ]

    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    wcs_path = workdir / f"{base_name}.wcs"
    solved_flag = workdir / f"{base_name}.solved"
    corr_path = workdir / f"{base_name}.corr"

    if proc.returncode == 0 and wcs_path.exists() and solved_flag.exists():
        return SolveResult(
            crop=crop,
            downsample=1,
            scale_low=scale_low,
            scale_high=scale_high,
            input_mode="xyls",
            wcs_path=wcs_path,
            stdout=proc.stdout,
            stderr=proc.stderr,
            corr_path=corr_path if corr_path.exists() else None,
        )

    return None


def run_solve_on_image(
    image_path: Path,
    crop: CropCandidate,
    downsample: int,
    scale_low: float,
    scale_high: float,
    workdir: Path,
    index_dir: Path,
) -> SolveResult | None:
    base_name = f"solve-{crop.name}-ds{downsample}-s{int(scale_low)}-{int(scale_high)}"
    command = [
        "solve-field",
        "--overwrite",
        "--no-plots",
        "--dir",
        str(workdir),
        "--out",
        base_name,
        "--index-dir",
        str(index_dir),
        "--cpulimit",
        "90",
        "--downsample",
        str(downsample),
        "--scale-units",
        "degwidth",
        "--scale-low",
        str(scale_low),
        "--scale-high",
        str(scale_high),
        "--new-fits",
        "none",
        str(image_path),
    ]

    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    wcs_path = workdir / f"{base_name}.wcs"
    solved_flag = workdir / f"{base_name}.solved"

    if proc.returncode == 0 and wcs_path.exists() and solved_flag.exists():
        return SolveResult(
            crop=crop,
            downsample=downsample,
            scale_low=scale_low,
            scale_high=scale_high,
            input_mode="image",
            wcs_path=wcs_path,
            stdout=proc.stdout,
            stderr=proc.stderr,
            corr_path=(workdir / f"{base_name}.corr") if (workdir / f"{base_name}.corr").exists() else None,
        )

    return None


def verify_solution(result: SolveResult) -> dict[str, Any]:
    corr_path = result.corr_path
    if corr_path is None or not corr_path.exists():
        return {
            "accepted": False,
            "reason": "missing-corr",
            "match_count": 0,
        }

    with fits.open(corr_path) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            return {
                "accepted": False,
                "reason": "empty-corr",
                "match_count": 0,
            }
        table = hdul[1].data
        match_count = len(table)
        if match_count == 0:
            return {
                "accepted": False,
                "reason": "zero-matches",
                "match_count": 0,
            }

        field_x = np.asarray(table["field_x"], dtype=np.float64)
        field_y = np.asarray(table["field_y"], dtype=np.float64)
        index_x = np.asarray(table["index_x"], dtype=np.float64)
        index_y = np.asarray(table["index_y"], dtype=np.float64)

    residuals = np.hypot(field_x - index_x, field_y - index_y)
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    median = float(np.median(residuals))
    max_residual = float(np.max(residuals))

    quadrant_counts = [0, 0, 0, 0]
    mid_x = result.crop.width / 2.0
    mid_y = result.crop.height / 2.0
    for x_value, y_value in zip(field_x, field_y, strict=True):
        quadrant = 0
        if x_value >= mid_x:
            quadrant += 1
        if y_value >= mid_y:
            quadrant += 2
        quadrant_counts[quadrant] += 1

    covered_quadrants = sum(1 for count in quadrant_counts if count > 0)
    spread_x = float(np.std(field_x) / max(result.crop.width, 1))
    spread_y = float(np.std(field_y) / max(result.crop.height, 1))

    accepted = (
        match_count >= 12
        and rms <= 6.0
        and median <= 3.0
        and max_residual <= 18.0
        and (
            covered_quadrants >= 3
            or (covered_quadrants >= 2 and min(spread_x, spread_y) >= 0.12)
        )
        and (spread_x + spread_y) >= 0.28
    )

    return {
        "accepted": accepted,
        "reason": "verified" if accepted else "insufficient-spread-or-match-quality",
        "match_count": match_count,
        "rms_px": round(rms, 3),
        "median_px": round(median, 3),
        "max_px": round(max_residual, 3),
        "covered_quadrants": covered_quadrants,
        "quadrant_counts": quadrant_counts,
        "spread_x": round(spread_x, 4),
        "spread_y": round(spread_y, 4),
    }


def compute_anchor_alignment(
    result: SolveResult,
    wcs: WCS,
    catalog: pd.DataFrame,
    star_names: dict[int, str],
    source_analysis: SourceAnalysis,
    image_width: int,
    image_height: int,
    magnitude_limit: float = 3.3,
    max_stars: int = 18,
) -> dict[str, Any]:
    candidate_hips = [hip for hip in star_names if hip in catalog.index]
    if not candidate_hips or not source_analysis.detections:
        return {
            "alignment_count": 0,
            "alignment_mean_px": 999.0,
            "alignment_median_px": 999.0,
            "alignment_p75_px": 999.0,
        }

    subset = catalog.loc[candidate_hips].copy()
    subset = subset[subset["magnitude"].notna() & (subset["magnitude"] <= magnitude_limit)].sort_values("magnitude")
    if subset.empty:
        return {
            "alignment_count": 0,
            "alignment_mean_px": 999.0,
            "alignment_median_px": 999.0,
            "alignment_p75_px": 999.0,
        }

    subset = subset.head(max_stars)
    projected_x, projected_y = project_points(
        wcs,
        subset["ra_degrees"].to_numpy(),
        subset["dec_degrees"].to_numpy(),
        result.crop,
    )

    anchors: list[tuple[float, float]] = []
    for x_value, y_value in zip(projected_x, projected_y, strict=True):
        x_float = float(x_value)
        y_float = float(y_value)
        if not (math.isfinite(x_float) and math.isfinite(y_float)):
            continue
        if not is_point_visible(x_float, y_float, image_width, image_height, margin=8.0):
            continue
        if not is_point_inside_crop(x_float, y_float, result.crop, margin=18.0):
            continue
        anchors.append((x_float, y_float))

    if not anchors:
        return {
            "alignment_count": 0,
            "alignment_mean_px": 999.0,
            "alignment_median_px": 999.0,
            "alignment_p75_px": 999.0,
        }

    detection_points = np.array(
        [
            (detection.x, detection.y)
            for detection in source_analysis.detections
            if detection.star_score >= 0.28 and is_point_inside_crop(detection.x, detection.y, result.crop, margin=12.0)
        ],
        dtype=np.float64,
    )
    if detection_points.size == 0:
        detection_points = np.array(
            [
                (detection.x, detection.y)
                for detection in source_analysis.detections
                if is_point_inside_crop(detection.x, detection.y, result.crop, margin=12.0)
            ],
            dtype=np.float64,
        )
    if detection_points.size == 0:
        return {
            "alignment_count": 0,
            "alignment_mean_px": 999.0,
            "alignment_median_px": 999.0,
            "alignment_p75_px": 999.0,
        }

    anchor_points = np.asarray(anchors, dtype=np.float64)
    deltas = anchor_points[:, None, :] - detection_points[None, :, :]
    nearest_distances = np.sqrt(np.min(np.sum(np.square(deltas), axis=2), axis=1))
    return {
        "alignment_count": int(nearest_distances.size),
        "alignment_mean_px": round(float(np.mean(nearest_distances)), 3),
        "alignment_median_px": round(float(np.median(nearest_distances)), 3),
        "alignment_p75_px": round(float(np.percentile(nearest_distances, 75)), 3),
    }


def enrich_solution_verification(
    result: SolveResult,
    verification: dict[str, Any],
    catalog: pd.DataFrame,
    star_names: dict[int, str],
    source_analysis: SourceAnalysis,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    enriched = dict(verification)
    wcs = load_wcs(result.wcs_path)
    enriched.update(
        compute_anchor_alignment(
            result,
            wcs,
            catalog,
            star_names,
            source_analysis,
            image_width,
            image_height,
        )
    )
    if enriched["alignment_count"] >= 8 and (
        float(enriched["alignment_mean_px"]) > 60.0 or float(enriched["alignment_p75_px"]) > 80.0
    ):
        enriched["accepted"] = False
        enriched["reason"] = "poor-anchor-alignment"
    return enriched


def is_strong_solution(result: SolveResult, image_width: int, image_height: int) -> bool:
    if result.verification is None or not result.verification.get("accepted"):
        return False
    area_ratio = (result.crop.width * result.crop.height) / max(image_width * image_height, 1)
    alignment_count = int(result.verification.get("alignment_count", 0))
    alignment_mean = float(result.verification.get("alignment_mean_px", 999.0))
    score = verification_score(result.verification, result.crop, image_width, image_height)
    return (
        (
            area_ratio >= 0.75
            and int(result.verification.get("match_count", 0)) >= 45
            and float(result.verification.get("rms_px", 99.0)) <= 3.0
            and float(result.verification.get("max_px", 99.0)) <= 12.0
            and int(result.verification.get("covered_quadrants", 0)) >= 4
            and (alignment_count < 3 or alignment_mean <= 36.0)
        )
        or (
            score >= 100.0
            and int(result.verification.get("match_count", 0)) >= 40
            and float(result.verification.get("rms_px", 99.0)) <= 3.6
            and float(result.verification.get("max_px", 99.0)) <= 10.0
            and int(result.verification.get("covered_quadrants", 0)) >= 4
            and (alignment_count < 3 or alignment_mean <= 24.0)
        )
    )


def estimate_scale_window(reference_result: SolveResult, target_crop: CropCandidate) -> tuple[float, float]:
    reference_metrics = compute_field_metrics(load_wcs(reference_result.wcs_path), reference_result.crop)
    reference_width = float(reference_metrics["field_width_deg"])
    scaled_width = reference_width * target_crop.width / max(reference_result.crop.width, 1)
    scale_low = max(1.0, min(120.0, scaled_width * 0.72))
    scale_high = max(scale_low + 1.0, min(120.0, scaled_width * 1.35))
    return round(scale_low, 3), round(scale_high, 3)


def solve_image(
    image: Image.Image,
    workdir: Path,
    index_dir: Path,
    catalog: pd.DataFrame,
    star_names: dict[int, str],
) -> tuple[SolveResult, list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    source_analysis = analyze_sources(image)
    accepted_results: list[SolveResult] = []
    scale_windows = [
        (20.0, 120.0),
        (5.0, 45.0),
        (1.0, 12.0),
    ]
    candidate_crops = build_crop_candidates(image.width, image.height, source_analysis)
    candidate_crops = [crop for crop in candidate_crops if crop.name == "full"] + [
        crop for crop in candidate_crops if crop.name != "full"
    ]

    if source_analysis.mode == "sep" and source_analysis.detections:
        for crop in candidate_crops:
            xylist_path, source_count = write_xylist(source_analysis, crop, workdir)
            if xylist_path is None:
                attempts.append(
                    {
                        "input_mode": "xyls",
                        "crop": crop.name,
                        "x": crop.x,
                        "y": crop.y,
                        "width": crop.width,
                        "height": crop.height,
                        "status": "skipped-insufficient-sources",
                        "source_count": source_count,
                    }
                )
                continue

            for scale_low, scale_high in scale_windows:
                result = run_solve_on_xylist(xylist_path, crop, scale_low, scale_high, workdir, index_dir)
                verification = verify_solution(result) if result else None
                if result and verification and verification["accepted"]:
                    verification = enrich_solution_verification(
                        result,
                        verification,
                        catalog,
                        star_names,
                        source_analysis,
                        image.width,
                        image.height,
                    )
                attempts.append(
                    {
                        "input_mode": "xyls",
                        "crop": crop.name,
                        "x": crop.x,
                        "y": crop.y,
                        "width": crop.width,
                        "height": crop.height,
                        "downsample": 1,
                        "scale_low_deg": scale_low,
                        "scale_high_deg": scale_high,
                        "source_count": source_count,
                        "status": "solved" if result and verification and verification["accepted"] else ("rejected" if result else "failed"),
                        "verification": verification,
                    }
                )
                if result and verification and verification["accepted"]:
                    result.verification = verification
                    accepted_results.append(result)
                    if is_strong_solution(result, image.width, image.height):
                        result.verification["candidate_score"] = round(
                            verification_score(result.verification, result.crop, image.width, image.height),
                            3,
                        )
                        return result, attempts, source_analysis.diagnostics
                    break

    xyls_results = [result for result in accepted_results if result.input_mode == "xyls"]
    if xyls_results:
        best_xyls = max(
            xyls_results,
            key=lambda item: verification_score(item.verification or {}, item.crop, image.width, image.height),
        )
        if is_strong_solution(best_xyls, image.width, image.height):
            best_xyls.verification["candidate_score"] = round(
                verification_score(best_xyls.verification, best_xyls.crop, image.width, image.height),
                3,
            )
            return best_xyls, attempts, source_analysis.diagnostics

    fallback_crops = candidate_crops
    fallback_scale_windows = scale_windows
    if xyls_results:
        best_xyls = max(
            xyls_results,
            key=lambda item: verification_score(item.verification or {}, item.crop, image.width, image.height),
        )
        fallback_names = [best_xyls.crop.name]
        if best_xyls.crop.name != "full":
            fallback_names.append("full")
        fallback_crops = [crop for crop in candidate_crops if crop.name in fallback_names]
        fallback_scale_windows = [estimate_scale_window(best_xyls, crop) for crop in fallback_crops]

    for crop_index, crop in enumerate(fallback_crops):
        crop_path = save_crop(image, crop, workdir)
        crop_accepted = False
        crop_scale_windows = (
            [fallback_scale_windows[crop_index]]
            if xyls_results and crop_index < len(fallback_scale_windows)
            else scale_windows
        )
        for scale_low, scale_high in crop_scale_windows:
            for downsample in (2, 4, 1):
                result = run_solve_on_image(crop_path, crop, downsample, scale_low, scale_high, workdir, index_dir)
                verification = verify_solution(result) if result else None
                if result and verification and verification["accepted"]:
                    verification = enrich_solution_verification(
                        result,
                        verification,
                        catalog,
                        star_names,
                        source_analysis,
                        image.width,
                        image.height,
                    )
                attempts.append(
                    {
                        "input_mode": "image",
                        "crop": crop.name,
                        "x": crop.x,
                        "y": crop.y,
                        "width": crop.width,
                        "height": crop.height,
                        "downsample": downsample,
                        "scale_low_deg": scale_low,
                        "scale_high_deg": scale_high,
                        "status": "solved" if result and verification and verification["accepted"] else ("rejected" if result else "failed"),
                        "verification": verification,
                    }
                )
                if result and verification and verification["accepted"]:
                    result.verification = verification
                    accepted_results.append(result)
                    crop_accepted = True
                    break
            if crop_accepted:
                break

    if accepted_results:
        best_result = max(
            accepted_results,
            key=lambda item: verification_score(item.verification or {}, item.crop, image.width, image.height),
        )
        if best_result.verification is not None:
            best_result.verification["candidate_score"] = round(
                verification_score(best_result.verification, best_result.crop, image.width, image.height),
                3,
            )
        return best_result, attempts, source_analysis.diagnostics

    raise RuntimeError("plate solving failed for all full-image and scored-crop attempts")


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

    magnitude_limit = float(overlay_detail_value(overlay_options, "star_magnitude_limit"))
    max_labels = int(overlay_detail_value(overlay_options, "star_label_limit"))
    bright_star_separation = float(overlay_detail_value(overlay_options, "star_bright_separation"))
    dim_star_separation = float(overlay_detail_value(overlay_options, "star_dim_separation"))

    subset = catalog.loc[candidate_hips].copy()
    subset = subset[subset["magnitude"] <= magnitude_limit].sort_values("magnitude")
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
        if not is_point_visible(float(x_value), float(y_value), image_width, image_height):
            continue
        if not is_point_inside_crop(float(x_value), float(y_value), crop, margin=12.0):
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
    for constellation in constellations:
        visible_segments: list[dict[str, Any]] = []
        label_points: list[tuple[float, float]] = []
        onscreen_segments = 0
        segment_keys: set[tuple[tuple[float, float], tuple[float, float]]] = set()

        for polyline in constellation.get("lines", []):
            for start_hip, end_hip in pairwise(polyline):
                if start_hip not in catalog.index or end_hip not in catalog.index:
                    continue
                start = catalog.loc[start_hip]
                end = catalog.loc[end_hip]
                segment_key = build_segment_key(
                    float(start["ra_degrees"]),
                    float(start["dec_degrees"]),
                    float(end["ra_degrees"]),
                    float(end["dec_degrees"]),
                )
                if segment_key in segment_keys:
                    continue
                x_values, y_values = project_points(
                    wcs,
                    np.array([start["ra_degrees"], end["ra_degrees"]]),
                    np.array([start["dec_degrees"], end["dec_degrees"]]),
                    crop,
                )
                start_x, end_x = float(x_values[0]), float(x_values[1])
                start_y, end_y = float(y_values[0]), float(y_values[1])
                if not all(math.isfinite(value) for value in (start_x, start_y, end_x, end_y)):
                    continue
                if not segment_intersects_crop(start_x, start_y, end_x, end_y, crop, margin=36.0):
                    continue
                if is_projected_segment_duplicate(visible_segments, start_x, start_y, end_x, end_y, duplicate_tolerance):
                    continue

                segment_keys.add(segment_key)

                if segment_intersects_crop(start_x, start_y, end_x, end_y, crop):
                    onscreen_segments += 1

                visible_segments.append(
                    {
                        "start": {"x": start_x, "y": start_y, "hip": int(start_hip)},
                        "end": {"x": end_x, "y": end_y, "hip": int(end_hip)},
                    }
                )

                if is_point_inside_crop(start_x, start_y, crop):
                    label_points.append((start_x, start_y))
                if is_point_inside_crop(end_x, end_y, crop):
                    label_points.append((end_x, end_y))

        for polyline in constellation.get("coord_lines", []):
            ra_values = np.array([point["ra_degrees"] for point in polyline])
            dec_values = np.array([point["dec_degrees"] for point in polyline])
            x_values, y_values = project_points(wcs, ra_values, dec_values, crop)
            projected_points = list(zip(polyline, x_values, y_values, strict=True))
            for (start_point, start_x_raw, start_y_raw), (end_point, end_x_raw, end_y_raw) in pairwise(projected_points):
                start_x = float(start_x_raw)
                start_y = float(start_y_raw)
                end_x = float(end_x_raw)
                end_y = float(end_y_raw)
                if not all(math.isfinite(value) for value in (start_x, start_y, end_x, end_y)):
                    continue

                segment_key = build_segment_key(
                    float(start_point["ra_degrees"]),
                    float(start_point["dec_degrees"]),
                    float(end_point["ra_degrees"]),
                    float(end_point["dec_degrees"]),
                )
                if segment_key in segment_keys:
                    continue
                if not segment_intersects_crop(start_x, start_y, end_x, end_y, crop, margin=36.0):
                    continue
                if is_projected_segment_duplicate(visible_segments, start_x, start_y, end_x, end_y, duplicate_tolerance):
                    continue

                segment_keys.add(segment_key)
                if segment_intersects_crop(start_x, start_y, end_x, end_y, crop):
                    onscreen_segments += 1

                visible_segments.append(
                    {
                        "start": {"x": start_x, "y": start_y},
                        "end": {"x": end_x, "y": end_y},
                    }
                )

                if is_point_inside_crop(start_x, start_y, crop):
                    label_points.append((start_x, start_y))
                if is_point_inside_crop(end_x, end_y, crop):
                    label_points.append((end_x, end_y))

        if not visible_segments:
            continue
        if len(label_points) < 2 and onscreen_segments == 0:
            continue

        if constellation.get("label_ra_degrees") is not None and constellation.get("label_dec_degrees") is not None:
            label_x_values, label_y_values = project_points(
                wcs,
                np.array([constellation["label_ra_degrees"]]),
                np.array([constellation["label_dec_degrees"]]),
                crop,
            )
            explicit_label_x = float(label_x_values[0])
            explicit_label_y = float(label_y_values[0])
            if math.isfinite(explicit_label_x) and math.isfinite(explicit_label_y) and is_point_visible(
                explicit_label_x,
                explicit_label_y,
                image_width,
                image_height,
                margin=48.0,
            ):
                if is_point_inside_crop(explicit_label_x, explicit_label_y, crop, margin=48.0):
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

    candidates: list[dict[str, Any]] = []
    for item in deep_sky_objects:
        if not is_interesting_dso(item, overlay_options):
            continue
        if item["magnitude"] is not None and item["magnitude"] > dso_magnitude_limit and not item["messier"] and not item["common_name"]:
            continue
        x_values, y_values = project_points(
            wcs,
            np.array([item["ra_degrees"]]),
            np.array([item["dec_degrees"]]),
            crop,
        )
        x_value = float(x_values[0])
        y_value = float(y_values[0])
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        if not is_point_visible(x_value, y_value, image_width, image_height, margin=28.0):
            continue
        if not is_point_inside_crop(x_value, y_value, crop, margin=28.0):
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


def draw_dso_marker(
    draw: ImageDraw.ImageDraw,
    item: dict[str, Any],
    radius: int,
    line_width: int,
) -> tuple[int, int, int, int]:
    x_value = item["x"]
    y_value = item["y"]
    marker, color = dso_style(item)
    return draw_dso_marker_primitive(draw, marker, x_value, y_value, radius, line_width, color)


def draw_dso_marker_primitive(
    draw: ImageDraw.ImageDraw,
    marker: str,
    x_value: float,
    y_value: float,
    radius: int,
    line_width: int,
    color: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    bounds = (
        int(round(x_value - radius)),
        int(round(y_value - radius)),
        int(round(x_value + radius)),
        int(round(y_value + radius)),
    )
    width = max(1, line_width)

    if marker == "square":
        draw.rectangle(bounds, outline=color, width=width)
    elif marker == "crossed_circle":
        draw.ellipse(bounds, outline=color, width=width)
        draw.line((x_value - radius, y_value, x_value + radius, y_value), fill=color, width=width)
        draw.line((x_value, y_value - radius, x_value, y_value + radius), fill=color, width=width)
    elif marker == "ring":
        draw.ellipse(bounds, outline=color, width=width)
        inner = max(2, radius // 2)
        draw.ellipse(
            (
                x_value - inner,
                y_value - inner,
                x_value + inner,
                y_value + inner,
            ),
            outline=color,
            width=width,
        )
    elif marker == "x_circle":
        draw.ellipse(bounds, outline=color, width=width)
        draw.line((x_value - radius, y_value - radius, x_value + radius, y_value + radius), fill=color, width=width)
        draw.line((x_value - radius, y_value + radius, x_value + radius, y_value - radius), fill=color, width=width)
    elif marker == "hexagon":
        vertical = radius * 0.86
        horizontal = radius * 0.5
        points = [
            (x_value - horizontal, y_value - vertical),
            (x_value + horizontal, y_value - vertical),
            (x_value + radius, y_value),
            (x_value + horizontal, y_value + vertical),
            (x_value - horizontal, y_value + vertical),
            (x_value - radius, y_value),
        ]
        draw.polygon(points, outline=color, width=width)
    elif marker == "diamond":
        points = [
            (x_value, y_value - radius),
            (x_value + radius, y_value),
            (x_value, y_value + radius),
            (x_value - radius, y_value),
        ]
        draw.polygon(points, outline=color, width=width)
    else:
        draw.ellipse(bounds, outline=color, width=width)

    return bounds


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


def draw_label_leader(
    draw: ImageDraw.ImageDraw,
    anchor_x: float,
    anchor_y: float,
    label_position: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int, int],
    stroke_width: int = 2,
) -> None:
    segment = compute_label_leader_segment(
        draw,
        anchor_x,
        anchor_y,
        label_position,
        text,
        font,
        stroke_width=stroke_width,
    )
    if segment is None:
        return
    draw.line(segment, fill=color, width=1)


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


def scale_overlay_scene(scene: dict[str, Any], scale: int) -> dict[str, Any]:
    if scale == 1:
        return deepcopy(scene)

    scaled = deepcopy(scene)
    scaled["image_width"] = int(round(float(scaled["image_width"]) * scale))
    scaled["image_height"] = int(round(float(scaled["image_height"]) * scale))

    crop = scaled["crop"]
    for key in ("x", "y", "width", "height"):
        crop[key] = int(round(float(crop[key]) * scale))

    for key in ("left", "top", "right", "bottom"):
        scaled["bounds"][key] = float(scaled["bounds"][key]) * scale

    for line in scaled["constellation_lines"]:
        for key in ("x1", "y1", "x2", "y2"):
            line[key] = float(line[key]) * scale
        line["line_width"] = max(1, int(round(float(line["line_width"]) * scale)))

    for marker in scaled["deep_sky_markers"]:
        for key in ("x", "y"):
            marker[key] = float(marker[key]) * scale
        marker["radius"] = max(1, int(round(float(marker["radius"]) * scale)))
        marker["line_width"] = max(1, int(round(float(marker["line_width"]) * scale)))

    for marker in scaled["star_markers"]:
        for key in ("x", "y"):
            marker[key] = float(marker[key]) * scale
        marker["radius"] = max(1, int(round(float(marker["radius"]) * scale)))

    for key in ("deep_sky_labels", "constellation_labels", "star_labels"):
        for label in scaled[key]:
            for coord in ("x", "y"):
                label[coord] = float(label[coord]) * scale
            label["font_size"] = max(1, int(round(float(label["font_size"]) * scale)))
            label["stroke_width"] = max(1, int(round(float(label["stroke_width"]) * scale)))
            if label.get("leader"):
                for coord in ("x1", "y1", "x2", "y2"):
                    label["leader"][coord] = float(label["leader"][coord]) * scale
                label["leader"]["line_width"] = max(1, int(round(float(label["leader"]["line_width"]) * scale)))

    return scaled


def render_overlay_scene_rgba(image_size: tuple[int, int], scene: dict[str, Any]) -> Image.Image:
    overlay = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_cache: dict[int, ImageFont.ImageFont] = {}

    def font_for_size(size: int) -> ImageFont.ImageFont:
        normalized_size = max(1, int(size))
        cached = font_cache.get(normalized_size)
        if cached is None:
            cached = load_font(normalized_size)
            font_cache[normalized_size] = cached
        return cached

    for line in scene["constellation_lines"]:
        draw.line(
            (line["x1"], line["y1"], line["x2"], line["y2"]),
            fill=tuple(line["rgba"]),
            width=int(line["line_width"]),
        )

    for marker in scene["deep_sky_markers"]:
        draw_dso_marker_primitive(
            draw,
            marker["marker"],
            marker["x"],
            marker["y"],
            int(marker["radius"]),
            int(marker["line_width"]),
            tuple(marker["rgba"]),
        )

    for marker in scene["star_markers"]:
        draw.ellipse(
            (
                marker["x"] - marker["radius"],
                marker["y"] - marker["radius"],
                marker["x"] + marker["radius"],
                marker["y"] + marker["radius"],
            ),
            fill=tuple(marker["fill_rgba"]),
            outline=tuple(marker["outline_rgba"]),
        )

    for collection_name in ("deep_sky_labels", "constellation_labels", "star_labels"):
        for label in scene[collection_name]:
            leader = label.get("leader")
            if leader:
                draw.line(
                    (leader["x1"], leader["y1"], leader["x2"], leader["y2"]),
                    fill=tuple(leader["rgba"]),
                    width=int(leader["line_width"]),
                )
            draw.text(
                (label["x"], label["y"]),
                label["text"],
                font=font_for_size(label["font_size"]),
                fill=tuple(label["text_rgba"]),
                stroke_width=int(label["stroke_width"]),
                stroke_fill=tuple(label["stroke_rgba"]),
            )

    return overlay


def render_overlay_scene(base_image: Image.Image, overlay_scene: dict[str, Any]) -> Image.Image:
    supersample = overlay_supersample_scale(base_image.width, base_image.height)

    if supersample > 1:
        overlay = render_overlay_scene_rgba(
            (base_image.width * supersample, base_image.height * supersample),
            scale_overlay_scene(overlay_scene, supersample),
        ).resize(base_image.size, LANCZOS_RESAMPLING)
    else:
        overlay = render_overlay_scene_rgba(base_image.size, overlay_scene)

    return Image.alpha_composite(base_image.copy().convert("RGBA"), overlay).convert("RGB")


def render_overlay_rgba(
    image_size: tuple[int, int],
    constellations: list[dict[str, Any]],
    named_stars: list[dict[str, Any]],
    deep_sky_objects: list[dict[str, Any]],
    crop: CropCandidate,
    overlay_options: dict[str, Any],
) -> Image.Image:
    image_width, image_height = image_size
    overlay = Image.new("RGBA", image_size, (0, 0, 0, 0))
    line_overlay = Image.new("RGBA", image_size, (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(line_overlay)

    min_dimension = min(image_width, image_height)
    line_width = max(1, min_dimension // 600)
    render_bounds = crop_bounds(crop)

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
                line_draw.line(
                    clipped_segment,
                    fill=line_color,
                    width=line_width,
                )

    overlay = Image.alpha_composite(overlay, line_overlay)
    draw = ImageDraw.Draw(overlay)
    occupied_boxes: list[tuple[float, float, float, float]] = []

    constellation_font = load_font(max(18, min_dimension // 52))
    dso_font = load_font(max(14, min_dimension // 74))
    star_font = load_font(max(12, min_dimension // 84))
    dso_radius = max(4, min_dimension // 250)
    star_radius = max(2, min_dimension // 320)

    show_dso_markers = overlay_layer_enabled(overlay_options, "deep_sky_markers")
    show_dso_labels = overlay_layer_enabled(overlay_options, "deep_sky_labels")
    show_constellation_labels = overlay_layer_enabled(overlay_options, "constellation_labels")
    show_contextual_labels = overlay_layer_enabled(overlay_options, "contextual_constellation_labels")
    show_star_markers = overlay_layer_enabled(overlay_options, "star_markers")
    show_star_labels = overlay_layer_enabled(overlay_options, "star_labels")
    show_label_leaders = overlay_layer_enabled(overlay_options, "label_leaders")

    for item in deep_sky_objects:
        if show_dso_markers:
            draw_dso_marker(draw, item, dso_radius, line_width)
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
        if show_label_leaders:
            draw_label_leader(
                draw,
                item["x"],
                item["y"],
                position,
                item["display_label"],
                dso_font,
                (165, 220, 255, 190),
                stroke_width=2,
            )
        draw.text(
            position,
            item["display_label"],
            font=dso_font,
            fill=(242, 246, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 220),
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
            draw.text(
                position,
                constellation["display_name"],
                font=constellation_font,
                fill=(225, 232, 245, 255),
                stroke_width=3,
                stroke_fill=(0, 0, 0, 230),
            )

    for star in named_stars:
        if show_star_markers:
            draw.ellipse(
                (
                    star["x"] - star_radius,
                    star["y"] - star_radius,
                    star["x"] + star_radius,
                    star["y"] + star_radius,
                ),
                fill=(255, 210, 150, 215),
                outline=(255, 255, 255, 210),
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
        if show_label_leaders:
            draw_label_leader(
                draw,
                star["x"],
                star["y"],
                position,
                star["name"],
                star_font,
                (255, 233, 188, 176),
                stroke_width=2,
            )
        draw.text(
            position,
            star["name"],
            font=star_font,
            fill=(250, 244, 236, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 220),
        )

    return overlay


try:
    LANCZOS_RESAMPLING = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - Pillow compatibility
    LANCZOS_RESAMPLING = Image.LANCZOS


def render_overlay(
    base_image: Image.Image,
    constellations: list[dict[str, Any]],
    named_stars: list[dict[str, Any]],
    deep_sky_objects: list[dict[str, Any]],
    crop: CropCandidate,
    overlay_options: dict[str, Any],
) -> Image.Image:
    supersample = overlay_supersample_scale(base_image.width, base_image.height)

    if supersample > 1:
        overlay = render_overlay_rgba(
            (base_image.width * supersample, base_image.height * supersample),
            scale_constellation_overlays(constellations, supersample),
            scale_positioned_overlay_items(named_stars, supersample),
            scale_positioned_overlay_items(deep_sky_objects, supersample),
            scale_crop_candidate(crop, supersample),
            overlay_options,
        ).resize(base_image.size, LANCZOS_RESAMPLING)
    else:
        overlay = render_overlay_rgba(
            base_image.size,
            constellations,
            named_stars,
            deep_sky_objects,
            crop,
            overlay_options,
        )

    return Image.alpha_composite(base_image.copy().convert("RGBA"), overlay).convert("RGB")


def summarize_solver_output(stdout: str, stderr: str) -> str:
    combined = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
    lines = [line for line in combined.splitlines() if line.strip()]
    return "\n".join(lines[-25:])


def annotate_image(
    input_path: Path,
    index_dir: Path,
    catalog: pd.DataFrame,
    constellations: list[dict[str, Any]],
    deep_sky_objects: list[dict[str, Any]],
    star_names: dict[int, str],
    overlay_options: dict[str, Any],
    localization: LocalizationBundle | None = None,
    output_image_path: Path | None = None,
) -> dict[str, Any]:
    total_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="star-annotator-") as tempdir:
        workdir = Path(tempdir)
        base_image: Image.Image | None = None
        annotated_image: Image.Image | None = None
        try:
            normalize_start = time.perf_counter()
            base_image, _ = normalize_image(input_path, workdir)
            normalize_ms = (time.perf_counter() - normalize_start) * 1000.0

            solve_start = time.perf_counter()
            solve_result, attempts, source_analysis = solve_image(base_image, workdir, index_dir, catalog, star_names)
            solve_ms = (time.perf_counter() - solve_start) * 1000.0

            scene_start = time.perf_counter()
            wcs = load_wcs(solve_result.wcs_path)

            named_stars = collect_named_stars(
                catalog,
                star_names,
                wcs,
                solve_result.crop,
                base_image.width,
                base_image.height,
                overlay_options,
            )
            visible_constellations = collect_constellations(
                catalog,
                constellations,
                wcs,
                solve_result.crop,
                base_image.width,
                base_image.height,
                overlay_options,
            )
            visible_deep_sky_objects = collect_deep_sky_objects(
                deep_sky_objects,
                wcs,
                solve_result.crop,
                base_image.width,
                base_image.height,
                overlay_options,
            )
            visible_constellations = add_contextual_constellation_labels(
                visible_constellations,
                visible_deep_sky_objects,
                {item["abbr"]: item for item in constellations},
                overlay_options,
            )
            scene_ms = (time.perf_counter() - scene_start) * 1000.0

            overlay_scene_start = time.perf_counter()
            overlay_scene = build_overlay_scene(
                base_image.size,
                visible_constellations,
                named_stars,
                visible_deep_sky_objects,
                solve_result.crop,
                overlay_options,
            )
            overlay_scene_ms = (time.perf_counter() - overlay_scene_start) * 1000.0

            render_ms = 0.0
            if output_image_path is not None:
                render_start = time.perf_counter()
                annotated_image = render_overlay_scene(base_image, overlay_scene)
                annotated_image.save(output_image_path)
                render_ms = (time.perf_counter() - render_start) * 1000.0

            return {
                "input_image": str(input_path),
                "output_image": str(output_image_path) if output_image_path is not None else None,
                "image_width": base_image.width,
                "image_height": base_image.height,
                "solve": compute_field_metrics(wcs, solve_result.crop),
                "solve_verification": solve_result.verification,
                "attempts": attempts,
                "source_analysis": source_analysis,
                "localization": {
                    "requested_locale": localization.requested_locale if localization is not None else "en",
                    "resolved_locale": localization.resolved_locale if localization is not None else "en",
                    "available_locales": localization.available_locales if localization is not None else ["en"],
                },
                "visible_named_stars": named_stars,
                "visible_constellations": visible_constellations,
                "visible_deep_sky_objects": visible_deep_sky_objects,
                "render_options": overlay_options,
                "overlay_scene": overlay_scene,
                "solver_log_tail": summarize_solver_output(solve_result.stdout, solve_result.stderr),
                "timings_ms": {
                    "normalize": round(normalize_ms, 2),
                    "solve": round(solve_ms, 2),
                    "scene": round(scene_ms, 2),
                    "overlay_scene": round(overlay_scene_ms, 2),
                    "render": round(render_ms, 2),
                    "total": round((time.perf_counter() - total_start) * 1000.0, 2),
                },
            }
        finally:
            if annotated_image is not None:
                annotated_image.close()
            if base_image is not None:
                base_image.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_image_path = Path(args.output_image).resolve() if args.output_image else None
    output_json_path = Path(args.output_json).resolve()
    index_dir = Path(args.index_dir).resolve()
    catalog_path = Path(args.catalog).resolve()
    constellation_paths = [Path(entry).resolve() for entry in args.constellations]
    star_names_path = Path(args.star_names).resolve()
    dso_paths = [Path(entry).resolve() for entry in args.dso_catalog]
    localization_paths = [Path(entry).resolve() for entry in args.localization]
    locale = canonicalize_locale_tag(args.locale)
    supplemental_dso_path = Path(args.supplemental_dso).resolve() if args.supplemental_dso else None
    overlay_options = parse_overlay_options(args.options_json)

    if output_image_path is not None:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    localization_bundle = load_localized_names(localization_paths, locale)
    star_names = load_star_names(star_names_path, localization_bundle.strings)
    constellations = load_constellations(constellation_paths, localization_bundle.strings)
    constellation_name_map = build_constellation_name_map(constellations)
    deep_sky_objects = load_deep_sky_objects(
        dso_paths,
        constellation_name_map,
        localization_bundle.strings,
        supplemental_dso_path,
    )
    required_hips = collect_required_hips(constellations, star_names)
    catalog = load_catalog(catalog_path, required_hips)
    result = annotate_image(
        input_path=input_path,
        index_dir=index_dir,
        catalog=catalog,
        constellations=constellations,
        deep_sky_objects=deep_sky_objects,
        star_names=star_names,
        overlay_options=overlay_options,
        localization=localization_bundle,
        output_image_path=output_image_path,
    )
    output_json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
