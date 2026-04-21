#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skyfield.data import stellarium

from annotate_localization import (
    SUPPLEMENTAL_CONSTELLATION_ABBR_OVERRIDES,
    normalize_constellation_key,
    normalize_human_alias,
    resolve_constellation_display_name,
)


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

        label_ra_degrees = float(location["right_ascension"]) if "right_ascension" in location else None
        label_dec_degrees = float(location["declination"]) if "declination" in location else None

        parsed_sources.append(
            {
                "name_key": raw_name_key,
                "english_name": english_name.title(),
                "label_ra_degrees": label_ra_degrees,
                "label_dec_degrees": label_dec_degrees,
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
