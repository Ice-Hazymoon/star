#!/usr/bin/env python3
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


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
    "mask_foreground": True,
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
    options["mask_foreground"] = bool(options.get("mask_foreground", True))
    return options
