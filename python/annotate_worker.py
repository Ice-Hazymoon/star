#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import annotate


def parse_asset_cache_limit() -> int:
    raw_value = os.environ.get("ANNOTATION_WORKER_ASSET_CACHE_SIZE", "4").strip()
    try:
        return max(1, min(16, int(raw_value)))
    except ValueError:
        return 4


MAX_ASSET_CACHE_SIZE = parse_asset_cache_limit()
ASSET_CACHE: OrderedDict[tuple[str, ...], dict[str, Any]] = OrderedDict()


def load_assets(
    catalog_path: Path,
    constellation_paths: list[Path],
    star_names_path: Path,
    dso_paths: list[Path],
    localization_paths: list[Path],
    locale: str,
    supplemental_dso_path: Path | None,
) -> dict[str, Any]:
    cache_key = (
        str(catalog_path),
        *(str(path) for path in constellation_paths),
        str(star_names_path),
        *(str(path) for path in dso_paths),
        *(str(path) for path in localization_paths),
        locale,
        str(supplemental_dso_path) if supplemental_dso_path is not None else "",
    )
    cached = ASSET_CACHE.get(cache_key)
    if cached is not None:
        ASSET_CACHE.move_to_end(cache_key)
        return cached

    localization_bundle = annotate.load_localized_names(localization_paths, locale)
    star_names = annotate.load_star_names(star_names_path, localization_bundle.strings)
    constellations = annotate.load_constellations(constellation_paths, localization_bundle.strings)
    constellation_name_map = annotate.build_constellation_name_map(constellations)
    deep_sky_objects = annotate.load_deep_sky_objects(
        dso_paths,
        constellation_name_map,
        localization_bundle.strings,
        supplemental_dso_path,
    )
    required_hips = annotate.collect_required_hips(constellations, star_names)
    catalog = annotate.load_catalog(catalog_path, required_hips)

    assets = {
        "catalog": catalog,
        "constellations": constellations,
        "deep_sky_objects": deep_sky_objects,
        "star_names": star_names,
        "localization": localization_bundle,
    }
    ASSET_CACHE[cache_key] = assets
    ASSET_CACHE.move_to_end(cache_key)
    while len(ASSET_CACHE) > MAX_ASSET_CACHE_SIZE:
        ASSET_CACHE.popitem(last=False)
    return assets


def process_job(payload: dict[str, Any]) -> dict[str, Any]:
    action = payload.get("action", "annotate")
    if action == "ping":
        return {"status": "ok"}

    index_dir = Path(payload["index_dir"]).resolve()
    catalog_path = Path(payload["catalog_path"]).resolve()
    constellation_paths = [Path(entry).resolve() for entry in payload["constellation_paths"]]
    star_names_path = Path(payload["star_names_path"]).resolve()
    dso_paths = [Path(entry).resolve() for entry in payload["dso_paths"]]
    localization_paths = [Path(entry).resolve() for entry in payload.get("localization_paths", [])]
    locale = annotate.canonicalize_locale_tag(payload.get("locale"))
    supplemental_dso_path = (
        Path(payload["supplemental_dso_path"]).resolve()
        if payload.get("supplemental_dso_path")
        else None
    )

    assets = load_assets(
        catalog_path,
        constellation_paths,
        star_names_path,
        dso_paths,
        localization_paths,
        locale,
        supplemental_dso_path,
    )

    if action == "preload":
        return {
            "status": "ok",
            "catalog_rows": int(len(assets["catalog"])),
            "constellation_count": int(len(assets["constellations"])),
            "deep_sky_object_count": int(len(assets["deep_sky_objects"])),
            "star_name_count": int(len(assets["star_names"])),
        }

    input_path = Path(payload["input_path"]).resolve()
    output_image_path = Path(payload["output_image_path"]).resolve() if payload.get("output_image_path") else None
    overlay_options = payload.get("overlay_options") or annotate.parse_overlay_options("")

    return annotate.annotate_image(
        input_path=input_path,
        index_dir=index_dir,
        catalog=assets["catalog"],
        constellations=assets["constellations"],
        deep_sky_objects=assets["deep_sky_objects"],
        star_names=assets["star_names"],
        overlay_options=overlay_options,
        localization=assets["localization"],
        output_image_path=output_image_path,
    )


def main() -> None:
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        request_id: str | None = None
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                request_id = payload.get("id")
            response = {
                "id": request_id,
                "ok": True,
                "result": process_job(payload),
            }
        except Exception as exc:  # pragma: no cover - worker protocol
            response = {
                "id": request_id,
                "ok": False,
                "error": str(exc),
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
