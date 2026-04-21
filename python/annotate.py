#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Importing annotate_image_ops sets Image.MAX_IMAGE_PIXELS and registers the
# FITSFixedWarning filter, preserving the side effects of the original module.
import annotate_image_ops  # noqa: F401

from annotate_catalog import load_catalog
from annotate_constellations import (
    build_constellation_name_map,
    collect_required_hips,
    load_constellations,
    load_star_names,
)
from annotate_deep_sky import load_deep_sky_objects
from annotate_geometry import compute_field_metrics, load_wcs
from annotate_image_ops import normalize_image
from annotate_localization import canonicalize_locale_tag, load_localized_names
from annotate_options import parse_overlay_options
from annotate_render import render_overlay_scene
from annotate_scene import (
    add_contextual_constellation_labels,
    build_overlay_scene,
    collect_constellations,
    collect_deep_sky_objects,
    collect_named_stars,
)
from annotate_sky_mask import (
    compute_sky_mask,
    filter_constellations as filter_constellations_by_sky_mask,
    filter_deep_sky_objects as filter_dsos_by_sky_mask,
    filter_named_stars as filter_named_stars_by_sky_mask,
    mask_is_trustworthy,
)
from annotate_solving import solve_image, summarize_solver_output
from annotate_types import LocalizationBundle


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
        base_image = None
        annotated_image = None
        try:
            normalize_start = time.perf_counter()
            base_image, _ = normalize_image(input_path, workdir)
            normalize_ms = (time.perf_counter() - normalize_start) * 1000.0

            # Sky mask runs *before* solve so it can clean ground-pixel
            # detections out of sep's output — this keeps the full-image
            # solve's RMS competitive with a clean sub-crop's, which in turn
            # lets the full-image candidate win verification_score and avoid
            # the truncated-render pathology on foreground-heavy images.
            sky_mask_start = time.perf_counter()
            sky_mask = None
            if bool(overlay_options.get("mask_foreground", True)):
                sky_mask = compute_sky_mask(base_image)
            sky_mask_ms = (time.perf_counter() - sky_mask_start) * 1000.0

            solve_start = time.perf_counter()
            solve_result, attempts, source_analysis = solve_image(
                base_image,
                workdir,
                index_dir,
                catalog,
                star_names,
                sky_mask=sky_mask,
            )
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

            # Validate the mask against plate-solved star positions before
            # applying it to the scene. If the model hallucinated a horizon
            # (common on pure night-sky images with vignetting), too many
            # real stars land on "ground" pixels and we drop the mask.
            mask_requested = bool(overlay_options.get("mask_foreground", True))
            mask_reason: str
            if sky_mask is None:
                mask_reason = "not_requested" if not mask_requested else "model_unavailable"
            else:
                star_positions = [(star["x"], star["y"]) for star in named_stars]
                if not mask_is_trustworthy(sky_mask, star_positions):
                    sky_mask = None
                    mask_reason = "untrustworthy"
                else:
                    mask_reason = "applied"
            if sky_mask is not None:
                named_stars = filter_named_stars_by_sky_mask(named_stars, sky_mask)
                visible_deep_sky_objects = filter_dsos_by_sky_mask(
                    visible_deep_sky_objects, sky_mask
                )
                visible_constellations = filter_constellations_by_sky_mask(
                    visible_constellations, sky_mask
                )

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
                "sky_mask_status": {
                    "requested": mask_requested,
                    "applied": sky_mask is not None,
                    "reason": mask_reason,
                },
                "overlay_scene": overlay_scene,
                "solver_log_tail": summarize_solver_output(solve_result.stdout, solve_result.stderr),
                "timings_ms": {
                    "normalize": round(normalize_ms, 2),
                    "solve": round(solve_ms, 2),
                    "scene": round(scene_ms, 2),
                    "sky_mask": round(sky_mask_ms, 2),
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
