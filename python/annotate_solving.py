#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

# solve-field's --cpulimit is CPU-time only; on a loaded system wall time can
# run 2-3× longer. Use these as wall-clock hard stops on each subprocess and a
# total budget for the whole solve loop so an un-solvable image can't lock up
# the worker indefinitely.
#
# Empirical distribution of per-attempt wall time (profiled on samples/):
#   successful solve:                        200ms – 1.3s
#   quick mismatch (wide/medium scale):      700ms – 2.1s
#   pathological mismatch (narrow scale on   20s   – 60s
#     a wide-field image — see orion):
# We size wall timeouts to cut the pathological tail without threatening the
# real-solve band; --cpulimit on solve-field produces a graceful early exit,
# and the Python wall timeout is a safety net for the case where solve-field
# doesn't honour its own limit quickly enough.
XYLIST_SUBPROCESS_TIMEOUT_S = 15.0
IMAGE_SUBPROCESS_TIMEOUT_S = 20.0
SOLVE_TIME_BUDGET_S = 30.0


class SolveTimeoutError(RuntimeError):
    """Raised when the solve loop exhausts its total time budget."""

from annotate_geometry import (
    compute_field_metrics,
    is_point_inside_crop,
    is_point_visible,
    load_wcs,
    project_points,
)
from annotate_image_ops import (
    analyze_sources,
    build_crop_candidates,
    save_crop,
)
from annotate_types import CropCandidate, SolveResult, SourceAnalysis, SourceDetection


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
    # Saturate rms / max-px penalties: when a foreground-noisy full-image solve
    # is competing against a small clean sub-crop, unbounded rms penalties let
    # the sub-crop win even though the full-image WCS is geometrically correct.
    # Past ~3 px rms and ~8 px max the fit is already "bad enough"; more
    # penalty doesn't change what action we should take, it just sinks the
    # full-image candidate below every sub-crop.
    capped_rms = min(rms_px, 3.0)
    capped_max = min(max_px, 8.0)
    return (
        log_matches * 32.0
        + spread * 40.0
        + area_ratio * 20.0
        + covered_quadrants * 4.0
        - capped_rms * 8.0
        - capped_max * 1.1
        - alignment_mean * 1.2
        - alignment_p75 * 0.6
    )


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
    max_wall_seconds: float = XYLIST_SUBPROCESS_TIMEOUT_S,
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
        "10",
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

    if max_wall_seconds <= 0:
        return None
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max_wall_seconds,
        )
    except subprocess.TimeoutExpired:
        return None
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
    max_wall_seconds: float = IMAGE_SUBPROCESS_TIMEOUT_S,
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
        "15",
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

    if max_wall_seconds <= 0:
        return None
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max_wall_seconds,
        )
    except subprocess.TimeoutExpired:
        return None
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
    match_count = int(result.verification.get("match_count", 0))
    rms_px = float(result.verification.get("rms_px", 99.0))
    max_px = float(result.verification.get("max_px", 99.0))
    covered_quadrants = int(result.verification.get("covered_quadrants", 0))
    score = verification_score(result.verification, result.crop, image_width, image_height)
    return (
        (
            area_ratio >= 0.75
            and match_count >= 45
            and rms_px <= 3.0
            and max_px <= 12.0
            and covered_quadrants >= 4
            and (alignment_count < 3 or alignment_mean <= 36.0)
        )
        or (
            score >= 100.0
            and match_count >= 40
            and rms_px <= 3.6
            and max_px <= 10.0
            and covered_quadrants >= 4
            and (alignment_count < 3 or alignment_mean <= 24.0)
        )
        # Full-image early-exit: once astrometry.net accepts the full frame
        # with decent match coverage, stop trying sub-crops even if their RMS
        # might be cleaner. A clean sub-crop WCS solves a smaller region, so
        # extrapolated object positions in the rest of the frame can drift;
        # a solved full-image fit — even with foreground-inflated RMS — gives
        # more trustworthy positions across the entire image.
        or (
            area_ratio >= 0.95
            and match_count >= 30
            and rms_px <= 8.0
            and max_px <= 24.0
            and covered_quadrants >= 4
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
    sky_mask: np.ndarray | None = None,
) -> tuple[SolveResult, list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    source_analysis = analyze_sources(image, sky_mask=sky_mask)
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

    solve_start = time.perf_counter()

    def remaining_budget() -> float:
        return SOLVE_TIME_BUDGET_S - (time.perf_counter() - solve_start)

    def budget_exhausted() -> bool:
        return remaining_budget() <= 0.0

    if source_analysis.mode == "sep" and source_analysis.detections:
        for crop in candidate_crops:
            if budget_exhausted():
                break
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

            # Once any crop has produced an accepted solve, we already know
            # roughly what scale the image sits at — don't keep running the
            # full (wide/medium/narrow) ladder on subsequent crops. The
            # narrow-scale pass in particular burns ~10s (capped by cpulimit)
            # on any wide-field image that doesn't match, which is pure waste
            # once we have a WCS.
            if accepted_results:
                estimated = estimate_scale_window(accepted_results[0], crop)
                crop_scale_windows = [estimated]
            else:
                crop_scale_windows = scale_windows
            for scale_low, scale_high in crop_scale_windows:
                if budget_exhausted():
                    break
                _attempt_start = time.perf_counter()
                result = run_solve_on_xylist(
                    xylist_path,
                    crop,
                    scale_low,
                    scale_high,
                    workdir,
                    index_dir,
                    max_wall_seconds=min(XYLIST_SUBPROCESS_TIMEOUT_S, remaining_budget()),
                )
                _attempt_ms = (time.perf_counter() - _attempt_start) * 1000.0
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
                        "wall_ms": round(_attempt_ms, 1),
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
        if budget_exhausted():
            break
        crop_path = save_crop(image, crop, workdir)
        crop_accepted = False
        crop_scale_windows = (
            [fallback_scale_windows[crop_index]]
            if xyls_results and crop_index < len(fallback_scale_windows)
            else scale_windows
        )
        for scale_low, scale_high in crop_scale_windows:
            if budget_exhausted():
                break
            for downsample in (2, 4, 1):
                if budget_exhausted():
                    break
                _attempt_start = time.perf_counter()
                result = run_solve_on_image(
                    crop_path,
                    crop,
                    downsample,
                    scale_low,
                    scale_high,
                    workdir,
                    index_dir,
                    max_wall_seconds=min(IMAGE_SUBPROCESS_TIMEOUT_S, remaining_budget()),
                )
                _attempt_ms = (time.perf_counter() - _attempt_start) * 1000.0
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
                        "wall_ms": round(_attempt_ms, 1),
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

    elapsed = time.perf_counter() - solve_start
    if budget_exhausted():
        raise SolveTimeoutError(
            f"plate solving aborted after {elapsed:.1f}s (budget {SOLVE_TIME_BUDGET_S:.0f}s); "
            f"image may be too wide-field, too distorted, or contain too few matchable stars"
        )
    raise RuntimeError("plate solving failed for all full-image and scored-crop attempts")


def summarize_solver_output(stdout: str, stderr: str) -> str:
    combined = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
    lines = [line for line in combined.splitlines() if line.strip()]
    return "\n".join(lines[-25:])
