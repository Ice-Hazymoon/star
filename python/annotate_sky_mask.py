#!/usr/bin/env python3
"""
Sky/foreground segmentation used to suppress annotations that would otherwise
be drawn on top of terrestrial features (sand, buildings, trees, people).

Model: SegFormer-b4 fine-tuned on ADE20K. The ADE20K `sky` class (index 2) is
used directly. Two preprocessing variants exist:

- *raw* works for images that are nearly all sky (ADE20K has plenty of
  star-field-like training images).
- *tonemap* (percentile stretch + gamma 0.45) is needed when there's a real
  foreground, because the model has no direct exposure to night scenes.

We pick between them up front from the downsampled image's median brightness
— pure night skies are extremely dark, foreground-containing shots are not —
so the expensive SegFormer pass usually runs exactly once per image. If the
raw path disagrees with the heuristic (model predicts little sky on what was
assumed to be pure sky) we fall back to a second pass on the tonemapped image.

Runtime: the model is loaded as an int8-quantized ONNX graph via onnxruntime
(built once at Docker build time by export_sky_mask_onnx.py). If the ONNX
file is missing (local dev), we fall back to loading the PyTorch model.

If torch/transformers and onnxruntime all fail, the helpers gracefully no-op.
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

# Env vars consulted by huggingface_hub / transformers during import — set
# defensively in case this module is imported before the worker entry point.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", module=r"huggingface_hub.*")

import numpy as np
from PIL import Image
from scipy import ndimage as _ndi

MODEL_ID = "nvidia/segformer-b4-finetuned-ade-512-512"
SKY_CLASS_INDEX = 2  # ADE20K `sky`
# Inference size. The model was finetuned on 512×512 ADE20K crops and loses
# a lot of accuracy at smaller input sizes — e.g. tonemap sky ratio on the
# orion-over-pines sample goes from 44% @ 512 to 0% @ 384. Keep at 512. The
# int8-quantized ONNX graph is fixed to this size too; changing it requires
# re-exporting via python/export_sky_mask_onnx.py.
INFERENCE_SIZE = 512
RAW_SKY_THRESHOLD = 0.6
# Below this median brightness on the downsampled image, we assume pure sky
# and take the raw-inference fast path. Anything above tonemaps directly.
PURE_SKY_MEDIAN_THRESHOLD = 25.0
# Erode the sky mask (i.e. grow the ground region) by this many pixels at
# small-scale before upsample, to give annotations a safety margin from the
# mask boundary.
SKY_SAFETY_MARGIN_PX = 12
# 3×3 full (Chebyshev / 8-connectivity) structuring element — matches the
# semantics of PIL's MinFilter(3) / MaxFilter(3).
_STRUCT_3X3 = _ndi.generate_binary_structure(2, 2)

ONNX_MODEL_FILENAME = "sky_mask_int8.onnx"
# Search order: explicit env var, then the Docker HF cache, then a local
# ./hf_cache for dev. export_sky_mask_onnx.py writes here during build.
_ONNX_SEARCH_PATHS = [
    os.environ.get("SKY_MASK_ONNX_PATH"),
    str(Path(os.environ.get("HF_HOME", "/app/hf_cache")) / ONNX_MODEL_FILENAME),
    str(Path(__file__).resolve().parent.parent / "hf_cache" / ONNX_MODEL_FILENAME),
]

_logger = logging.getLogger(__name__)
_load_attempted = False
_model: Any = None      # torch fallback
_session: Any = None    # onnxruntime InferenceSession (preferred)
_processor: Any = None


def _find_onnx_model() -> Path | None:
    for candidate in _ONNX_SEARCH_PATHS:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def _load_processor() -> Any:
    from transformers import AutoImageProcessor
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    try:
        from huggingface_hub import logging as hub_logging

        hub_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 - logging is optional
        pass

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    # Override the processor's default 512×512 resize. This matches the fixed
    # input size of the exported ONNX graph.
    try:
        processor.size = {"height": INFERENCE_SIZE, "width": INFERENCE_SIZE}
    except Exception:  # noqa: BLE001 - older processor API variants
        pass
    return processor


def _try_load_onnx(processor: Any) -> Any | None:
    onnx_path = _find_onnx_model()
    if onnx_path is None:
        _logger.info("sky-mask ONNX file not found; using PyTorch fallback")
        return None
    try:
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        _logger.info("sky-mask model loaded via ONNX: %s", onnx_path)
        return session
    except Exception as exc:  # noqa: BLE001 - fall through to PyTorch
        _logger.warning(
            "failed to load ONNX sky-mask model (%s); using PyTorch fallback", exc
        )
        return None


def _try_load_torch() -> Any | None:
    try:
        from transformers import SegformerForSemanticSegmentation

        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
        model.eval()
        _logger.info("sky-mask model loaded via PyTorch: %s", MODEL_ID)
        return model
    except Exception as exc:  # noqa: BLE001 - graceful degradation
        _logger.warning("sky-mask torch model unavailable (%s)", exc)
        return None


def _load_model() -> bool:
    """Populate the module-level model + processor. Returns True on success."""
    global _load_attempted, _model, _session, _processor
    if _processor is not None and (_session is not None or _model is not None):
        return True
    if _load_attempted:
        return False
    _load_attempted = True
    try:
        _processor = _load_processor()
    except Exception as exc:  # noqa: BLE001
        _logger.warning("sky-mask processor unavailable (%s); masking disabled", exc)
        return False

    _session = _try_load_onnx(_processor)
    if _session is not None:
        return True
    _model = _try_load_torch()
    if _model is not None:
        return True

    _logger.warning("sky-mask model unavailable; masking disabled")
    _processor = None
    return False


def preload() -> bool:
    """Explicit warmup hook. Returns True when the model is ready."""
    return _load_model()


def _tonemap(rgb: np.ndarray) -> Image.Image:
    arr = rgb.astype(np.float32) / 255.0
    lo = np.percentile(arr, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(arr, 98, axis=(0, 1), keepdims=True)
    stretched = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    gamma = stretched ** 0.45
    return Image.fromarray((gamma * 255).astype(np.uint8), "RGB")


def _downsample(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    w, h = rgb.size
    scale = INFERENCE_SIZE / max(w, h)
    if scale >= 1.0:
        return rgb
    # LANCZOS here is load-bearing for accuracy: on images with fine
    # foreground detail (e.g. tree silhouettes against a star field) BILINEAR
    # blurs enough to confuse the model — tonemap sky ratio on
    # orion-over-pines collapses from 33% (LANCZOS) to 4% (BILINEAR).
    return rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _infer_logits(pixel_values: np.ndarray) -> np.ndarray:
    """Run the segformer forward pass. Accepts preprocessor output (float32,
    NCHW) and returns logits (float32, shape [1, num_classes, H/4, W/4])."""
    if _session is not None:
        return _session.run(["logits"], {"pixel_values": pixel_values})[0]
    import torch

    with torch.no_grad():
        outputs = _model(pixel_values=torch.from_numpy(pixel_values))
    return outputs.logits.cpu().numpy()


def _run_segformer(img: Image.Image) -> np.ndarray:
    inputs = _processor(images=img, return_tensors="np")
    pixel_values = inputs["pixel_values"].astype(np.float32)
    logits = _infer_logits(pixel_values)
    # Argmax at the logits' native resolution (typically 96×96 for 384 input);
    # the final resize to source size happens in _cleanup_and_resize, so an
    # intermediate upsample-then-argmax would only add cost without affecting
    # the final mask quality after the safety-margin erosion.
    pred_small = np.argmax(logits[0], axis=0)
    sky_small = ((pred_small == SKY_CLASS_INDEX).astype(np.uint8) * 255)
    mask = Image.fromarray(sky_small, "L").resize(
        (img.width, img.height), Image.BILINEAR
    )
    return (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.uint8)


def _drop_floating_ground_blobs(sky_mask: np.ndarray) -> np.ndarray:
    """
    A ground blob floating inside the sky (not connected to any image border) is
    almost always a misclassification — bright Milky Way cores, compact star
    clusters, a lone satellite trail, etc. Real foreground touches the frame.
    Label every ground component and keep only the ones touching the frame.
    """
    ground = sky_mask == 0
    labels, num = _ndi.label(ground, structure=_STRUCT_3X3)
    if num == 0:
        return sky_mask.copy()
    border = np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
    )
    border_labels = np.unique(border[border != 0])
    keep = np.zeros(num + 1, dtype=bool)
    keep[border_labels] = True
    result = sky_mask.copy()
    result[ground & ~keep[labels]] = 1
    return result


def _cleanup_and_resize(small_mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    mask_bool = small_mask.astype(bool)
    # Morphological closing at small scale: fill holes in sky introduced by
    # misclassified bright regions (stars, Milky Way cores). 3×3 full structure
    # × 2 iterations ≈ PIL MaxFilter(5) + MinFilter(5).
    closed = _ndi.binary_closing(mask_bool, structure=_STRUCT_3X3, iterations=2)
    cleaned = _drop_floating_ground_blobs(closed.astype(np.uint8))
    # Safety margin: erode sky so annotations near the horizon don't hug the
    # boundary.
    eroded = _ndi.binary_erosion(
        cleaned.astype(bool),
        structure=_STRUCT_3X3,
        iterations=max(0, SKY_SAFETY_MARGIN_PX),
    )
    full = Image.fromarray((eroded.astype(np.uint8) * 255), "L").resize(
        target_size, Image.BILINEAR
    )
    return (np.asarray(full, dtype=np.uint8) >= 128).astype(np.uint8)


def compute_sky_mask(image: Image.Image) -> np.ndarray | None:
    """
    Return a uint8 mask (0/1) the same size as `image`. 1 = sky.

    Returns None if the underlying model couldn't load — callers should then
    skip masking rather than fail.
    """
    if not _load_model():
        return None

    small = _downsample(image)
    rgb_small = np.asarray(small, dtype=np.uint8)

    # Pick the preprocessing up front from image statistics so we only run the
    # expensive SegFormer pass once. A very dark median means "almost certainly
    # pure night sky" → raw works. Anything brighter has foreground and needs
    # tonemap to get a usable mask from the ADE20K model.
    gray = rgb_small.mean(axis=-1) if rgb_small.ndim == 3 else rgb_small
    median_brightness = float(np.median(gray))

    if median_brightness < PURE_SKY_MEDIAN_THRESHOLD:
        chosen = _run_segformer(small)
        # Pre-check was wrong (model says mostly ground on a "dark" image).
        # Retry once with tonemap; this is the only case that runs two passes.
        if chosen.mean() < RAW_SKY_THRESHOLD:
            chosen = _run_segformer(_tonemap(rgb_small))
    else:
        chosen = _run_segformer(_tonemap(rgb_small))

    return _cleanup_and_resize(chosen, image.size)


def _in_sky(mask: np.ndarray, x: float, y: float) -> bool:
    height, width = mask.shape
    xi = max(0, min(width - 1, int(round(float(x)))))
    yi = max(0, min(height - 1, int(round(float(y)))))
    return bool(mask[yi, xi])


def mask_is_trustworthy(
    mask: np.ndarray,
    star_positions: list[tuple[float, float]],
    min_sky_ratio: float = 0.25,
) -> bool:
    """Sanity-check a mask against plate-solved star positions. Stars can only
    appear in the sky, so if almost none of them land in the mask's sky
    region, the segmentation model hallucinated a horizon (common on pure
    night-sky images with a brightness vignette) and the mask should be
    discarded. The threshold is intentionally low: in a real horizon shot
    (e.g. Orion rising behind trees) many named stars legitimately sit at or
    below the treeline, so a 50% rejection rule would nuke correct masks."""
    if not star_positions:
        return True
    in_sky = sum(1 for x, y in star_positions if _in_sky(mask, x, y))
    return (in_sky / len(star_positions)) >= min_sky_ratio


def filter_named_stars(
    named_stars: list[dict[str, Any]], mask: np.ndarray | None
) -> list[dict[str, Any]]:
    if mask is None:
        return named_stars
    return [star for star in named_stars if _in_sky(mask, star["x"], star["y"])]


def filter_deep_sky_objects(
    objects: list[dict[str, Any]], mask: np.ndarray | None
) -> list[dict[str, Any]]:
    if mask is None:
        return objects
    return [obj for obj in objects if _in_sky(mask, obj["x"], obj["y"])]


def _clip_segment_to_sky(
    segment: dict[str, Any], mask: np.ndarray
) -> dict[str, Any] | None:
    """Return the in-sky portion of a constellation segment, or None if the
    whole segment is on ground. Uses binary search to find the mask boundary
    when only one endpoint is in sky."""
    start = segment["start"]
    end = segment["end"]
    start_in_sky = _in_sky(mask, start["x"], start["y"])
    end_in_sky = _in_sky(mask, end["x"], end["y"])
    if start_in_sky and end_in_sky:
        return segment
    if not start_in_sky and not end_in_sky:
        return None

    sky_point = start if start_in_sky else end
    ground_point = end if start_in_sky else start
    sx, sy = float(sky_point["x"]), float(sky_point["y"])
    gx, gy = float(ground_point["x"]), float(ground_point["y"])
    for _ in range(14):
        mx, my = (sx + gx) / 2.0, (sy + gy) / 2.0
        if _in_sky(mask, mx, my):
            sx, sy = mx, my
        else:
            gx, gy = mx, my

    boundary = {"x": sx, "y": sy}
    if start_in_sky:
        return {"start": start, "end": boundary}
    return {"start": boundary, "end": end}


def _relocate_label_to_sky(
    kept_segments: list[dict[str, Any]], mask: np.ndarray
) -> tuple[float, float] | None:
    """Pick an in-sky anchor for a constellation whose original label fell on ground.
    Prefer the centroid of kept segments; fall back to the midpoint of each kept
    segment. Both endpoints of a kept segment are in sky by construction."""
    if not kept_segments:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for segment in kept_segments:
        xs.extend([segment["start"]["x"], segment["end"]["x"]])
        ys.extend([segment["start"]["y"], segment["end"]["y"]])
    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)
    if _in_sky(mask, centroid_x, centroid_y):
        return centroid_x, centroid_y

    for segment in kept_segments:
        mid_x = (segment["start"]["x"] + segment["end"]["x"]) / 2.0
        mid_y = (segment["start"]["y"] + segment["end"]["y"]) / 2.0
        if _in_sky(mask, mid_x, mid_y):
            return mid_x, mid_y

    first = kept_segments[0]["start"]
    return float(first["x"]), float(first["y"])


def _label_is_near_border(
    label_x: float, label_y: float, mask: np.ndarray, pad_px: float = 8.0
) -> bool:
    """A label clamped to the image edge by the backend renders off-screen in
    the layout engine and gets silently dropped. Treat borderline positions as
    invalid so we can try a better one."""
    height, width = mask.shape
    return (
        label_x <= pad_px
        or label_x >= width - pad_px
        or label_y <= pad_px
        or label_y >= height - pad_px
    )


def filter_constellations(
    constellations: list[dict[str, Any]], mask: np.ndarray | None
) -> list[dict[str, Any]]:
    if mask is None:
        return constellations
    filtered: list[dict[str, Any]] = []
    for constellation in constellations:
        kept_segments = []
        for segment in constellation.get("segments", []):
            clipped = _clip_segment_to_sky(segment, mask)
            if clipped is not None:
                kept_segments.append(clipped)

        label_x = constellation["label_x"]
        label_y = constellation["label_y"]
        label_in_sky = _in_sky(mask, label_x, label_y)
        label_at_border = _label_is_near_border(label_x, label_y, mask)
        if not kept_segments and not label_in_sky:
            continue

        new_entry = dict(constellation)
        new_entry["segments"] = kept_segments

        if not label_in_sky or label_at_border:
            relocated = _relocate_label_to_sky(kept_segments, mask)
            if relocated is not None:
                # Accept relocation only if the new position is itself not on a
                # border — otherwise stick with the original.
                rx, ry = relocated
                if not _label_is_near_border(rx, ry, mask):
                    new_entry["label_x"] = float(rx)
                    new_entry["label_y"] = float(ry)
                    new_entry["show_label"] = True
                elif not label_in_sky:
                    new_entry["show_label"] = False
            elif not label_in_sky:
                new_entry["show_label"] = False

        filtered.append(new_entry)
    return filtered
