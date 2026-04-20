#!/usr/bin/env python3
"""
Sky/foreground segmentation used to suppress annotations that would otherwise
be drawn on top of terrestrial features (sand, buildings, trees, people).

Model: SegFormer-b4 fine-tuned on ADE20K. The ADE20K `sky` class (index 2) is
used directly; two preprocessing variants are tried:

- *raw* works for images that are nearly all sky (ADE20K has plenty of
  star-field-like training images).
- *tonemap* (percentile stretch + gamma 0.45) is needed when there's a real
  foreground, because the model has no direct exposure to night scenes.

An adaptive rule picks between the two based on the raw sky ratio.

If torch/transformers aren't importable, the helpers gracefully no-op.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any

# Env vars consulted by huggingface_hub / transformers during import — set
# defensively in case this module is imported before the worker entry point.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", module=r"huggingface_hub.*")

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MODEL_ID = "nvidia/segformer-b4-finetuned-ade-512-512"
SKY_CLASS_INDEX = 2  # ADE20K `sky`
INFERENCE_SIZE = 512
RAW_SKY_THRESHOLD = 0.6
# Erode the sky mask (i.e. grow the ground region) by this many pixels at
# small-scale before upsample, to give annotations a safety margin from the
# mask boundary. At 512-scale, 12px ≈ 36-50px in a typical source image —
# enough that a star+label sitting on the horizon isn't painted into the
# foreground even when the segmentation boundary is slightly off.
SKY_SAFETY_MARGIN_PX = 12

_logger = logging.getLogger(__name__)
_load_attempted = False
_model: Any = None
_processor: Any = None


def _load_model() -> tuple[Any, Any] | None:
    global _load_attempted, _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    if _load_attempted:
        return None
    _load_attempted = True
    try:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        try:
            from huggingface_hub import logging as hub_logging

            hub_logging.set_verbosity_error()
        except Exception:  # noqa: BLE001 - logging is optional
            pass

        _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        _model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
        _model.eval()
        _logger.info("sky-mask model loaded: %s", MODEL_ID)
        return _model, _processor
    except Exception as exc:  # noqa: BLE001 - graceful degradation
        _logger.warning("sky-mask model unavailable (%s); masking disabled", exc)
        _model = None
        _processor = None
        return None


def preload() -> bool:
    """Explicit warmup hook. Returns True when the model is ready."""
    return _load_model() is not None


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
    return rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _run_segformer(img: Image.Image) -> np.ndarray:
    import torch

    inputs = _processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)
    upsampled = torch.nn.functional.interpolate(
        outputs.logits,
        size=(img.height, img.width),
        mode="bilinear",
        align_corners=False,
    )
    pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
    return (pred == SKY_CLASS_INDEX).astype(np.uint8)


def _drop_floating_ground_blobs(sky_mask: np.ndarray) -> np.ndarray:
    """
    A ground blob floating inside the sky (not connected to any image border) is
    almost always a misclassification — bright Milky Way cores, compact star
    clusters, a lone satellite trail, etc. Real foreground touches the frame.
    Flood-fill from a ground-painted frame around the image and flip any blob
    the flood didn't reach back to sky.
    """
    height, width = sky_mask.shape
    ground = ((1 - sky_mask) * 255).astype(np.uint8)

    padded = Image.new("L", (width + 2, height + 2), 255)
    padded.paste(Image.fromarray(ground, "L"), (1, 1))
    ImageDraw.floodfill(padded, (0, 0), 128)

    reached = (np.asarray(padded)[1:-1, 1:-1] == 128).astype(np.uint8)
    floating = ((ground > 0) & (reached == 0)).astype(np.uint8)
    return (sky_mask | floating).astype(np.uint8)


def _cleanup_and_resize(small_mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    mask_img = Image.fromarray((small_mask * 255).astype(np.uint8), "L")
    # Morphological closing at small scale: fill holes in sky introduced by
    # misclassified bright regions (stars, Milky Way cores).
    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(size=5))
    cleaned = (np.asarray(mask_img, dtype=np.uint8) >= 128).astype(np.uint8)
    cleaned = _drop_floating_ground_blobs(cleaned)

    # Safety margin: erode sky so annotations near the horizon don't hug the
    # boundary. MinFilter(3) erodes 1px per pass.
    safety_img = Image.fromarray((cleaned * 255).astype(np.uint8), "L")
    for _ in range(max(0, SKY_SAFETY_MARGIN_PX)):
        safety_img = safety_img.filter(ImageFilter.MinFilter(3))

    full = safety_img.resize(target_size, Image.BILINEAR)
    return (np.asarray(full, dtype=np.uint8) >= 128).astype(np.uint8)


def compute_sky_mask(image: Image.Image) -> np.ndarray | None:
    """
    Return a uint8 mask (0/1) the same size as `image`. 1 = sky.

    Returns None if the underlying model couldn't load — callers should then
    skip masking rather than fail.
    """
    if _load_model() is None:
        return None

    small = _downsample(image)
    rgb_small = np.asarray(small, dtype=np.uint8)

    raw_mask = _run_segformer(small)
    raw_ratio = float(raw_mask.mean())
    if raw_ratio >= RAW_SKY_THRESHOLD:
        chosen = raw_mask
    else:
        chosen = _run_segformer(_tonemap(rgb_small))

    return _cleanup_and_resize(chosen, image.size)


def _in_sky(mask: np.ndarray, x: float, y: float) -> bool:
    height, width = mask.shape
    xi = max(0, min(width - 1, int(round(float(x)))))
    yi = max(0, min(height - 1, int(round(float(y)))))
    return bool(mask[yi, xi])


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
