#!/usr/bin/env python3
"""
Export the SegFormer sky-segmentation model to ONNX + apply dynamic int8
quantization. Intended to run once during Docker build, producing the file
annotate_sky_mask.py loads at runtime via onnxruntime.

Usage:
    python3 export_sky_mask_onnx.py <output_dir>

The script is intentionally tolerant of the dependency environment being set
up for build-time use (torch, transformers, onnx, onnxruntime all available)
but reaches into runtime config from annotate_sky_mask so the two stay in
sync (MODEL_ID, INFERENCE_SIZE, output filename).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Import from the sibling runtime module so MODEL_ID + INFERENCE_SIZE are
# defined in exactly one place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_sky_mask import (  # noqa: E402
    INFERENCE_SIZE,
    MODEL_ID,
    ONNX_MODEL_FILENAME,
)


class _SegformerLogitsWrapper(torch.nn.Module):
    """SegformerForSemanticSegmentation returns a SemanticSegmenterOutput,
    which torch.onnx.export can't serialize directly. This wrapper exposes
    just the logits tensor."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def export(output_dir: Path) -> Path:
    from transformers import SegformerForSemanticSegmentation

    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "sky_mask_fp32.onnx"
    int8_path = output_dir / ONNX_MODEL_FILENAME

    if int8_path.exists():
        print(f"[export] {int8_path} already exists, skipping")
        return int8_path

    print(f"[export] loading {MODEL_ID}")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.eval()
    wrapper = _SegformerLogitsWrapper(model)

    dummy = torch.randn(1, 3, INFERENCE_SIZE, INFERENCE_SIZE)
    print(f"[export] writing fp32 ONNX → {fp32_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(fp32_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            # Fixed input size; the runtime forces this via _processor.size.
            opset_version=14,
            do_constant_folding=True,
        )

    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"[export] dynamic int8 quantization → {int8_path}")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
    )
    # torch 2.11's dynamo-based exporter writes large models with external
    # weights — the .onnx file is a tiny graph definition pointing at a sibling
    # .data file (e.g. sky_mask_fp32.onnx.data ≈ 245 MB for b4). Delete the
    # whole group so we don't leave the sidecar behind.
    for stale in fp32_path.parent.glob(fp32_path.name + "*"):
        stale.unlink()
    size_mb = int8_path.stat().st_size / (1024 * 1024)
    print(f"[export] done: {int8_path} ({size_mb:.1f} MB)")
    return int8_path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: export_sky_mask_onnx.py <output_dir>", file=sys.stderr)
        raise SystemExit(2)
    export(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
