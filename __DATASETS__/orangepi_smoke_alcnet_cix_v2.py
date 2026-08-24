#!/usr/bin/env python3
"""
ALCNet CIX smoke-test on Orange Pi 6 Plus.

Spatial dimensions are inferred from input.npy / reference.npy.
Supports [1,1,H,W] such as [1,1,640,512].
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from noe_engine import EngineInfer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", type=Path, default=Path("alcnet_cix_output.npy"))
    p.add_argument("--report", type=Path, default=Path("alcnet_cix_smoke_report.json"))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "std_ms": float(a.std()),
    }


def compare(ref: np.ndarray, got: np.ndarray, threshold: float) -> dict:
    if ref.shape != got.shape:
        raise RuntimeError(f"Shape mismatch: reference={ref.shape}, cix={got.shape}")

    ref = ref.astype(np.float32, copy=False)
    got = got.astype(np.float32, copy=False)
    diff = np.abs(ref - got)

    ref_mask = ref > threshold
    got_mask = got > threshold
    intersection = int(np.logical_and(ref_mask, got_mask).sum())
    union = int(np.logical_or(ref_mask, got_mask).sum())
    flips = np.logical_xor(ref_mask, got_mask)

    return {
        "mae": float(diff.mean()),
        "max_abs_error": float(diff.max()),
        "rmse": float(np.sqrt(np.mean(np.square(ref - got)))),
        "mask_iou": 1.0 if union == 0 else float(intersection / union),
        "threshold_flip_pixels": int(flips.sum()),
        "reference": {
            "min": float(ref.min()),
            "max": float(ref.max()),
            "mean": float(ref.mean()),
            "foreground_pixels": int(ref_mask.sum()),
        },
        "cix": {
            "min": float(got.min()),
            "max": float(got.max()),
            "mean": float(got.mean()),
            "foreground_pixels": int(got_mask.sum()),
        },
    }


def main() -> int:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    input_path = args.input.expanduser().resolve()

    x = np.asarray(np.load(input_path), dtype=np.float32)
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 1:
        raise RuntimeError(f"Expected [1,1,H,W] input, got {x.shape}")
    x = np.ascontiguousarray(x)
    expected_output_size = int(np.prod(x.shape))

    print("=" * 72)
    print("ALCNet CIX / Orange Pi smoke test")
    print("=" * 72)
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print(f"Input shape/dtype: {x.shape} / {x.dtype}")
    print(
        f"Input min/max/mean: {float(x.min()):.8f} / "
        f"{float(x.max()):.8f} / {float(x.mean()):.8f}"
    )

    t0 = time.perf_counter()
    engine = EngineInfer(str(model_path))
    init_ms = (time.perf_counter() - t0) * 1000.0

    try:
        t0 = time.perf_counter()
        outputs = engine.forward(x)
        first_e2e_ms = (time.perf_counter() - t0) * 1000.0
        first_npu_ms = engine.get_cur_dur() * 1000.0

        if len(outputs) != 1:
            raise RuntimeError(f"Expected 1 output, got {len(outputs)}")

        raw = np.asarray(outputs[0])
        if raw.size != expected_output_size:
            raise RuntimeError(
                f"Unexpected CIX output size: {raw.size}; expected {expected_output_size}"
            )

        y = np.ascontiguousarray(raw.astype(np.float32, copy=False).reshape(x.shape))

        print(f"\nEngine init: {init_ms:.3f} ms")
        print(f"Raw output shape/dtype: {raw.shape} / {raw.dtype}")
        print(f"CIX output reshaped: {y.shape}")
        print(
            f"CIX output min/max/mean: {float(y.min()):.10g} / "
            f"{float(y.max()):.10g} / {float(y.mean()):.10g}"
        )
        print(f"First NPU: {first_npu_ms:.3f} ms")
        print(f"First E2E: {first_e2e_ms:.3f} ms")

        for _ in range(args.warmup):
            engine.forward(x)

        npu_times = []
        e2e_times = []
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            engine.forward(x)
            e2e_times.append((time.perf_counter() - t0) * 1000.0)
            npu_times.append(engine.get_cur_dur() * 1000.0)
    finally:
        engine.clean()

    npu = stats(npu_times)
    e2e = stats(e2e_times)

    print("\nPerformance:")
    print(
        f"NPU: mean={npu['mean_ms']:.3f} ms, "
        f"median={npu['median_ms']:.3f} ms, p95={npu['p95_ms']:.3f} ms"
    )
    print(
        f"E2E: mean={e2e['mean_ms']:.3f} ms, "
        f"median={e2e['median_ms']:.3f} ms, p95={e2e['p95_ms']:.3f} ms"
    )
    print(f"NPU FPS: {1000.0 / npu['mean_ms']:.2f}")
    print(f"E2E FPS: {1000.0 / e2e['mean_ms']:.2f}")

    comparison = None
    if args.reference:
        reference = np.asarray(np.load(args.reference), dtype=np.float32)
        comparison = compare(reference, y, args.threshold)

        print("\nFP32 reference vs CIX:")
        print(f"MAE:             {comparison['mae']:.10g}")
        print(f"Max abs error:   {comparison['max_abs_error']:.10g}")
        print(f"RMSE:            {comparison['rmse']:.10g}")
        print(f"Mask IoU @0.5:   {comparison['mask_iou']:.10f}")
        print(f"Threshold flips: {comparison['threshold_flip_pixels']}")
        print(
            f"Foreground pixels: reference={comparison['reference']['foreground_pixels']}, "
            f"cix={comparison['cix']['foreground_pixels']}"
        )
        print(
            f"Reference min/max/mean: {comparison['reference']['min']:.10g} / "
            f"{comparison['reference']['max']:.10g} / {comparison['reference']['mean']:.10g}"
        )
        print(
            f"CIX min/max/mean:       {comparison['cix']['min']:.10g} / "
            f"{comparison['cix']['max']:.10g} / {comparison['cix']['mean']:.10g}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, y)

    report = {
        "model": str(model_path),
        "input": str(input_path),
        "input_shape": list(x.shape),
        "raw_output_shape": list(raw.shape),
        "reshaped_output_shape": list(y.shape),
        "init_ms": float(init_ms),
        "first_npu_ms": float(first_npu_ms),
        "first_e2e_ms": float(first_e2e_ms),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "npu": npu,
        "e2e": e2e,
        "npu_fps": float(1000.0 / npu["mean_ms"]),
        "e2e_fps": float(1000.0 / e2e["mean_ms"]),
        "comparison": comparison,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved output: {args.output}")
    print(f"Saved report: {args.report}")
    print("\nSMOKE RUNTIME OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
