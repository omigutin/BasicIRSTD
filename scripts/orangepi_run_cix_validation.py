"""Последовательно запускает validation tensor через CIX на Orange Pi."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np

INPUT_SHAPE = (1, 1, 640, 512)
OUTPUT_SHAPE = (1, 1, 640, 512)
FLAT_OUTPUT_SIZE = 640 * 512


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт минимальный CLI Orange Pi validation runner."""

    parser = argparse.ArgumentParser(description="Run ALCNet CIX validation on Orange Pi")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> Path:
    """Запускает EngineInfer по одному кадру и сохраняет единый output tensor."""

    from noe_engine import EngineInfer

    inputs = np.load(args.input, mmap_mode="r")
    if inputs.ndim != 4 or tuple(inputs.shape[1:]) != INPUT_SHAPE[1:]:
        raise ValueError(f"Expected input shape [N, 1, 640, 512], got {inputs.shape}")
    if inputs.dtype != np.float32:
        raise ValueError(f"Expected float32 input, got {inputs.dtype}")

    engine = EngineInfer(str(args.model))
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        outputs = np.lib.format.open_memmap(
            args.output, mode="w+", dtype=np.float32, shape=inputs.shape
        )
        npu_timings_ms: list[float] = []
        for index in range(len(inputs)):
            frame = inputs[index:index + 1]
            raw_outputs = engine.forward(frame)
            npu_timings_ms.append(float(engine.get_cur_dur()) * 1000.0)
            if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) != 1:
                raise ValueError("EngineInfer.forward() must return a single-item output list")
            flattened = np.asarray(raw_outputs[0])
            if flattened.dtype != np.float32:
                raise ValueError(f"Expected float32 CIX output, got {flattened.dtype}")
            if flattened.size != FLAT_OUTPUT_SIZE:
                raise ValueError(
                    f"Expected flattened CIX output size {FLAT_OUTPUT_SIZE}, "
                    f"got {flattened.size}"
                )
            outputs[index] = flattened.reshape(OUTPUT_SHAPE)[0]
        outputs.flush()
    finally:
        engine.clean()

    total_npu_ms = sum(npu_timings_ms)
    average_npu_ms = total_npu_ms / len(npu_timings_ms) if npu_timings_ms else 0.0
    npu_fps = 1000.0 / average_npu_ms if average_npu_ms else 0.0
    print(
        f"frames={len(npu_timings_ms)} total_npu_ms={total_npu_ms:.3f} "
        f"average_npu_ms={average_npu_ms:.3f} npu_fps={npu_fps:.3f}"
    )
    return args.output


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Запускает Orange Pi CLI и сообщает ожидаемые ошибки."""

    try:
        path = run(build_argument_parser().parse_args(argv))
        print(f"CIX outputs saved: {path}")
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
