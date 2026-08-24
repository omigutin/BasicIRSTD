"""Запускает validation tensor пакетами через CIX на Orange Pi."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from itertools import chain
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, cast, Iterator, Optional, Protocol, Sequence

import numpy as np

FRAME_SHAPE = (1, 640, 512)
SUPPORTED_BATCH_SIZES = (1, 4)


@dataclass(frozen=True)
class TimingSummary:
    """Хранит batch timing, среднее время обработки кадра и throughput."""

    batch_mean_ms: float
    batch_median_ms: float
    batch_p95_ms: float
    ms_per_frame: float
    fps: float


class Engine(Protocol):
    """Описывает используемую runner часть интерфейса NOE Engine."""

    def forward(self, input_tensor: np.ndarray) -> object:
        """Выполняет один inference-вызов."""

    def get_cur_dur(self) -> float:
        """Возвращает длительность последнего NPU-вызова в секундах."""

    def clean(self) -> None:
        """Освобождает ресурсы vendor runtime."""


def load_engine_factory() -> Callable[[str], Engine]:
    """Загружает системный NOE runtime или сообщает, как исправить окружение."""

    try:
        module = import_module("NOE_Engine")
    except ImportError as error:
        raise RuntimeError(
            "NOE_Engine is unavailable; verify CIX/NOE runtime installation "
            "and Python environment"
        ) from error
    factory = getattr(module, "EngineInfer", None)
    if factory is None:
        raise RuntimeError("NOE_Engine does not provide EngineInfer")
    return cast(Callable[[str], Engine], factory)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI Orange Pi validation runner."""

    parser = argparse.ArgumentParser(description="Run CIX validation on Orange Pi")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument(
        "--batch-size", type=int, choices=SUPPORTED_BATCH_SIZES, default=1
    )
    return parser


def _timing_summary(
    timings_ms: Sequence[float], batch_size: int
) -> TimingSummary:
    """Считает latency вызова, время обработки кадра и FPS."""

    if batch_size not in SUPPORTED_BATCH_SIZES:
        raise ValueError(f"Unsupported batch size: {batch_size}")
    if not timings_ms:
        return TimingSummary(0.0, 0.0, 0.0, 0.0, 0.0)
    values = np.asarray(timings_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    return TimingSummary(
        batch_mean_ms=mean_ms,
        batch_median_ms=float(np.median(values)),
        batch_p95_ms=float(np.percentile(values, 95)),
        ms_per_frame=mean_ms / batch_size,
        fps=batch_size * 1000.0 / mean_ms if mean_ms else 0.0,
    )


def iter_padded_batches(
    inputs: np.ndarray, batch_size: int
) -> Iterator[tuple[np.ndarray, int, bool]]:
    """Формирует полные batches и сообщает число реальных кадров."""

    if batch_size not in SUPPORTED_BATCH_SIZES:
        raise ValueError(f"Unsupported batch size: {batch_size}")
    for start in range(0, len(inputs), batch_size):
        real_batch = np.asarray(inputs[start:start + batch_size])
        real_count = len(real_batch)
        if real_count == 0:
            continue
        padded = real_count < batch_size
        if padded:
            padding = np.repeat(real_batch[-1:], batch_size - real_count, axis=0)
            real_batch = np.concatenate((real_batch, padding), axis=0)
        yield np.ascontiguousarray(real_batch), real_count, padded


def reshape_cix_output(
    raw_outputs: object, batch_size: int, frame_shape: tuple[int, int, int] = FRAME_SHAPE
) -> np.ndarray:
    """Преобразует единственный flattened CIX output в NCHW batch."""

    if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) != 1:
        raise ValueError("EngineInfer.forward() must return a single-item output list")
    flattened = np.asarray(raw_outputs[0])
    if flattened.dtype != np.float32:
        raise ValueError(f"Expected float32 CIX output, got {flattened.dtype}")
    expected_size = batch_size * int(np.prod(frame_shape))
    if flattened.size != expected_size:
        raise ValueError(
            f"Expected flattened CIX output size {expected_size}, got {flattened.size}"
        )
    return np.ascontiguousarray(flattened.reshape((batch_size, *frame_shape)))


def store_real_outputs(
    destination: np.ndarray,
    start: int,
    batch_outputs: np.ndarray,
    real_count: int,
) -> None:
    """Сохраняет только outputs реальных, а не padding-кадров."""

    if not 0 < real_count <= len(batch_outputs):
        raise ValueError(f"Invalid real output count: {real_count}")
    destination[start:start + real_count] = batch_outputs[:real_count]


def run(args: argparse.Namespace) -> Path:
    """Запускает EngineInfer пакетами и сохраняет outputs по кадрам."""

    if not args.model.is_file():
        raise FileNotFoundError(f"CIX model file not found: {args.model}")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    inputs = np.load(args.input, mmap_mode="r")
    if inputs.ndim != 4 or tuple(inputs.shape[1:]) != FRAME_SHAPE:
        raise ValueError(f"Expected input shape [N, 1, 640, 512], got {inputs.shape}")
    if inputs.dtype != np.float32:
        raise ValueError(f"Expected float32 input, got {inputs.dtype}")
    if args.warmup < 0:
        raise ValueError("Warmup count must not be negative")
    if args.batch_size not in SUPPORTED_BATCH_SIZES:
        raise ValueError(f"Unsupported batch size: {args.batch_size}")

    engine = load_engine_factory()(str(args.model))
    padded_batches = 0
    batch_count = 0
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        outputs = np.lib.format.open_memmap(
            args.output, mode="w+", dtype=np.float32, shape=inputs.shape
        )
        npu_timings_ms: list[float] = []
        e2e_timings_ms: list[float] = []
        batches = iter_padded_batches(inputs, args.batch_size)
        first_batch = next(batches, None)
        if first_batch is not None:
            warmup_batch = first_batch[0]
            for _ in range(args.warmup):
                engine.forward(warmup_batch)

            batches = chain((first_batch,), batches)

        output_index = 0
        for batch, real_count, padded in batches:
            started = perf_counter()
            raw_outputs = engine.forward(batch)
            e2e_timings_ms.append((perf_counter() - started) * 1000.0)
            npu_timings_ms.append(float(engine.get_cur_dur()) * 1000.0)
            batch_outputs = reshape_cix_output(raw_outputs, args.batch_size)
            store_real_outputs(outputs, output_index, batch_outputs, real_count)
            output_index += real_count
            batch_count += 1
            padded_batches += int(padded)
        outputs.flush()
    finally:
        engine.clean()

    npu = _timing_summary(npu_timings_ms, args.batch_size)
    e2e = _timing_summary(e2e_timings_ms, args.batch_size)
    print(
        f"frames={len(inputs)} batches={batch_count} batch_size={args.batch_size} "
        f"padded_batches={padded_batches} warmup={args.warmup}"
    )
    print(
        f"npu_batch_ms mean={npu.batch_mean_ms:.3f} "
        f"median={npu.batch_median_ms:.3f} p95={npu.batch_p95_ms:.3f} "
        f"npu_ms_per_frame={npu.ms_per_frame:.3f} fps={npu.fps:.3f}"
    )
    print(
        f"e2e_batch_ms mean={e2e.batch_mean_ms:.3f} "
        f"median={e2e.batch_median_ms:.3f} p95={e2e.batch_p95_ms:.3f} "
        f"e2e_ms_per_frame={e2e.ms_per_frame:.3f} fps={e2e.fps:.3f}"
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
