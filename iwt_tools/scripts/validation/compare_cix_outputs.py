"""Сравнивает эталонные PyTorch maps с INT8 outputs CIX на validation."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np

from iwt_tools.cix.cix_dataset import read_manifest
from iwt_tools.cix.output_comparison import (
    EXPECTED_FRAME_SHAPE,
    FrameComparison,
    compare_frame,
    ground_truth_for_frame,
)
from iwt_tools.evaluation.evaluation_core import SIZE_CLASSES, load_ground_truth_index



def _ratio(numerator: float, denominator: float) -> float:
    """Безопасно вычисляет отношение, возвращая ноль при пустом знаменателе."""

    return numerator / denominator if denominator else 0.0


def _summary(
    rows: Sequence[FrameComparison], size_gt: Counter[str],
    pytorch_size_tp: Counter[str], cix_size_tp: Counter[str],
) -> dict[str, object]:
    """Собирает aggregate pixel, object и size-class metrics."""

    result: dict[str, object] = {
        "frames": len(rows),
        "mae": _ratio(sum(row.mae for row in rows), len(rows)),
        "max_abs_error": max(row.max_abs_error for row in rows),
        "mean_mask_iou": _ratio(sum(row.mask_iou for row in rows), len(rows)),
        "threshold_flips": sum(row.threshold_flips for row in rows),
        "pytorch_foreground_pixels": sum(row.pytorch_foreground_pixels for row in rows),
        "cix_foreground_pixels": sum(row.cix_foreground_pixels for row in rows),
        "pytorch_components": sum(row.pytorch_components for row in rows),
        "cix_components": sum(row.cix_components for row in rows),
    }
    for runtime in ("pytorch", "cix"):
        tp = sum(getattr(row, f"{runtime}_tp") for row in rows)
        fn = sum(getattr(row, f"{runtime}_fn") for row in rows)
        fp = sum(getattr(row, f"{runtime}_fp") for row in rows)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        result.update({
            f"{runtime}_tp": tp, f"{runtime}_fn": fn, f"{runtime}_fp": fp,
            f"{runtime}_precision": precision, f"{runtime}_recall": recall,
            f"{runtime}_f1": _ratio(2 * precision * recall, precision + recall),
        })
    for size_class in SIZE_CLASSES:
        key = size_class.lower()
        result[f"{key}_gt"] = size_gt[size_class]
        result[f"pytorch_{key}_recall"] = _ratio(
            pytorch_size_tp[size_class], size_gt[size_class]
        )
        result[f"cix_{key}_recall"] = _ratio(cix_size_tp[size_class], size_gt[size_class])
    return result


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    """Записывает одну из таблиц сравнения с фиксированными колонками."""

    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI сравнения CIX outputs с PyTorch reference."""

    parser = argparse.ArgumentParser(description="Compare PyTorch and CIX outputs")
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--cix-outputs", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> Path:
    """Проверяет артефакты, считает метрики и записывает два CSV отчёта."""

    frames = read_manifest(args.validation / "validation_manifest.csv")
    pytorch = np.load(args.validation / "pytorch_outputs.npy", mmap_mode="r")
    cix = np.load(args.cix_outputs, mmap_mode="r")
    expected_shape = (len(frames), *EXPECTED_FRAME_SHAPE)
    if pytorch.shape != expected_shape or cix.shape != expected_shape:
        raise ValueError(
            f"Expected PyTorch and CIX shape {expected_shape}, got {pytorch.shape} and {cix.shape}"
        )
    if not np.issubdtype(pytorch.dtype, np.floating) or not np.issubdtype(cix.dtype, np.floating):
        raise ValueError("PyTorch and CIX outputs must use a floating dtype")
    gt_index = dict(load_ground_truth_index(args.ground_truth))
    rows: list[FrameComparison] = []
    size_gt: Counter[str] = Counter()
    pytorch_size_tp: Counter[str] = Counter()
    cix_size_tp: Counter[str] = Counter()
    for index, frame in enumerate(frames):
        ground_truth = ground_truth_for_frame(frame, gt_index)
        size_gt.update(record.size_class for record in ground_truth.scored)
        row, pytorch_tp, cix_tp = compare_frame(frame, pytorch[index], cix[index], ground_truth)
        rows.append(row)
        pytorch_size_tp.update(pytorch_tp)
        cix_size_tp.update(cix_tp)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.output / "frame_comparison.csv",
        [field.name for field in fields(FrameComparison)],
        [asdict(row) for row in rows],
    )
    summary = _summary(rows, size_gt, pytorch_size_tp, cix_size_tp)
    _write_csv(args.output / "summary.csv", list(summary), [summary])
    return args.output


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Запускает CLI и сообщает ожидаемые ошибки пользователю."""

    try:
        path = run(build_argument_parser().parse_args(argv))
        print(f"Comparison saved: {path}")
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
