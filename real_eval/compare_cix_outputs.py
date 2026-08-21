"""Сравнивает эталонные PyTorch maps с INT8 outputs CIX на validation."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np

from .cix_dataset import DatasetFrame, read_manifest
from .evaluation_core import (
    GroundTruthFrame, SIZE_CLASSES, load_ground_truth_index, match_frame_predictions,
)
from .evaluate_model import extract_predicted_components
from .model_runner import PredictionResult

EXPECTED_FRAME_SHAPE = (1, 640, 512)
THRESHOLD = 0.5


@dataclass(frozen=True)
class FrameComparison:
    """Хранит численную и объектную разницу одного validation кадра."""

    source_file: str
    frame_type: str
    source_set: str
    size_class: str
    mae: float
    max_abs_error: float
    mask_iou: float
    threshold_flips: int
    pytorch_foreground_pixels: int
    cix_foreground_pixels: int
    pytorch_components: int
    cix_components: int
    pytorch_tp: int
    pytorch_fn: int
    pytorch_fp: int
    cix_tp: int
    cix_fn: int
    cix_fp: int


def _prediction_result(probability: np.ndarray) -> PredictionResult:
    """Адаптирует probability map к существующему component extractor."""

    if probability.shape != EXPECTED_FRAME_SHAPE:
        raise ValueError(f"Expected frame output shape {EXPECTED_FRAME_SHAPE}, got {probability.shape}")
    probability_map = probability[0]
    binary_mask = probability_map > THRESHOLD
    foreground = int(np.count_nonzero(binary_mask))
    return PredictionResult(
        probability_map=probability_map,
        binary_mask=binary_mask,
        max_score=float(probability_map.max()),
        target_found=foreground > 0,
        foreground_pixels=foreground,
        component_count=0,
        inference_ms=0.0,
    )


def _ground_truth_for_frame(
    frame: DatasetFrame, positive_index: dict[str, GroundTruthFrame]
) -> GroundTruthFrame:
    """Возвращает штатный GT либо пустой scored GT для negative кадра."""

    if frame.frame_type == "positive":
        if frame.source_file not in positive_index:
            raise ValueError(f"Positive frame is absent from ground truth: {frame.source_file}")
        return positive_index[frame.source_file]
    return GroundTruthFrame(frame.source_file, (), ())


def compare_frame(
    frame: DatasetFrame,
    pytorch_map: np.ndarray,
    cix_map: np.ndarray,
    ground_truth: GroundTruthFrame,
) -> tuple[FrameComparison, Counter[str], Counter[str]]:
    """Считает pixel metrics и переиспользует штатный object matching."""

    pytorch = _prediction_result(pytorch_map)
    cix = _prediction_result(cix_map)
    pytorch_components = extract_predicted_components(pytorch)
    cix_components = extract_predicted_components(cix)
    pytorch_matching = match_frame_predictions(ground_truth, pytorch_components)
    cix_matching = match_frame_predictions(ground_truth, cix_components)
    difference = np.abs(
        pytorch.probability_map.astype(np.float64) - cix.probability_map.astype(np.float64)
    )
    union = np.count_nonzero(pytorch.binary_mask | cix.binary_mask)
    intersection = np.count_nonzero(pytorch.binary_mask & cix.binary_mask)
    mask_iou = float(intersection / union) if union else 1.0
    pytorch_size_tp = Counter(
        item.ground_truth.size_class
        for item in pytorch_matching.scored_matches if item.prediction is not None
    )
    cix_size_tp = Counter(
        item.ground_truth.size_class
        for item in cix_matching.scored_matches if item.prediction is not None
    )
    return FrameComparison(
        source_file=frame.source_file, frame_type=frame.frame_type,
        source_set=frame.source_set, size_class=frame.size_class,
        mae=float(difference.mean()),
        max_abs_error=float(difference.max()), mask_iou=mask_iou,
        threshold_flips=int(np.count_nonzero(pytorch.binary_mask ^ cix.binary_mask)),
        pytorch_foreground_pixels=pytorch.foreground_pixels,
        cix_foreground_pixels=cix.foreground_pixels,
        pytorch_components=len(pytorch_components), cix_components=len(cix_components),
        pytorch_tp=pytorch_matching.tp, pytorch_fn=pytorch_matching.fn,
        pytorch_fp=pytorch_matching.fp, cix_tp=cix_matching.tp,
        cix_fn=cix_matching.fn, cix_fp=cix_matching.fp,
    ), pytorch_size_tp, cix_size_tp


def _ratio(numerator: float, denominator: float) -> float:
    """Безопасно вычисляет отношение, возвращая ноль при пустом знаменателе."""

    return numerator / denominator if denominator else 0.0


def _add_negative_summary(
    result: dict[str, object], rows: Sequence[FrameComparison], runtime: str,
) -> None:
    """Добавляет штатные negative-метрики отдельно от object precision."""

    for source_set in (None, "clear_sky", "clear_horizon"):
        selected = [
            row for row in rows
            if row.frame_type == "negative"
            and (source_set is None or row.source_set == source_set)
        ]
        prefix = "negative" if source_set is None else source_set
        components = sum(getattr(row, f"{runtime}_components") for row in selected)
        pixels = sum(getattr(row, f"{runtime}_foreground_pixels") for row in selected)
        frames_with_fp = sum(
            getattr(row, f"{runtime}_components") > 0 for row in selected
        )
        result.update({
            f"{runtime}_{prefix}_frames": len(selected),
            f"{runtime}_{prefix}_frames_with_fp": frames_with_fp,
            f"{runtime}_{prefix}_frame_fp_rate": _ratio(frames_with_fp, len(selected)),
            f"{runtime}_{prefix}_fp_components": components,
            f"{runtime}_{prefix}_fp_components_per_frame": _ratio(
                components, len(selected)
            ),
            f"{runtime}_{prefix}_fp_pixels": pixels,
        })


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
    positive_rows = [row for row in rows if row.frame_type == "positive"]
    for runtime in ("pytorch", "cix"):
        tp = sum(getattr(row, f"{runtime}_tp") for row in positive_rows)
        fn = sum(getattr(row, f"{runtime}_fn") for row in positive_rows)
        fp = sum(getattr(row, f"{runtime}_fp") for row in positive_rows)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        result.update({
            f"{runtime}_tp": tp, f"{runtime}_fn": fn, f"{runtime}_fp": fp,
            f"{runtime}_precision": precision, f"{runtime}_recall": recall,
            f"{runtime}_f1": _ratio(2 * precision * recall, precision + recall),
        })
        _add_negative_summary(result, rows, runtime)
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

    parser = argparse.ArgumentParser(description="Compare ALCNet PyTorch and CIX outputs")
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
        ground_truth = _ground_truth_for_frame(frame, gt_index)
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
