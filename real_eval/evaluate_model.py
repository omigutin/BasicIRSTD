"""Объектная оценка одной BasicIRSTD-модели по bbox ground truth."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from math import hypot
from pathlib import Path
import shutil
import sys
from typing import Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import ModelConfig
from .model_runner import ModelRunner, PredictionResult
from .sources import IMAGE_EXTENSIONS, FrameData, ImageDirectorySource
from .visualization import to_display_image


MATCH_DISTANCE_PX = 3.0
SIZE_CLASSES = ("Tiny", "Small", "Medium", "Large")


@dataclass(frozen=True)
class GroundTruthRecord:
    """Один проверенный bbox ground truth."""

    image: str
    annotation_id: int
    size_class: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class PredictedComponent:
    """Одна связная компонента бинарного prediction."""

    index: int
    label: int
    x1: int
    y1: int
    x2: int
    y2: int
    centroid_x: float
    centroid_y: float
    area_pixels: int
    max_score: float


@dataclass(frozen=True)
class ObjectMatch:
    """Результат object-level matching одного GT bbox."""

    ground_truth: GroundTruthRecord
    prediction: Optional[PredictedComponent]
    distance_px: Optional[float]


@dataclass(frozen=True)
class PositiveObjectRow:
    """Строка объектного positive-отчёта."""

    image: str
    annotation_id: int
    size_class: str
    gt_x1: float
    gt_y1: float
    gt_x2: float
    gt_y2: float
    matched: bool
    prediction_index: Optional[int]
    prediction_centroid_x: Optional[float]
    prediction_centroid_y: Optional[float]
    match_distance_px: Optional[float]
    prediction_area: Optional[int]
    prediction_score: Optional[float]


@dataclass(frozen=True)
class PositiveImageRow:
    """Строка покадрового positive-отчёта."""

    image: str
    gt_count: int
    predicted_count: int
    tp: int
    fn: int
    fp: int
    tiny_gt: int
    tiny_tp: int
    small_gt: int
    small_tp: int
    medium_gt: int
    medium_tp: int
    large_gt: int
    large_tp: int
    inference_ms: float


@dataclass(frozen=True)
class NegativeImageRow:
    """Строка покадрового negative-отчёта."""

    image: str
    source_set: str
    predicted_count: int
    false_positive_pixels: int
    inference_ms: float


@dataclass
class NegativeStats:
    """Накопленные показатели одного negative-набора."""

    frames: int = 0
    frames_with_fp: int = 0
    fp_components: int = 0
    fp_pixels: int = 0
    inference_ms: float = 0.0
    evaluated_pixels: int = 0


def extract_predicted_components(result: PredictionResult) -> tuple[PredictedComponent, ...]:
    """Извлекает все компоненты без дополнительной фильтрации."""

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        result.binary_mask.astype(np.uint8), connectivity=8
    )
    components: list[PredictedComponent] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        component_scores = result.probability_map[labels == label]
        components.append(PredictedComponent(
            index=label,
            label=label,
            x1=x,
            y1=y,
            x2=x + width,
            y2=y + height,
            centroid_x=float(centroids[label, 0]),
            centroid_y=float(centroids[label, 1]),
            area_pixels=area,
            max_score=float(component_scores.max()),
        ))
    return tuple(components)


def distance_from_centroid_to_bbox(
    prediction: PredictedComponent,
    ground_truth: GroundTruthRecord,
) -> float:
    """Считает расстояние centroid до GT bbox для object detection, не segmentation."""

    nearest_x = min(max(prediction.centroid_x, ground_truth.x1), ground_truth.x2)
    nearest_y = min(max(prediction.centroid_y, ground_truth.y1), ground_truth.y2)
    return hypot(prediction.centroid_x - nearest_x, prediction.centroid_y - nearest_y)


def match_predictions_to_ground_truth(
    ground_truth: Sequence[GroundTruthRecord],
    predictions: Sequence[PredictedComponent],
    max_distance_px: float = MATCH_DISTANCE_PX,
) -> tuple[tuple[ObjectMatch, ...], tuple[PredictedComponent, ...]]:
    """Выполняет one-to-one bbox matching, а не оценку mask IoU/Dice."""

    if max_distance_px < 0:
        raise ValueError("Match distance must not be negative")
    if not ground_truth:
        return (), tuple(predictions)
    if not predictions:
        return tuple(ObjectMatch(item, None, None) for item in ground_truth), ()

    distances = np.asarray([
        [distance_from_centroid_to_bbox(prediction, gt) for prediction in predictions]
        for gt in ground_truth
    ], dtype=np.float64)
    # Недопустимый штраф больше любой суммы допустимых расстояний: сначала
    # максимизируется число matches, затем минимизируется общая дистанция.
    penalty = (max(len(ground_truth), len(predictions)) + 1) * (max_distance_px + 1.0)
    costs = np.where(distances <= max_distance_px, distances, penalty)
    gt_indices, prediction_indices = linear_sum_assignment(costs)
    accepted: dict[int, tuple[int, float]] = {}
    used_predictions: set[int] = set()
    for gt_index, prediction_index in zip(gt_indices, prediction_indices):
        distance = float(distances[gt_index, prediction_index])
        if distance <= max_distance_px:
            accepted[int(gt_index)] = (int(prediction_index), distance)
            used_predictions.add(int(prediction_index))

    matches = tuple(
        ObjectMatch(gt, predictions[accepted[index][0]], accepted[index][1])
        if index in accepted else ObjectMatch(gt, None, None)
        for index, gt in enumerate(ground_truth)
    )
    unmatched = tuple(
        prediction for index, prediction in enumerate(predictions)
        if index not in used_predictions
    )
    return matches, unmatched


def _require_columns(reader: csv.DictReader, required: Iterable[str], path: Path) -> None:
    """Проверяет обязательные колонки входного CSV."""

    missing = set(required) - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")


def load_ground_truth_objects(root: Path) -> Mapping[str, tuple[GroundTruthRecord, ...]]:
    """Загружает объектный индекс из подготовленного CSV."""

    path = root / "ground_truth_objects.csv"
    required = {"image", "annotation_id", "size_class", "x1", "y1", "x2", "y2"}
    grouped: dict[str, list[GroundTruthRecord]] = defaultdict(list)
    seen_ids: set[int] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        _require_columns(reader, required, path)
        for row in reader:
            annotation_id = int(row["annotation_id"])
            if annotation_id in seen_ids:
                raise ValueError(f"Duplicate annotation id: {annotation_id}")
            seen_ids.add(annotation_id)
            size_class = row["size_class"]
            if size_class not in SIZE_CLASSES:
                raise ValueError(f"Unsupported size class: {size_class}")
            record = GroundTruthRecord(
                image=Path(row["image"]).as_posix(), annotation_id=annotation_id,
                size_class=size_class, x1=float(row["x1"]), y1=float(row["y1"]),
                x2=float(row["x2"]), y2=float(row["y2"]),
            )
            if record.x2 <= record.x1 or record.y2 <= record.y1:
                raise ValueError(f"Invalid GT bbox for annotation {annotation_id}")
            grouped[record.image].append(record)
    return {name: tuple(items) for name, items in grouped.items()}


def _load_expected_images(root: Path) -> set[str]:
    """Загружает список positive-кадров из ground_truth_images.csv."""

    path = root / "ground_truth_images.csv"
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        _require_columns(reader, {"image", "gt_target_count"}, path)
        return {Path(row["image"]).as_posix() for row in reader}


def _load_negative_index(root: Path) -> Mapping[str, set[str]]:
    """Загружает ожидаемые negative-кадры по source set."""

    path = root / "negative_images.csv"
    grouped: dict[str, set[str]] = defaultdict(set)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        _require_columns(reader, {"image", "source_set"}, path)
        for row in reader:
            source_set = row["source_set"]
            if source_set not in {"clear_horizon", "clear_sky"}:
                raise ValueError(f"Unsupported negative source set: {source_set}")
            grouped[source_set].add(Path(row["image"]).as_posix())
    return grouped


def _directory_names(root: Path) -> set[str]:
    """Получает относительные имена кадров без чтения изображений."""

    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _write_rows(path: Path, row_type: type, rows: Sequence[object]) -> None:
    """Записывает dataclass-строки с фиксированным порядком полей."""

    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(row_type)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _apply_component_tint(
    image: np.ndarray,
    labels: np.ndarray,
    components: Sequence[PredictedComponent],
    color: tuple[int, int, int],
    alpha: float = 0.18,
) -> None:
    """Накладывает лёгкий цвет только на пиксели выбранных компонентов."""

    if not components:
        return
    component_labels = np.asarray([component.label for component in components])
    mask = np.isin(labels, component_labels)
    tinted = image[mask].astype(np.float32)
    image[mask] = np.clip(
        tinted * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def _centroid_inside_bbox(
    prediction: PredictedComponent,
    ground_truth: GroundTruthRecord,
) -> bool:
    """Проверяет попадание centroid в bbox уже найденного GT только для показа."""

    return (
        ground_truth.x1 <= prediction.centroid_x <= ground_truth.x2
        and ground_truth.y1 <= prediction.centroid_y <= ground_truth.y2
    )


def _draw_positive_visualization(
    image: np.ndarray,
    result: PredictionResult,
    matches: Sequence[ObjectMatch],
    unmatched: Sequence[PredictedComponent],
) -> np.ndarray:
    """Рисует GT жёлтым, matched зелёным и независимые FP красным."""

    output = to_display_image(image)
    _, labels = cv2.connectedComponents(
        result.binary_mask.astype(np.uint8), connectivity=8
    )
    matched_predictions = tuple(
        match.prediction for match in matches if match.prediction is not None
    )
    matched_ground_truth = tuple(
        match.ground_truth for match in matches if match.prediction is not None
    )
    visible_unmatched = tuple(
        prediction
        for prediction in unmatched
        if not any(
            _centroid_inside_bbox(prediction, ground_truth)
            for ground_truth in matched_ground_truth
        )
    )

    _apply_component_tint(output, labels, matched_predictions, (0, 255, 0))
    _apply_component_tint(output, labels, visible_unmatched, (255, 0, 0))

    # GT всегда остаётся жёлтым: сочетание yellow + green означает найденную цель.
    for match in matches:
        gt = match.ground_truth
        cv2.rectangle(
            output,
            (round(gt.x1), round(gt.y1)),
            (round(gt.x2), round(gt.y2)),
            (255, 255, 0),
            1,
        )
        if match.prediction is not None:
            pred = match.prediction
            cv2.rectangle(
                output, (pred.x1, pred.y1), (pred.x2, pred.y2), (0, 255, 0), 1
            )
    for pred in visible_unmatched:
        cv2.rectangle(
            output, (pred.x1, pred.y1), (pred.x2, pred.y2), (255, 0, 0), 1
        )
    return output


def _draw_negative_error(
    image: np.ndarray,
    result: PredictionResult,
    predictions: Sequence[PredictedComponent],
) -> np.ndarray:
    """Рисует negative predictions красной маской и bbox без текста."""

    output = to_display_image(image)
    _, labels = cv2.connectedComponents(
        result.binary_mask.astype(np.uint8), connectivity=8
    )
    _apply_component_tint(output, labels, predictions, (255, 0, 0))
    for prediction in predictions:
        cv2.rectangle(
            output,
            (prediction.x1, prediction.y1),
            (prediction.x2, prediction.y2),
            (255, 0, 0),
            1,
        )
    return output


def _save_error(path: Path, image: np.ndarray) -> None:
    """Сохраняет RGB error image с относительной структурой."""

    output_path = path if path.suffix.lower() == ".png" else path.with_name(f"{path.name}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Cannot write error image: {output_path}")


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Возвращает ноль для отношения с нулевым знаменателем."""

    return numerator / denominator if denominator else 0.0


def _evaluate_positive(
    runner: ModelRunner,
    source: ImageDirectorySource,
    gt_index: Mapping[str, tuple[GroundTruthRecord, ...]],
    visualizations_root: Path,
) -> tuple[list[PositiveImageRow], list[PositiveObjectRow], int, int, float]:
    """Оценивает positive-набор и возвращает false pixels, pixels и timing."""

    image_rows: list[PositiveImageRow] = []
    object_rows: list[PositiveObjectRow] = []
    false_pixels = total_pixels = 0
    total_inference_ms = 0.0
    for frame in source:
        result = runner.predict(frame.image)
        predictions = extract_predicted_components(result)
        gt_objects = gt_index[frame.source_name]
        matches, unmatched = match_predictions_to_ground_truth(gt_objects, predictions)
        matched = [item for item in matches if item.prediction is not None]
        size_gt = Counter(item.size_class for item in gt_objects)
        size_tp = Counter(item.ground_truth.size_class for item in matched)
        row = PositiveImageRow(
            image=frame.source_name, gt_count=len(gt_objects), predicted_count=len(predictions),
            tp=len(matched), fn=len(gt_objects) - len(matched), fp=len(unmatched),
            tiny_gt=size_gt["Tiny"], tiny_tp=size_tp["Tiny"],
            small_gt=size_gt["Small"], small_tp=size_tp["Small"],
            medium_gt=size_gt["Medium"], medium_tp=size_tp["Medium"],
            large_gt=size_gt["Large"], large_tp=size_tp["Large"],
            inference_ms=result.inference_ms,
        )
        image_rows.append(row)
        for item in matches:
            pred = item.prediction
            object_rows.append(PositiveObjectRow(
                image=frame.source_name, annotation_id=item.ground_truth.annotation_id,
                size_class=item.ground_truth.size_class,
                gt_x1=item.ground_truth.x1, gt_y1=item.ground_truth.y1,
                gt_x2=item.ground_truth.x2, gt_y2=item.ground_truth.y2,
                matched=pred is not None, prediction_index=pred.index if pred else None,
                prediction_centroid_x=pred.centroid_x if pred else None,
                prediction_centroid_y=pred.centroid_y if pred else None,
                match_distance_px=item.distance_px,
                prediction_area=pred.area_pixels if pred else None,
                prediction_score=pred.max_score if pred else None,
            ))
        false_pixels += sum(item.area_pixels for item in unmatched)
        total_pixels += int(frame.image.shape[0] * frame.image.shape[1])
        total_inference_ms += result.inference_ms
        _save_error(
            visualizations_root / "positive" / frame.source_name,
            _draw_positive_visualization(frame.image, result, matches, unmatched),
        )
    return image_rows, object_rows, false_pixels, total_pixels, total_inference_ms


def _evaluate_negative(
    runner: ModelRunner,
    source: ImageDirectorySource,
    source_set: str,
    errors_root: Path,
) -> tuple[list[NegativeImageRow], NegativeStats]:
    """Оценивает один negative-набор, где каждая компонента является FP."""

    rows: list[NegativeImageRow] = []
    stats = NegativeStats()
    for frame in source:
        result = runner.predict(frame.image)
        predictions = extract_predicted_components(result)
        row = NegativeImageRow(frame.source_name, source_set, len(predictions),
                               result.foreground_pixels, result.inference_ms)
        rows.append(row)
        stats.frames += 1
        stats.frames_with_fp += int(bool(predictions))
        stats.fp_components += len(predictions)
        stats.fp_pixels += result.foreground_pixels
        stats.inference_ms += result.inference_ms
        stats.evaluated_pixels += int(frame.image.shape[0] * frame.image.shape[1])
        if predictions:
            _save_error(
                errors_root / source_set / frame.source_name,
                _draw_negative_error(frame.image, result, predictions),
            )
    return rows, stats


def _build_summary(
    config: ModelConfig,
    positives: Sequence[PositiveImageRow],
    positive_false_pixels: int,
    positive_pixels: int,
    positive_inference_ms: float,
    negatives: Mapping[str, NegativeStats],
) -> dict[str, object]:
    """Собирает object-, frame-, pixel- и performance-метрики запуска."""

    tp, fn, fp = (sum(getattr(row, name) for row in positives) for name in ("tp", "fn", "fp"))
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    all_negative = NegativeStats(
        frames=sum(item.frames for item in negatives.values()),
        frames_with_fp=sum(item.frames_with_fp for item in negatives.values()),
        fp_components=sum(item.fp_components for item in negatives.values()),
        fp_pixels=sum(item.fp_pixels for item in negatives.values()),
        inference_ms=sum(item.inference_ms for item in negatives.values()),
        evaluated_pixels=sum(item.evaluated_pixels for item in negatives.values()),
    )
    total_frames = len(positives) + all_negative.frames
    overall_ms = _safe_ratio(positive_inference_ms + all_negative.inference_ms, total_frames)
    summary: dict[str, object] = {
        "model": config.model_name, "train_dataset": config.train_dataset_name,
        "checkpoint": config.checkpoint_path.name, "threshold": config.threshold,
        "positive_frames": len(positives), "gt_objects": tp + fn,
        "tp": tp, "fn": fn, "fp": fp, "precision": precision, "recall": recall,
        "f1": _safe_ratio(2 * precision * recall, precision + recall),
        "positive_frames_with_all_targets_detected": sum(row.fn == 0 for row in positives),
        "positive_frames_with_missed_targets": sum(row.fn > 0 for row in positives),
        "positive_frames_with_false_positives": sum(row.fp > 0 for row in positives),
        "total_predicted_components_positive": sum(row.predicted_count for row in positives),
    }
    for prefix in ("tiny", "small", "medium", "large"):
        gt = sum(getattr(row, f"{prefix}_gt") for row in positives)
        size_tp = sum(getattr(row, f"{prefix}_tp") for row in positives)
        summary.update({f"{prefix}_gt": gt, f"{prefix}_tp": size_tp,
                        f"{prefix}_fn": gt - size_tp,
                        f"{prefix}_recall": _safe_ratio(size_tp, gt)})
    summary.update({
        "negative_frames": all_negative.frames,
        "negative_frames_with_fp": all_negative.frames_with_fp,
        "negative_frame_fp_rate": _safe_ratio(all_negative.frames_with_fp, all_negative.frames),
        "negative_fp_components": all_negative.fp_components,
        "negative_fp_components_per_frame": _safe_ratio(all_negative.fp_components, all_negative.frames),
        "negative_fp_pixels": all_negative.fp_pixels,
    })
    for source_set in ("clear_horizon", "clear_sky"):
        stats = negatives[source_set]
        summary.update({
            f"{source_set}_frames": stats.frames,
            f"{source_set}_frames_with_fp": stats.frames_with_fp,
            f"{source_set}_frame_fp_rate": _safe_ratio(stats.frames_with_fp, stats.frames),
            f"{source_set}_fp_components": stats.fp_components,
            f"{source_set}_fp_components_per_frame": _safe_ratio(stats.fp_components, stats.frames),
            f"{source_set}_fp_pixels": stats.fp_pixels,
        })
    summary.update({
        "false_alarm_pixel_rate": _safe_ratio(
            positive_false_pixels + all_negative.fp_pixels,
            positive_pixels + all_negative.evaluated_pixels),
        "positive_avg_inference_ms": _safe_ratio(positive_inference_ms, len(positives)),
        "negative_avg_inference_ms": _safe_ratio(all_negative.inference_ms, all_negative.frames),
        "overall_avg_inference_ms": overall_ms, "avg_inference_ms": overall_ms,
        "fps_inference": _safe_ratio(1000.0, overall_ms),
    })
    return summary


def _write_dict_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Записывает словарные строки с общей схемой."""

    if not rows:
        raise ValueError("Cannot write an empty summary")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _update_model_comparison(path: Path, summary: Mapping[str, object]) -> None:
    """Атомарно обновляет run по model/dataset/checkpoint/threshold."""

    key_fields = ("model", "train_dataset", "checkpoint", "threshold")
    rows: list[dict[str, object]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))
    key = tuple(str(summary[field]) for field in key_fields)
    rows = [row for row in rows if tuple(str(row.get(field, "")) for field in key_fields) != key]
    rows.append(dict(summary))
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_dict_rows(temporary, rows)
    temporary.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI одной BasicIRSTD evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate one BasicIRSTD model")
    parser.add_argument("--positive", required=True, type=Path)
    parser.add_argument("--clear-horizon", required=True, type=Path)
    parser.add_argument("--clear-sky", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--dataset-dir", default=Path("datasets"), type=Path)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--device", default="cuda")
    return parser


def run(args: argparse.Namespace) -> Path:
    """Выполняет warmup, три прогона и запись evaluation-отчётов."""

    config = ModelConfig(args.model, args.checkpoint, args.train_dataset,
                         args.dataset_dir, args.threshold, args.device)
    positive_source = ImageDirectorySource(args.positive)
    negative_sources = {
        "clear_horizon": ImageDirectorySource(args.clear_horizon),
        "clear_sky": ImageDirectorySource(args.clear_sky),
    }
    gt_index = load_ground_truth_objects(args.ground_truth)
    expected_positive = _load_expected_images(args.ground_truth)
    negative_index = _load_negative_index(args.ground_truth)
    actual_positive = _directory_names(args.positive)
    if actual_positive != expected_positive or actual_positive != set(gt_index):
        raise ValueError("Positive dataset does not match ground truth CSV indexes")
    for source_set, source in negative_sources.items():
        root = args.clear_horizon if source_set == "clear_horizon" else args.clear_sky
        if _directory_names(root) != negative_index.get(source_set, set()):
            raise ValueError(f"Negative dataset does not match CSV index: {source_set}")

    run_root = args.output / f"{args.model}_{args.train_dataset}"
    run_root.mkdir(parents=True, exist_ok=True)
    errors_root = run_root / "errors"
    if errors_root.exists():
        shutil.rmtree(errors_root)
    errors_root.mkdir()
    visualizations_root = run_root / "visualizations"
    positive_visualizations = visualizations_root / "positive"
    if positive_visualizations.exists():
        shutil.rmtree(positive_visualizations)
    positive_visualizations.mkdir(parents=True)

    runner = ModelRunner(config)
    first_frame: FrameData = next(iter(positive_source))
    runner.predict(first_frame.image)  # warmup не входит ни в отчёты, ни в timing

    positives, objects, positive_false_pixels, positive_pixels, positive_ms = (
        _evaluate_positive(
            runner, positive_source, gt_index, visualizations_root
        )
    )
    negative_rows: list[NegativeImageRow] = []
    negative_stats: dict[str, NegativeStats] = {}
    for source_set, source in negative_sources.items():
        rows, stats = _evaluate_negative(runner, source, source_set, errors_root)
        negative_rows.extend(rows)
        negative_stats[source_set] = stats

    summary = _build_summary(config, positives, positive_false_pixels,
                             positive_pixels, positive_ms, negative_stats)
    _write_rows(run_root / "positive_images.csv", PositiveImageRow, positives)
    _write_rows(run_root / "positive_objects.csv", PositiveObjectRow, objects)
    _write_rows(run_root / "negative_images.csv", NegativeImageRow, negative_rows)
    _write_dict_rows(run_root / "summary.csv", [summary])
    args.output.mkdir(parents=True, exist_ok=True)
    _update_model_comparison(args.output / "model_comparison.csv", summary)
    return run_root


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Обрабатывает ожидаемые ошибки CLI и возвращает exit code."""

    try:
        output = run(build_argument_parser().parse_args(argv))
        print(f"Evaluation completed: {output}")
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
