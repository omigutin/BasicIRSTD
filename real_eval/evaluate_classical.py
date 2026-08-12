"""Объективный benchmark классического морфологического Top-Hat."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import shutil
import sys
from typing import Mapping, Optional, Sequence



def _load_runtime_dependencies() -> None:
    """Лениво загружает OpenCV и существующее evaluation-ядро после CLI."""

    if "TopHatRunner" in globals():
        return
    import cv2 as cv2_module
    import numpy as numpy_module
    from .classical_runner import BorderType as border_type
    from .classical_runner import ClassicalDetection as classical_detection
    from .classical_runner import ClassicalMethod as classical_method
    from .classical_runner import ClassicalPredictionResult as prediction_result
    from .classical_runner import StructuringElementShape as element_shape
    from .classical_runner import ThresholdStrategy as threshold_strategy
    from .classical_runner import TopHatConfig as top_hat_config
    from .classical_runner import TopHatRunner as top_hat_runner
    from . import evaluate_model as evaluation
    from .sources import ImageDirectorySource as image_directory_source
    from .visualization import to_display_image as display_image

    globals().update({
        "cv2": cv2_module,
        "np": numpy_module,
        "BorderType": border_type,
        "ClassicalDetection": classical_detection,
        "ClassicalMethod": classical_method,
        "ClassicalPredictionResult": prediction_result,
        "StructuringElementShape": element_shape,
        "ThresholdStrategy": threshold_strategy,
        "TopHatConfig": top_hat_config,
        "TopHatRunner": top_hat_runner,
        "GroundTruthRecord": evaluation.GroundTruthRecord,
        "MATCH_DISTANCE_PX": evaluation.MATCH_DISTANCE_PX,
        "ObjectMatch": evaluation.ObjectMatch,
        "PositiveObjectRow": evaluation.PositiveObjectRow,
        "PredictedComponent": evaluation.PredictedComponent,
        "_directory_names": evaluation._directory_names,
        "_load_expected_images": evaluation._load_expected_images,
        "_load_negative_index": evaluation._load_negative_index,
        "_safe_ratio": evaluation._safe_ratio,
        "_write_dict_rows": evaluation._write_dict_rows,
        "load_ground_truth_objects": evaluation.load_ground_truth_objects,
        "match_predictions_to_ground_truth": evaluation.match_predictions_to_ground_truth,
        "ImageDirectorySource": image_directory_source,
        "to_display_image": display_image,
    })


@dataclass(frozen=True)
class ClassicalPositiveImageRow:
    """Хранит object-метрики и timing одного positive-кадра."""

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
    algorithm_ms: float
    processing_ms: float
    inference_ms: float


@dataclass(frozen=True)
class ClassicalNegativeImageRow:
    """Хранит detections, pixels и timing одного negative-кадра."""

    image: str
    source_set: str
    predicted_count: int
    false_positive_pixels: int
    algorithm_ms: float
    processing_ms: float
    inference_ms: float


@dataclass
class ClassicalNegativeStats:
    """Накапливает показатели одного negative-набора."""

    frames: int = 0
    frames_with_fp: int = 0
    fp_components: int = 0
    fp_pixels: int = 0
    algorithm_ms: float = 0.0
    processing_ms: float = 0.0
    evaluated_pixels: int = 0


@dataclass(frozen=True)
class PositiveEvaluation:
    """Объединяет positive-отчёты, pixel counters и timing."""

    image_rows: tuple[ClassicalPositiveImageRow, ...]
    object_rows: tuple[PositiveObjectRow, ...]
    false_positive_pixels: int
    evaluated_pixels: int
    algorithm_ms: float
    processing_ms: float


def _to_predicted_component(detection: ClassicalDetection) -> PredictedComponent:
    """Адаптирует classical detection к существующему общему matcher."""

    return PredictedComponent(
        index=detection.index,
        label=detection.label,
        x1=detection.x1,
        y1=detection.y1,
        x2=detection.x2,
        y2=detection.y2,
        centroid_x=detection.centroid_x,
        centroid_y=detection.centroid_y,
        area_pixels=detection.area_pixels,
        max_score=detection.max_response,
    )


def _write_rows(path: Path, row_type: type, rows: Sequence[object]) -> None:
    """Записывает типизированные строки с фиксированным порядком колонок."""

    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[field.name for field in fields(row_type)]
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _apply_component_tint(
    image: np.ndarray,
    labels: np.ndarray,
    components: Sequence[PredictedComponent],
    color: tuple[int, int, int],
    alpha: float = 0.18,
) -> None:
    """Добавляет лёгкий цвет на pixels выбранных компонентов."""

    if not components:
        return
    selected_labels = np.asarray([component.label for component in components])
    mask = np.isin(labels, selected_labels)
    pixels = image[mask].astype(np.float32)
    image[mask] = np.clip(
        pixels * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def _draw_positive(
    image: np.ndarray,
    result: ClassicalPredictionResult,
    matches: Sequence[ObjectMatch],
    unmatched: Sequence[PredictedComponent],
) -> np.ndarray:
    """Рисует GT жёлтым, TP зелёным и FP красным без текста."""

    output = to_display_image(image)
    matched = tuple(
        item.prediction for item in matches if item.prediction is not None
    )
    _apply_component_tint(output, result.labels, matched, (0, 255, 0))
    _apply_component_tint(output, result.labels, unmatched, (255, 0, 0))
    for item in matches:
        ground_truth = item.ground_truth
        cv2.rectangle(
            output,
            (round(ground_truth.x1), round(ground_truth.y1)),
            (round(ground_truth.x2), round(ground_truth.y2)),
            (255, 255, 0),
            1,
        )
        if item.prediction is not None:
            prediction = item.prediction
            cv2.rectangle(
                output,
                (prediction.x1, prediction.y1),
                (prediction.x2, prediction.y2),
                (0, 255, 0),
                1,
            )
    for prediction in unmatched:
        cv2.rectangle(
            output,
            (prediction.x1, prediction.y1),
            (prediction.x2, prediction.y2),
            (255, 0, 0),
            1,
        )
    return output


def _draw_negative(
    image: np.ndarray,
    result: ClassicalPredictionResult,
    predictions: Sequence[PredictedComponent],
) -> np.ndarray:
    """Рисует все negative detections красными без текста."""

    output = to_display_image(image)
    _apply_component_tint(output, result.labels, predictions, (255, 0, 0))
    for prediction in predictions:
        cv2.rectangle(
            output,
            (prediction.x1, prediction.y1),
            (prediction.x2, prediction.y2),
            (255, 0, 0),
            1,
        )
    return output


def _save_rgb(path: Path, image: np.ndarray) -> None:
    """Сохраняет RGB-картинку как PNG с относительной структурой."""

    output = path if path.suffix.lower() == ".png" else path.with_name(
        f"{path.name}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Cannot write visualization: {output}")


def _evaluate_positive(
    runner: TopHatRunner,
    source: ImageDirectorySource,
    gt_index: Mapping[str, tuple[GroundTruthRecord, ...]],
    visualizations_root: Path,
) -> PositiveEvaluation:
    """Оценивает positive-набор общим centroid-to-bbox matcher."""

    image_rows: list[ClassicalPositiveImageRow] = []
    object_rows: list[PositiveObjectRow] = []
    false_positive_pixels = 0
    evaluated_pixels = 0
    algorithm_ms = 0.0
    processing_ms = 0.0
    for frame in source:
        result = runner.predict(frame.image)
        predictions = tuple(
            _to_predicted_component(item) for item in result.detections
        )
        ground_truth = gt_index[frame.source_name]
        matches, unmatched = match_predictions_to_ground_truth(
            ground_truth, predictions, MATCH_DISTANCE_PX
        )
        matched = tuple(item for item in matches if item.prediction is not None)
        size_gt = Counter(item.size_class for item in ground_truth)
        size_tp = Counter(item.ground_truth.size_class for item in matched)
        image_rows.append(ClassicalPositiveImageRow(
            image=frame.source_name,
            gt_count=len(ground_truth),
            predicted_count=len(predictions),
            tp=len(matched),
            fn=len(ground_truth) - len(matched),
            fp=len(unmatched),
            tiny_gt=size_gt["Tiny"],
            tiny_tp=size_tp["Tiny"],
            small_gt=size_gt["Small"],
            small_tp=size_tp["Small"],
            medium_gt=size_gt["Medium"],
            medium_tp=size_tp["Medium"],
            large_gt=size_gt["Large"],
            large_tp=size_tp["Large"],
            algorithm_ms=result.algorithm_ms,
            processing_ms=result.processing_ms,
            inference_ms=result.processing_ms,
        ))
        for item in matches:
            prediction = item.prediction
            object_rows.append(PositiveObjectRow(
                image=frame.source_name,
                annotation_id=item.ground_truth.annotation_id,
                size_class=item.ground_truth.size_class,
                gt_x1=item.ground_truth.x1,
                gt_y1=item.ground_truth.y1,
                gt_x2=item.ground_truth.x2,
                gt_y2=item.ground_truth.y2,
                matched=prediction is not None,
                prediction_index=prediction.index if prediction else None,
                prediction_centroid_x=prediction.centroid_x if prediction else None,
                prediction_centroid_y=prediction.centroid_y if prediction else None,
                match_distance_px=item.distance_px,
                prediction_area=prediction.area_pixels if prediction else None,
                prediction_score=prediction.max_score if prediction else None,
            ))
        false_positive_pixels += sum(item.area_pixels for item in unmatched)
        evaluated_pixels += int(frame.image.shape[0] * frame.image.shape[1])
        algorithm_ms += result.algorithm_ms
        processing_ms += result.processing_ms
        _save_rgb(
            visualizations_root / "positive" / frame.source_name,
            _draw_positive(frame.image, result, matches, unmatched),
        )
    return PositiveEvaluation(
        tuple(image_rows),
        tuple(object_rows),
        false_positive_pixels,
        evaluated_pixels,
        algorithm_ms,
        processing_ms,
    )


def _evaluate_negative(
    runner: TopHatRunner,
    source: ImageDirectorySource,
    source_set: str,
    errors_root: Path,
) -> tuple[list[ClassicalNegativeImageRow], ClassicalNegativeStats]:
    """Оценивает negative-набор, где каждый компонент является FP."""

    rows: list[ClassicalNegativeImageRow] = []
    stats = ClassicalNegativeStats()
    for frame in source:
        result = runner.predict(frame.image)
        predictions = tuple(
            _to_predicted_component(item) for item in result.detections
        )
        rows.append(ClassicalNegativeImageRow(
            image=frame.source_name,
            source_set=source_set,
            predicted_count=len(predictions),
            false_positive_pixels=result.foreground_pixels,
            algorithm_ms=result.algorithm_ms,
            processing_ms=result.processing_ms,
            inference_ms=result.processing_ms,
        ))
        stats.frames += 1
        stats.frames_with_fp += int(bool(predictions))
        stats.fp_components += len(predictions)
        stats.fp_pixels += result.foreground_pixels
        stats.algorithm_ms += result.algorithm_ms
        stats.processing_ms += result.processing_ms
        stats.evaluated_pixels += int(frame.image.shape[0] * frame.image.shape[1])
        if predictions:
            _save_rgb(
                errors_root / source_set / frame.source_name,
                _draw_negative(frame.image, result, predictions),
            )
    return rows, stats


def _build_summary(
    config: TopHatConfig,
    positives: PositiveEvaluation,
    negatives: Mapping[str, ClassicalNegativeStats],
) -> dict[str, object]:
    """Формирует общие object-, pixel-, frame- и timing-метрики."""

    tp = sum(row.tp for row in positives.image_rows)
    fn = sum(row.fn for row in positives.image_rows)
    fp = sum(row.fp for row in positives.image_rows)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    negative_frames = sum(item.frames for item in negatives.values())
    negative_frames_with_fp = sum(item.frames_with_fp for item in negatives.values())
    negative_components = sum(item.fp_components for item in negatives.values())
    negative_pixels = sum(item.fp_pixels for item in negatives.values())
    negative_algorithm_ms = sum(item.algorithm_ms for item in negatives.values())
    negative_processing_ms = sum(item.processing_ms for item in negatives.values())
    negative_evaluated_pixels = sum(item.evaluated_pixels for item in negatives.values())
    total_frames = len(positives.image_rows) + negative_frames
    total_algorithm_ms = positives.algorithm_ms + negative_algorithm_ms
    total_processing_ms = positives.processing_ms + negative_processing_ms
    overall_algorithm_ms = _safe_ratio(total_algorithm_ms, total_frames)
    overall_processing_ms = _safe_ratio(total_processing_ms, total_frames)
    summary: dict[str, object] = {
        "runner_type": "classical",
        "model": "Top-Hat",
        "train_dataset": "none",
        "checkpoint": "",
        "method": config.method.value,
        "threshold": config.threshold,
        "threshold_type": "top_hat_response_absolute",
        "threshold_strategy": config.threshold_strategy.value,
        "kernel_size": config.kernel_size,
        "kernel_shape": config.kernel_shape.value,
        "border_type": config.border_type.value,
        "connectivity": config.connectivity,
        "minimum_component_area": config.minimum_component_area,
        "positive_frames": len(positives.image_rows),
        "gt_objects": tp + fn,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
        "positive_frames_with_all_targets_detected": sum(
            row.fn == 0 for row in positives.image_rows
        ),
        "positive_frames_with_missed_targets": sum(
            row.fn > 0 for row in positives.image_rows
        ),
        "positive_frames_with_false_positives": sum(
            row.fp > 0 for row in positives.image_rows
        ),
        "total_predicted_components_positive": sum(
            row.predicted_count for row in positives.image_rows
        ),
    }
    for prefix in ("tiny", "small", "medium", "large"):
        gt_count = sum(
            getattr(row, f"{prefix}_gt") for row in positives.image_rows
        )
        size_tp = sum(
            getattr(row, f"{prefix}_tp") for row in positives.image_rows
        )
        summary.update({
            f"{prefix}_gt": gt_count,
            f"{prefix}_tp": size_tp,
            f"{prefix}_fn": gt_count - size_tp,
            f"{prefix}_recall": _safe_ratio(size_tp, gt_count),
        })
    summary.update({
        "negative_frames": negative_frames,
        "negative_frames_with_fp": negative_frames_with_fp,
        "negative_frame_fp_rate": _safe_ratio(
            negative_frames_with_fp, negative_frames
        ),
        "negative_fp_components": negative_components,
        "negative_fp_components_per_frame": _safe_ratio(
            negative_components, negative_frames
        ),
        "negative_fp_pixels": negative_pixels,
    })
    for source_set in ("clear_horizon", "clear_sky"):
        stats = negatives[source_set]
        summary.update({
            f"{source_set}_frames": stats.frames,
            f"{source_set}_frames_with_fp": stats.frames_with_fp,
            f"{source_set}_frame_fp_rate": _safe_ratio(
                stats.frames_with_fp, stats.frames
            ),
            f"{source_set}_fp_components": stats.fp_components,
            f"{source_set}_fp_components_per_frame": _safe_ratio(
                stats.fp_components, stats.frames
            ),
            f"{source_set}_fp_pixels": stats.fp_pixels,
        })
    summary.update({
        "false_alarm_pixel_rate": _safe_ratio(
            positives.false_positive_pixels + negative_pixels,
            positives.evaluated_pixels + negative_evaluated_pixels,
        ),
        "positive_avg_algorithm_ms": _safe_ratio(
            positives.algorithm_ms, len(positives.image_rows)
        ),
        "negative_avg_algorithm_ms": _safe_ratio(
            negative_algorithm_ms, negative_frames
        ),
        "overall_avg_algorithm_ms": overall_algorithm_ms,
        "positive_avg_processing_ms": _safe_ratio(
            positives.processing_ms, len(positives.image_rows)
        ),
        "negative_avg_processing_ms": _safe_ratio(
            negative_processing_ms, negative_frames
        ),
        "overall_avg_processing_ms": overall_processing_ms,
        "fps_processing": _safe_ratio(1000.0, overall_processing_ms),
        "positive_avg_inference_ms": _safe_ratio(
            positives.processing_ms, len(positives.image_rows)
        ),
        "negative_avg_inference_ms": _safe_ratio(
            negative_processing_ms, negative_frames
        ),
        "overall_avg_inference_ms": overall_processing_ms,
        "avg_inference_ms": overall_processing_ms,
        "fps_inference": _safe_ratio(1000.0, overall_processing_ms),
    })
    return summary


def _update_comparison(path: Path, summary: Mapping[str, object]) -> None:
    """Атомарно заменяет classical run с тем же полным ключом."""

    key_fields = (
        "runner_type",
        "model",
        "method",
        "threshold_strategy",
        "threshold",
        "kernel_size",
        "kernel_shape",
        "border_type",
    )
    rows: list[dict[str, object]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))
    key = tuple(str(summary[field]) for field in key_fields)
    rows = [
        row for row in rows
        if tuple(str(row.get(field, "")) for field in key_fields) != key
    ]
    rows.append(dict(summary))
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_dict_rows(temporary, rows)
    temporary.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI зафиксированного classical benchmark."""

    parser = argparse.ArgumentParser(description="Evaluate classical IRSTD Top-Hat")
    parser.add_argument("--positive", required=True, type=Path)
    parser.add_argument("--clear-horizon", required=True, type=Path)
    parser.add_argument("--clear-sky", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", default="top_hat", choices=["top_hat"])
    parser.add_argument("--kernel-size", default=9, type=int)
    parser.add_argument(
        "--kernel-shape",
        default="ellipse",
        choices=["rectangle", "ellipse", "cross"],
    )
    parser.add_argument("--threshold-strategy", default="fixed", choices=["fixed"])
    parser.add_argument("--threshold", default=10.0, type=float)
    parser.add_argument("--border-type", default="reflect101", choices=["reflect101"])
    return parser


def _build_config(args: argparse.Namespace) -> TopHatConfig:
    """Преобразует проверенные CLI-значения в типизированный config."""

    return TopHatConfig(
        method=ClassicalMethod(args.method),
        kernel_size=args.kernel_size,
        kernel_shape=StructuringElementShape(args.kernel_shape),
        threshold_strategy=ThresholdStrategy(args.threshold_strategy),
        threshold=args.threshold,
        border_type=BorderType(args.border_type),
        connectivity=8,
        minimum_component_area=1,
    )


def run(args: argparse.Namespace) -> Path:
    """Проверяет datasets, выполняет benchmark и сохраняет отчёты."""

    _load_runtime_dependencies()
    config = _build_config(args)
    runner = TopHatRunner(config)
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
    for source_set in ("clear_horizon", "clear_sky"):
        root = args.clear_horizon if source_set == "clear_horizon" else args.clear_sky
        if _directory_names(root) != negative_index.get(source_set, set()):
            raise ValueError(f"Negative dataset does not match CSV index: {source_set}")

    run_root = args.output / "Top-Hat_classical"
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

    positives = _evaluate_positive(
        runner, positive_source, gt_index, visualizations_root
    )
    negative_rows: list[ClassicalNegativeImageRow] = []
    negative_stats: dict[str, ClassicalNegativeStats] = {}
    for source_set, source in negative_sources.items():
        rows, stats = _evaluate_negative(runner, source, source_set, errors_root)
        negative_rows.extend(rows)
        negative_stats[source_set] = stats

    summary = _build_summary(config, positives, negative_stats)
    _write_rows(
        run_root / "positive_images.csv",
        ClassicalPositiveImageRow,
        positives.image_rows,
    )
    _write_rows(
        run_root / "positive_objects.csv", PositiveObjectRow, positives.object_rows
    )
    _write_rows(
        run_root / "negative_images.csv", ClassicalNegativeImageRow, negative_rows
    )
    _write_dict_rows(run_root / "summary.csv", [summary])
    _update_comparison(args.output / "model_comparison.csv", summary)
    return run_root


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Печатает понятную CLI-ошибку и возвращает exit code."""

    try:
        output = run(build_argument_parser().parse_args(argv))
        print(f"Evaluation completed: {output}")
    except (
        FileNotFoundError,
        ImportError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
