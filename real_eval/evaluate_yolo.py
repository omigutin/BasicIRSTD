"""Объективная bbox-оценка пользовательской Ultralytics YOLO-модели."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Optional, Sequence


def _load_runtime_dependencies() -> None:
    """Загружает CV/ML-зависимости только после разбора CLI-параметров."""

    import cv2 as cv2_module
    import numpy as numpy_module
    from . import evaluate_model as evaluation
    from .sources import FrameData as frame_data
    from .sources import ImageDirectorySource as image_directory_source
    from .visualization import to_display_image as display_image
    from .yolo_runner import YoloDetection as yolo_detection
    from .yolo_runner import YoloModelConfig as yolo_model_config
    from .yolo_runner import YoloModelRunner as yolo_model_runner

    globals().update({
        "cv2": cv2_module,
        "np": numpy_module,
        "GroundTruthRecord": evaluation.GroundTruthRecord,
        "PositiveImageRow": evaluation.PositiveImageRow,
        "PositiveObjectRow": evaluation.PositiveObjectRow,
        "PredictedComponent": evaluation.PredictedComponent,
        "_directory_names": evaluation._directory_names,
        "_load_expected_images": evaluation._load_expected_images,
        "_load_negative_index": evaluation._load_negative_index,
        "_safe_ratio": evaluation._safe_ratio,
        "_write_dict_rows": evaluation._write_dict_rows,
        "_write_rows": evaluation._write_rows,
        "load_ground_truth_objects": evaluation.load_ground_truth_objects,
        "match_predictions_to_ground_truth": evaluation.match_predictions_to_ground_truth,
        "FrameData": frame_data,
        "ImageDirectorySource": image_directory_source,
        "to_display_image": display_image,
        "YoloDetection": yolo_detection,
        "YoloModelConfig": yolo_model_config,
        "YoloModelRunner": yolo_model_runner,
    })


@dataclass(frozen=True)
class YoloNegativeImageRow:
    """Одна строка negative YOLO-отчёта без pixel-метрики."""

    image: str
    source_set: str
    predicted_count: int
    false_positive_pixels: Optional[int]
    inference_ms: float


@dataclass
class YoloNegativeStats:
    """Накопленные object/frame показатели negative-набора."""

    frames: int = 0
    frames_with_fp: int = 0
    fp_components: int = 0
    inference_ms: float = 0.0


def _to_matching_component(detection: YoloDetection) -> PredictedComponent:
    """Адаптирует centroid к общему matcher без вычисления pixel area."""

    return PredictedComponent(
        index=detection.index, label=detection.index,
        x1=round(detection.x1), y1=round(detection.y1),
        x2=round(detection.x2), y2=round(detection.y2),
        centroid_x=detection.centroid_x, centroid_y=detection.centroid_y,
        area_pixels=0, max_score=detection.confidence,
    )


def _save_image(path: Path, image: np.ndarray) -> None:
    """Сохраняет RGB visualization с уникальным исходным расширением."""

    output = path if path.suffix.lower() == ".png" else path.with_name(f"{path.name}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Cannot write visualization: {output}")


def _draw_positive(
    image: np.ndarray,
    ground_truth: Sequence[GroundTruthRecord],
    matched_indices: set[int],
    detections: Sequence[YoloDetection],
) -> np.ndarray:
    """Рисует GT жёлтым, matched YOLO зелёным, unmatched красным."""

    output = to_display_image(image)
    for gt in ground_truth:
        cv2.rectangle(output, (round(gt.x1), round(gt.y1)),
                      (round(gt.x2), round(gt.y2)), (255, 255, 0), 1)
    for index, detection in enumerate(detections):
        color = (0, 255, 0) if index in matched_indices else (255, 0, 0)
        cv2.rectangle(output, (round(detection.x1), round(detection.y1)),
                      (round(detection.x2), round(detection.y2)), color, 1)
    return output


def _draw_negative(image: np.ndarray, detections: Sequence[YoloDetection]) -> np.ndarray:
    """Рисует все negative YOLO bbox красным без текста."""

    output = to_display_image(image)
    for detection in detections:
        cv2.rectangle(output, (round(detection.x1), round(detection.y1)),
                      (round(detection.x2), round(detection.y2)), (255, 0, 0), 1)
    return output


def _evaluate_positive(
    runner: YoloModelRunner,
    source: ImageDirectorySource,
    gt_index: Mapping[str, tuple[GroundTruthRecord, ...]],
    visualizations_root: Path,
) -> tuple[list[PositiveImageRow], list[PositiveObjectRow], float]:
    """Оценивает positive YOLO detections общим centroid-to-GT-bbox matcher."""

    image_rows: list[PositiveImageRow] = []
    object_rows: list[PositiveObjectRow] = []
    total_inference_ms = 0.0
    for frame in source:
        result = runner.predict(frame.image)
        detections = result.detections
        components = tuple(_to_matching_component(item) for item in detections)
        gt_objects = gt_index[frame.source_name]
        matches, unmatched = match_predictions_to_ground_truth(gt_objects, components)
        matched = [item for item in matches if item.prediction is not None]
        matched_component_indices = {item.prediction.index for item in matched}
        detection_positions = {
            detection.index: index for index, detection in enumerate(detections)
        }
        size_gt = Counter(item.size_class for item in gt_objects)
        size_tp = Counter(item.ground_truth.size_class for item in matched)
        image_rows.append(PositiveImageRow(
            image=frame.source_name, gt_count=len(gt_objects),
            predicted_count=len(detections), tp=len(matched),
            fn=len(gt_objects) - len(matched), fp=len(unmatched),
            tiny_gt=size_gt["Tiny"], tiny_tp=size_tp["Tiny"],
            small_gt=size_gt["Small"], small_tp=size_tp["Small"],
            medium_gt=size_gt["Medium"], medium_tp=size_tp["Medium"],
            large_gt=size_gt["Large"], large_tp=size_tp["Large"],
            inference_ms=result.inference_ms,
        ))
        for match in matches:
            component = match.prediction
            detection = (
                detections[detection_positions[component.index]] if component else None
            )
            object_rows.append(PositiveObjectRow(
                image=frame.source_name,
                annotation_id=match.ground_truth.annotation_id,
                size_class=match.ground_truth.size_class,
                gt_x1=match.ground_truth.x1, gt_y1=match.ground_truth.y1,
                gt_x2=match.ground_truth.x2, gt_y2=match.ground_truth.y2,
                matched=detection is not None,
                prediction_index=detection.index if detection else None,
                prediction_centroid_x=detection.centroid_x if detection else None,
                prediction_centroid_y=detection.centroid_y if detection else None,
                match_distance_px=match.distance_px, prediction_area=None,
                prediction_score=detection.confidence if detection else None,
            ))
        matched_positions = {
            detection_positions[index] for index in matched_component_indices
        }
        _save_image(
            visualizations_root / "positive" / frame.source_name,
            _draw_positive(frame.image, gt_objects, matched_positions, detections),
        )
        total_inference_ms += result.inference_ms
    return image_rows, object_rows, total_inference_ms


def _evaluate_negative(
    runner: YoloModelRunner,
    source: ImageDirectorySource,
    source_set: str,
    errors_root: Path,
) -> tuple[list[YoloNegativeImageRow], YoloNegativeStats]:
    """Оценивает negative-набор, где каждый YOLO bbox является FP."""

    rows: list[YoloNegativeImageRow] = []
    stats = YoloNegativeStats()
    for frame in source:
        result = runner.predict(frame.image)
        detections = result.detections
        rows.append(YoloNegativeImageRow(
            frame.source_name, source_set, len(detections), None, result.inference_ms
        ))
        stats.frames += 1
        stats.frames_with_fp += int(bool(detections))
        stats.fp_components += len(detections)
        stats.inference_ms += result.inference_ms
        if detections:
            _save_image(errors_root / source_set / frame.source_name,
                        _draw_negative(frame.image, detections))
    return rows, stats


def _build_summary(
    args: argparse.Namespace,
    positives: Sequence[PositiveImageRow],
    positive_inference_ms: float,
    negatives: Mapping[str, YoloNegativeStats],
) -> dict[str, object]:
    """Формирует YOLO summary с пустыми segmentation pixel metrics."""

    tp = sum(row.tp for row in positives)
    fn = sum(row.fn for row in positives)
    fp = sum(row.fp for row in positives)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    negative_frames = sum(item.frames for item in negatives.values())
    negative_frames_with_fp = sum(item.frames_with_fp for item in negatives.values())
    negative_components = sum(item.fp_components for item in negatives.values())
    negative_ms = sum(item.inference_ms for item in negatives.values())
    total_frames = len(positives) + negative_frames
    overall_ms = _safe_ratio(positive_inference_ms + negative_ms, total_frames)
    summary: dict[str, object] = {
        "runner_type": "yolo", "model": args.model,
        "train_dataset": args.train_dataset, "checkpoint": args.checkpoint.name,
        "threshold": "", "confidence_threshold": args.confidence,
        "nms_iou": args.iou, "imgsz": args.imgsz,
        "class_id": "" if args.class_id is None else args.class_id,
        "positive_frames": len(positives), "gt_objects": tp + fn,
        "tp": tp, "fn": fn, "fp": fp, "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
        "positive_frames_with_all_targets_detected": sum(row.fn == 0 for row in positives),
        "positive_frames_with_missed_targets": sum(row.fn > 0 for row in positives),
        "positive_frames_with_false_positives": sum(row.fp > 0 for row in positives),
        "total_predicted_components_positive": sum(row.predicted_count for row in positives),
    }
    for prefix in ("tiny", "small", "medium", "large"):
        gt_count = sum(getattr(row, f"{prefix}_gt") for row in positives)
        size_tp = sum(getattr(row, f"{prefix}_tp") for row in positives)
        summary.update({f"{prefix}_gt": gt_count, f"{prefix}_tp": size_tp,
                        f"{prefix}_fn": gt_count - size_tp,
                        f"{prefix}_recall": _safe_ratio(size_tp, gt_count)})
    summary.update({
        "negative_frames": negative_frames,
        "negative_frames_with_fp": negative_frames_with_fp,
        "negative_frame_fp_rate": _safe_ratio(negative_frames_with_fp, negative_frames),
        "negative_fp_components": negative_components,
        "negative_fp_components_per_frame": _safe_ratio(negative_components, negative_frames),
        "negative_fp_pixels": "", "false_alarm_pixel_rate": "",
    })
    for source_set in ("clear_horizon", "clear_sky"):
        stats = negatives[source_set]
        summary.update({
            f"{source_set}_frames": stats.frames,
            f"{source_set}_frames_with_fp": stats.frames_with_fp,
            f"{source_set}_frame_fp_rate": _safe_ratio(stats.frames_with_fp, stats.frames),
            f"{source_set}_fp_components": stats.fp_components,
            f"{source_set}_fp_components_per_frame": _safe_ratio(stats.fp_components, stats.frames),
            f"{source_set}_fp_pixels": "",
        })
    summary.update({
        "positive_avg_inference_ms": _safe_ratio(positive_inference_ms, len(positives)),
        "negative_avg_inference_ms": _safe_ratio(negative_ms, negative_frames),
        "overall_avg_inference_ms": overall_ms, "avg_inference_ms": overall_ms,
        "fps_inference": _safe_ratio(1000.0, overall_ms),
    })
    return summary


def _update_comparison(path: Path, summary: Mapping[str, object]) -> None:
    """Обновляет YOLO run по backend/model/dataset/checkpoint/confidence."""

    key_fields = (
        "runner_type", "model", "train_dataset", "checkpoint",
        "confidence_threshold",
    )
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
    """Создаёт CLI объективного YOLO benchmark."""

    parser = argparse.ArgumentParser(description="Evaluate one Ultralytics YOLO model")
    parser.add_argument("--positive", required=True, type=Path)
    parser.add_argument("--clear-horizon", required=True, type=Path)
    parser.add_argument("--clear-sky", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--confidence", default=0.25, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--class-id", type=int)
    parser.add_argument("--device", default="cuda")
    return parser


def run(args: argparse.Namespace) -> Path:
    """Валидирует индексы, выполняет warmup и три потоковых прогона."""

    _load_runtime_dependencies()
    positive_source = ImageDirectorySource(args.positive)
    negative_sources = {
        "clear_horizon": ImageDirectorySource(args.clear_horizon),
        "clear_sky": ImageDirectorySource(args.clear_sky),
    }
    gt_index = load_ground_truth_objects(args.ground_truth)
    expected_positive = _load_expected_images(args.ground_truth)
    if _directory_names(args.positive) != expected_positive or set(gt_index) != expected_positive:
        raise ValueError("Positive dataset does not match ground truth CSV indexes")
    negative_index = _load_negative_index(args.ground_truth)
    roots = {"clear_horizon": args.clear_horizon, "clear_sky": args.clear_sky}
    for source_set, root in roots.items():
        if _directory_names(root) != negative_index.get(source_set, set()):
            raise ValueError(f"Negative dataset does not match CSV index: {source_set}")

    run_root = args.output / f"{args.model}_{args.train_dataset}"
    errors_root = run_root / "errors"
    positive_visualizations = run_root / "visualizations" / "positive"
    for generated in (errors_root, positive_visualizations):
        if generated.exists():
            shutil.rmtree(generated)
        generated.mkdir(parents=True)

    runner = YoloModelRunner(YoloModelConfig(
        checkpoint_path=args.checkpoint, device=args.device,
        confidence_threshold=args.confidence, nms_iou=args.iou,
        imgsz=args.imgsz, class_id=args.class_id,
    ))
    first_frame: FrameData = next(iter(positive_source))
    runner.predict(first_frame.image)
    positive_rows, object_rows, positive_ms = _evaluate_positive(
        runner, positive_source, gt_index, run_root / "visualizations"
    )
    negative_rows: list[YoloNegativeImageRow] = []
    negative_stats: dict[str, YoloNegativeStats] = {}
    for source_set, source in negative_sources.items():
        rows, stats = _evaluate_negative(runner, source, source_set, errors_root)
        negative_rows.extend(rows)
        negative_stats[source_set] = stats
    summary = _build_summary(args, positive_rows, positive_ms, negative_stats)
    _write_rows(run_root / "positive_images.csv", PositiveImageRow, positive_rows)
    _write_rows(run_root / "positive_objects.csv", PositiveObjectRow, object_rows)
    _write_rows(run_root / "negative_images.csv", YoloNegativeImageRow, negative_rows)
    _write_dict_rows(run_root / "summary.csv", [summary])
    args.output.mkdir(parents=True, exist_ok=True)
    _update_comparison(args.output / "model_comparison.csv", summary)
    return run_root


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Печатает ожидаемые ошибки CLI и возвращает exit code."""

    try:
        output = run(build_argument_parser().parse_args(argv))
        print(f"YOLO evaluation completed: {output}")
    except (FileNotFoundError, ImportError, NotADirectoryError, OSError,
            RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
