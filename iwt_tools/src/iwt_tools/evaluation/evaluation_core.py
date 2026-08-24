"""Общие структуры ground truth и object-level matching для evaluator-ов."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from math import hypot
from pathlib import Path
from typing import Mapping, Optional, Sequence

MATCH_DISTANCE_PX = 3.0
SIZE_CLASSES = ("Tiny", "Small", "Medium", "Large")


class GroundTruthRole(str, Enum):
    """Определяет, участвует ли объект ground truth в scoring."""

    SCORED = "scored"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class GroundTruthRecord:
    """Один bbox ground truth с явной ролью в benchmark."""

    image: str
    annotation_id: int
    size_class: Optional[str]
    x1: float
    y1: float
    x2: float
    y2: float
    category_name: str = "bpla"
    role: GroundTruthRole = GroundTruthRole.SCORED


@dataclass(frozen=True)
class GroundTruthFrame:
    """Разделяет scored и uncertain объекты одного source-кадра."""

    image: str
    scored: tuple[GroundTruthRecord, ...]
    uncertain: tuple[GroundTruthRecord, ...]

    @property
    def is_scored_positive(self) -> bool:
        """Показывает наличие хотя бы одного scored BPLA."""

        return bool(self.scored)

    @property
    def is_ignore_only(self) -> bool:
        """Показывает кадр только с uncertain GT, который не является negative."""

        return not self.scored and bool(self.uncertain)


@dataclass(frozen=True)
class PredictedComponent:
    """Одна predicted-компонента, пригодная для общего matching."""

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
    """Результат primary one-to-one matching одного scored GT."""

    ground_truth: GroundTruthRecord
    prediction: Optional[PredictedComponent]
    distance_px: Optional[float]


@dataclass(frozen=True)
class IgnoredPrediction:
    """Prediction, поглощённый uncertain ground-truth областью."""

    prediction: PredictedComponent
    uncertain_ground_truth: GroundTruthRecord
    distance_px: float


@dataclass(frozen=True)
class FrameMatchingResult:
    """Полный результат scored matching и uncertain post-classification."""

    scored_matches: tuple[ObjectMatch, ...]
    ignored_uncertain: tuple[IgnoredPrediction, ...]
    false_positives: tuple[PredictedComponent, ...]

    @property
    def tp(self) -> int:
        """Возвращает число matched scored GT."""

        return sum(item.prediction is not None for item in self.scored_matches)

    @property
    def fn(self) -> int:
        """Возвращает число unmatched scored GT."""

        return sum(item.prediction is None for item in self.scored_matches)

    @property
    def fp(self) -> int:
        """Возвращает число predictions вне scored и uncertain GT."""

        return len(self.false_positives)


def distance_from_centroid_to_bbox(
    prediction: PredictedComponent,
    ground_truth: GroundTruthRecord,
) -> float:
    """Считает расстояние centroid до bbox, не оценивая mask IoU/Dice."""

    nearest_x = min(max(prediction.centroid_x, ground_truth.x1), ground_truth.x2)
    nearest_y = min(max(prediction.centroid_y, ground_truth.y1), ground_truth.y2)
    return hypot(prediction.centroid_x - nearest_x, prediction.centroid_y - nearest_y)


def match_predictions_to_ground_truth(
    ground_truth: Sequence[GroundTruthRecord],
    predictions: Sequence[PredictedComponent],
    max_distance_px: float = MATCH_DISTANCE_PX,
) -> tuple[tuple[ObjectMatch, ...], tuple[PredictedComponent, ...]]:
    """Выполняет прежний Hungarian matching scored GT без изменения критерия."""

    if max_distance_px < 0:
        raise ValueError("Match distance must not be negative")
    if not ground_truth:
        return (), tuple(predictions)
    if not predictions:
        return tuple(ObjectMatch(item, None, None) for item in ground_truth), ()
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    distances = np.asarray([
        [distance_from_centroid_to_bbox(prediction, gt) for prediction in predictions]
        for gt in ground_truth
    ], dtype=np.float64)
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


def classify_uncertain_predictions(
    predictions: Sequence[PredictedComponent],
    uncertain_ground_truth: Sequence[GroundTruthRecord],
    max_distance_px: float = MATCH_DISTANCE_PX,
) -> tuple[tuple[IgnoredPrediction, ...], tuple[PredictedComponent, ...]]:
    """Поглощает unmatched predictions uncertain bbox-областями без one-to-one."""

    ignored: list[IgnoredPrediction] = []
    false_positives: list[PredictedComponent] = []
    for prediction in predictions:
        candidates: list[tuple[float, int, GroundTruthRecord]] = []
        for ground_truth in uncertain_ground_truth:
            distance = distance_from_centroid_to_bbox(prediction, ground_truth)
            if distance <= max_distance_px:
                candidates.append((distance, ground_truth.annotation_id, ground_truth))
        if candidates:
            distance, _, ground_truth = min(candidates, key=lambda item: (item[0], item[1]))
            ignored.append(IgnoredPrediction(prediction, ground_truth, distance))
        else:
            false_positives.append(prediction)
    return tuple(ignored), tuple(false_positives)


def match_frame_predictions(
    ground_truth: GroundTruthFrame,
    predictions: Sequence[PredictedComponent],
    max_distance_px: float = MATCH_DISTANCE_PX,
) -> FrameMatchingResult:
    """Сначала матчится scored GT, затем uncertain поглощает остаток."""

    matches, unmatched = match_predictions_to_ground_truth(
        ground_truth.scored, predictions, max_distance_px
    )
    ignored, false_positives = classify_uncertain_predictions(
        unmatched, ground_truth.uncertain, max_distance_px
    )
    return FrameMatchingResult(matches, ignored, false_positives)


def load_ground_truth_index(root: Path) -> Mapping[str, GroundTruthFrame]:
    """Читает новую или legacy CSV-схему и сохраняет кадры без annotations."""

    images_path = root / "ground_truth_images.csv"
    objects_path = root / "ground_truth_objects.csv"
    with images_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if "image" not in (reader.fieldnames or ()):
            raise ValueError(f"Missing columns in {images_path}: ['image']")
        image_names = [Path(row["image"]).as_posix() for row in reader]
    if len(image_names) != len(set(image_names)):
        raise ValueError("Duplicate images in ground_truth_images.csv")

    scored: dict[str, list[GroundTruthRecord]] = defaultdict(list)
    uncertain: dict[str, list[GroundTruthRecord]] = defaultdict(list)
    seen_ids: set[int] = set()
    required = {"image", "annotation_id", "size_class", "x1", "y1", "x2", "y2"}
    with objects_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {objects_path}: {sorted(missing)}")
        has_roles = "gt_role" in (reader.fieldnames or ())
        for row in reader:
            annotation_id = int(row["annotation_id"])
            if annotation_id in seen_ids:
                raise ValueError(f"Duplicate annotation id: {annotation_id}")
            seen_ids.add(annotation_id)
            role = GroundTruthRole(row["gt_role"]) if has_roles else GroundTruthRole.SCORED
            size_class = row["size_class"] or None
            if role is GroundTruthRole.SCORED and size_class not in SIZE_CLASSES:
                raise ValueError(f"Unsupported size class: {size_class}")
            if role is GroundTruthRole.UNCERTAIN and size_class is not None:
                raise ValueError("Uncertain ground truth must not have a size class")
            record = GroundTruthRecord(
                image=Path(row["image"]).as_posix(), annotation_id=annotation_id,
                size_class=size_class, x1=float(row["x1"]), y1=float(row["y1"]),
                x2=float(row["x2"]), y2=float(row["y2"]),
                category_name=row.get("category_name", "bpla"), role=role,
            )
            if record.x2 <= record.x1 or record.y2 <= record.y1:
                raise ValueError(f"Invalid GT bbox for annotation {annotation_id}")
            (scored if role is GroundTruthRole.SCORED else uncertain)[record.image].append(record)
    known = set(image_names)
    unknown = (set(scored) | set(uncertain)) - known
    if unknown:
        raise ValueError(f"Ground truth objects reference unknown images: {sorted(unknown)}")
    return {
        name: GroundTruthFrame(name, tuple(scored[name]), tuple(uncertain[name]))
        for name in image_names
    }


def load_ground_truth_objects(root: Path) -> Mapping[str, tuple[GroundTruthRecord, ...]]:
    """Возвращает legacy-индекс только scored объектов для совместимости."""

    return {name: frame.scored for name, frame in load_ground_truth_index(root).items()}


def has_all_scored_targets_detected(gt_count: int, fn: int) -> bool:
    """Исключает кадры без scored GT из счётчика all-targets-detected."""

    return gt_count > 0 and fn == 0
