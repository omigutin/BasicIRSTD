"""Переиспользуемое сравнение выходов PyTorch и CIX."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .cix_dataset import DatasetFrame
from ..evaluation.evaluation_core import GroundTruthFrame, match_frame_predictions
from ..evaluation.evaluate_model import extract_predicted_components
from ..evaluation.model_runner import PredictionResult

EXPECTED_FRAME_SHAPE = (1, 640, 512)
THRESHOLD = 0.5

@dataclass(frozen=True)
class FrameComparison:
    """Хранит численную и объектную разницу одного validation кадра."""

    source_file: str
    frame_type: str
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


def ground_truth_for_frame(
    frame: DatasetFrame, positive_index: dict[str, GroundTruthFrame]
) -> GroundTruthFrame:
    """Возвращает штатный GT либо пустой scored GT для negative кадра."""

    if frame.frame_type == "positive":
        if frame.source_file not in positive_index:
            raise ValueError(f"Positive frame is absent from ground truth: {frame.source_file}")
        return positive_index[frame.source_file]
    return GroundTruthFrame(frame.source_file, (), ())


# Совместимость с кодом, который использовал прежнее приватное имя.
_ground_truth_for_frame = ground_truth_for_frame


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
        size_class=frame.size_class, mae=float(difference.mean()),
        max_abs_error=float(difference.max()), mask_iou=mask_iou,
        threshold_flips=int(np.count_nonzero(pytorch.binary_mask ^ cix.binary_mask)),
        pytorch_foreground_pixels=pytorch.foreground_pixels,
        cix_foreground_pixels=cix.foreground_pixels,
        pytorch_components=len(pytorch_components), cix_components=len(cix_components),
        pytorch_tp=pytorch_matching.tp, pytorch_fn=pytorch_matching.fn,
        pytorch_fp=pytorch_matching.fp, cix_tp=cix_matching.tp,
        cix_fn=cix_matching.fn, cix_fp=cix_matching.fp,
    ), pytorch_size_tp, cix_size_tp
