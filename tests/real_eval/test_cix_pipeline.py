"""Проверяет deterministic split и метрики ALCNet/CIX pipeline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from real_eval.cix_dataset import (
    DatasetFrame, select_calibration_frames, validation_frames,
)
from real_eval.compare_cix_outputs import FrameComparison, _summary, compare_frame
from real_eval.evaluation_core import (
    GroundTruthFrame, GroundTruthRecord, GroundTruthRole, SIZE_CLASSES,
)
from real_eval.model_runner import ModelRunner


def _candidate(
    index: int, size_class: str = "Tiny", reason: str = "size_class_coverage"
) -> DatasetFrame:
    """Создаёт positive-кандидат для тестовой выборки."""

    return DatasetFrame(
        f"positive/{index:03d}.png", "positive", "positive", size_class,
        reason,
    )


def test_calibration_is_deterministic_and_validation_is_disjoint() -> None:
    """Calibration имеет 128 кадров, покрывает strata и не входит в validation."""

    positives = [
        _candidate(index, SIZE_CLASSES[index % 4], "missed_target")
        for index in range(32)
    ] + [
        _candidate(
            index, SIZE_CLASSES[index % 4],
            "near_threshold" if index % 2 else "low_detection_score",
        )
        for index in range(32, 64)
    ] + [
        _candidate(index, SIZE_CLASSES[index % 4]) for index in range(64, 160)
    ]
    negatives = [
        DatasetFrame(f"sky/{index:03d}.png", "negative", "clear_sky", "", "clear_sky")
        for index in range(40)
    ] + [
        DatasetFrame(
            f"horizon/{index:03d}.png", "negative", "clear_horizon", "", "clear_horizon"
        )
        for index in range(40)
    ]
    frames = tuple(positives + negatives)
    first = select_calibration_frames(frames, count=128, seed=7)
    second = select_calibration_frames(frames, count=128, seed=7)
    validation = validation_frames(frames, first)

    assert first == second
    assert len(first) == 128
    selected_positive = [frame for frame in first if frame.frame_type == "positive"]
    selected_negative = [frame for frame in first if frame.frame_type == "negative"]
    assert len(selected_positive) == 64
    assert len(selected_negative) == 64
    assert sum(frame.source_set == "clear_sky" for frame in first) == 32
    assert sum(frame.source_set == "clear_horizon" for frame in first) == 32
    for size_class in SIZE_CLASSES:
        assert any(size_class in frame.size_class for frame in first)
    assert {frame.source_file for frame in first}.isdisjoint(
        frame.source_file for frame in validation
    )
    assert len(validation) == len(frames) - 128


def test_calibration_falls_back_to_any_remaining_positive() -> None:
    """Selector набирает 64 positive, даже если обычных кадров недостаточно."""

    positives = [
        _candidate(
            index,
            SIZE_CLASSES[index % len(SIZE_CLASSES)],
            "size_class_coverage" if index < 8 else "missed_target",
        )
        for index in range(80)
    ]
    negatives = [
        DatasetFrame(f"sky/{index}.png", "negative", "clear_sky", "", "clear_sky")
        for index in range(32)
    ] + [
        DatasetFrame(
            f"horizon/{index}.png", "negative", "clear_horizon", "", "clear_horizon"
        )
        for index in range(32)
    ]

    selected = select_calibration_frames(tuple(positives + negatives), count=128, seed=3)
    selected_positive = [frame for frame in selected if frame.frame_type == "positive"]

    assert len(selected_positive) == 64
    assert len({frame.source_file for frame in selected_positive}) == 64
    assert sum(frame.selection_reason == "missed_target" for frame in selected_positive) > 16


def test_compare_frame_reuses_object_matching_and_strict_threshold() -> None:
    """Comparator считает threshold flips и штатный TP/FN для одного объекта."""

    frame = DatasetFrame("positive/frame.png", "positive", "positive", "Tiny", "test")
    record = GroundTruthRecord(
        image=frame.source_file, annotation_id=1, size_class="Tiny",
        x1=9, y1=9, x2=12, y2=12, role=GroundTruthRole.SCORED,
    )
    ground_truth = GroundTruthFrame(frame.source_file, (record,), ())
    pytorch = np.zeros((1, 640, 512), dtype=np.float32)
    cix = np.zeros_like(pytorch)
    pytorch[0, 10, 10] = 0.6
    cix[0, 10, 10] = 0.5

    row, pytorch_size_tp, cix_size_tp = compare_frame(
        frame, pytorch, cix, ground_truth
    )

    assert row.threshold_flips == 1
    assert row.pytorch_tp == 1
    assert row.cix_fn == 1
    assert pytorch_size_tp["Tiny"] == 1
    assert cix_size_tp["Tiny"] == 0


def test_negative_fp_does_not_reduce_object_precision() -> None:
    """Negative FP учитывается отдельно и не входит в object precision."""

    common = dict(
        size_class="", mae=0.0, max_abs_error=0.0, mask_iou=1.0,
        threshold_flips=0, pytorch_fn=0, cix_fn=0,
    )
    positive = FrameComparison(
        source_file="positive.png", frame_type="positive", source_set="positive",
        pytorch_foreground_pixels=1, cix_foreground_pixels=1,
        pytorch_components=1, cix_components=1,
        pytorch_tp=1, pytorch_fp=0, cix_tp=1, cix_fp=0, **common,
    )
    negative = FrameComparison(
        source_file="sky.png", frame_type="negative", source_set="clear_sky",
        pytorch_foreground_pixels=3, cix_foreground_pixels=4,
        pytorch_components=2, cix_components=1,
        pytorch_tp=0, pytorch_fp=2, cix_tp=0, cix_fp=1, **common,
    )

    summary = _summary((positive, negative), Counter(), Counter(), Counter())

    assert summary["pytorch_precision"] == 1.0
    assert summary["cix_precision"] == 1.0
    assert summary["pytorch_fp"] == 0
    assert summary["cix_fp"] == 0
    assert summary["pytorch_negative_fp_components"] == 2
    assert summary["cix_negative_fp_components"] == 1
    assert summary["pytorch_clear_sky_frames_with_fp"] == 1
    assert summary["cix_clear_sky_fp_pixels"] == 4


def test_run_preprocessed_returns_full_model_tensor() -> None:
    """Новый метод выполняет только модель и не обрезает padded output."""

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cpu")
    runner._model = torch.nn.Identity()
    tensor = torch.zeros((1, 1, 640, 512), dtype=torch.float32)

    output = runner.run_preprocessed(tensor)

    assert output is tensor
    assert tuple(output.shape) == (1, 1, 640, 512)


def test_run_preprocessed_moves_input_to_model_device() -> None:
    """Метод сам переносит готовый CPU tensor на устройство модели."""

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("meta")
    runner._model = torch.nn.Identity()
    tensor = torch.zeros((1, 1, 640, 512), dtype=torch.float32)

    output = runner.run_preprocessed(tensor)

    assert output.device == runner.device
    assert tuple(output.shape) == (1, 1, 640, 512)
