"""Проверяет deterministic split и метрики ALCNet/CIX pipeline."""

from __future__ import annotations

import numpy as np
import torch

from iwt_tools.cix.cix_dataset import (
    DatasetFrame, select_calibration_frames, validation_frames,
)
from iwt_tools.cix.output_comparison import compare_frame
from iwt_tools.evaluation.evaluation_core import (
    GroundTruthFrame, GroundTruthRecord, GroundTruthRole, SIZE_CLASSES,
)
from iwt_tools.evaluation.model_runner import ModelRunner


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
