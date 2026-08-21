"""Проверяет deterministic split и метрики ALCNet/CIX pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from real_eval.cix_dataset import (
    DatasetFrame, select_calibration_frames, validation_frames,
)
from real_eval.compare_cix_outputs import compare_frame
from real_eval.evaluation_core import (
    GroundTruthFrame, GroundTruthRecord, GroundTruthRole, SIZE_CLASSES,
)
from real_eval.model_runner import ModelRunner


def _candidate(index: int, size_class: str = "Tiny") -> DatasetFrame:
    """Создаёт positive-кандидат для тестовой выборки."""

    return DatasetFrame(
        f"positive/{index:03d}.png", "positive", "positive", size_class,
        "missed_target" if index % 11 == 0 else "size_class_coverage",
    )


def test_calibration_is_deterministic_and_validation_is_disjoint() -> None:
    """Calibration имеет 128 кадров, покрывает strata и не входит в validation."""

    positives = [_candidate(index, SIZE_CLASSES[index % 4]) for index in range(140)]
    negatives = [
        DatasetFrame(f"sky/{index:03d}.png", "negative", "clear_sky", "", "clear_sky")
        for index in range(24)
    ] + [
        DatasetFrame(
            f"horizon/{index:03d}.png", "negative", "clear_horizon", "", "clear_horizon"
        )
        for index in range(24)
    ]
    frames = tuple(positives + negatives)
    first = select_calibration_frames(frames, count=128, seed=7)
    second = select_calibration_frames(frames, count=128, seed=7)
    validation = validation_frames(frames, first)

    assert first == second
    assert len(first) == 128
    assert {frame.source_set for frame in first} >= {"clear_sky", "clear_horizon"}
    for size_class in SIZE_CLASSES:
        assert any(size_class in frame.size_class for frame in first)
    assert {frame.source_file for frame in first}.isdisjoint(
        frame.source_file for frame in validation
    )
    assert len(validation) == len(frames) - 128


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
