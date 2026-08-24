"""Проверяет статический Batch 4 exporter и Orange Pi batching helpers."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_script(relative_path: str, module_name: str) -> ModuleType:
    """Загружает CLI script как модуль без запуска main()."""

    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def orange_runner() -> ModuleType:
    """Загружает helpers Orange runner без noe_engine."""

    return _load_script(
        "iwt_tools/scripts/orangepi/run_cix_validation.py", "alcnet_orange_runner"
    )


def test_exporter_parser_accepts_batch_1_and_4() -> None:
    """Exporter принимает Batch 1 и Batch 4, сохраняя default Batch 1."""

    exporter = _load_script(
        "iwt_tools/scripts/conversion/export_alcnet_to_onnx.py", "alcnet_exporter"
    )
    parser: argparse.ArgumentParser = exporter.build_argument_parser()

    assert parser.parse_args([]).batch_size == 1
    assert parser.parse_args(["--batch-size", "1"]).batch_size == 1
    assert parser.parse_args(["--batch-size", "4"]).batch_size == 4


def test_exporter_stacks_four_identical_frames() -> None:
    """Один подготовленный кадр повторяется в статический Batch 4."""

    exporter = _load_script(
        "iwt_tools/scripts/conversion/export_alcnet_to_onnx.py", "alcnet_exporter_stack"
    )
    frame = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)

    batch = exporter.repeat_export_input(frame, 4)

    assert batch.shape == (4, 1, 3, 4)
    assert batch.flags.c_contiguous
    np.testing.assert_array_equal(batch, np.repeat(frame, 4, axis=0))


def test_flattened_batch_4_output_is_reshaped(orange_runner: ModuleType) -> None:
    """Flattened CIX output восстанавливается как NCHW Batch 4."""

    flattened = np.arange(4 * 12, dtype=np.float32)

    output = orange_runner.reshape_cix_output([flattened], 4, (1, 3, 4))

    assert output.shape == (4, 1, 3, 4)
    np.testing.assert_array_equal(output.ravel(), flattened)


def test_batch_outputs_are_stored_per_frame(orange_runner: ModuleType) -> None:
    """Batch outputs сохраняются в прежнем per-frame layout."""

    destination = np.zeros((4, 1, 2, 2), dtype=np.float32)
    batch_outputs = np.arange(16, dtype=np.float32).reshape(4, 1, 2, 2)

    orange_runner.store_real_outputs(destination, 0, batch_outputs, 4)

    np.testing.assert_array_equal(destination, batch_outputs)


def test_last_batch_is_padded_but_extra_output_is_not_stored(
    orange_runner: ModuleType,
) -> None:
    """Последний batch дополняется, но padding output не попадает в storage."""

    inputs = np.arange(5 * 4, dtype=np.float32).reshape(5, 1, 2, 2)
    batches = list(orange_runner.iter_padded_batches(inputs, 4))
    last_batch, real_count, padded = batches[-1]
    destination = np.zeros_like(inputs)

    orange_runner.store_real_outputs(destination, 4, last_batch + 100, real_count)

    assert last_batch.shape == (4, 1, 2, 2)
    assert real_count == 1
    assert padded is True
    np.testing.assert_array_equal(last_batch[0], last_batch[1])
    np.testing.assert_array_equal(destination[4], inputs[4] + 100)
    assert destination.shape == inputs.shape


def test_fps_uses_frames_per_batch(orange_runner: ModuleType) -> None:
    """Throughput учитывает четыре кадра в одном inference-вызове."""

    summary = orange_runner._timing_summary([8.0, 8.0], batch_size=4)

    assert summary.batch_mean_ms == pytest.approx(8.0)
    assert summary.ms_per_frame == pytest.approx(2.0)
    assert summary.fps == pytest.approx(500.0)
