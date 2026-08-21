"""Экспортирует disjoint validation inputs и эталонные outputs PyTorch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np

from .cix_dataset import (
    index_source_images, load_dataset_frames, read_manifest, resolve_source,
    validation_frames, write_manifest,
)
from .config import ModelConfig
from .model_runner import ModelRunner
from .sources import ImageSource

INPUT_SHAPE = (1, 1, 640, 512)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI validation export."""

    parser = argparse.ArgumentParser(description="Export ALCNet/CIX validation dataset")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-dir", default=Path("datasets"), type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def _copy_validation_ground_truth(
    ground_truth_root: Path, output: Path, validation_sources: set[str]
) -> None:
    """Сохраняет только относящиеся к validation строки штатного ground truth."""

    for filename in ("ground_truth_images.csv", "ground_truth_objects.csv", "negative_images.csv"):
        source = ground_truth_root / filename
        with source.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            if not fieldnames or "image" not in fieldnames:
                raise ValueError(f"Missing image column in {source}")
            rows = [row for row in reader if Path(row["image"]).as_posix() in validation_sources]
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    """Сохраняет все оставшиеся inputs, PyTorch maps, manifest и ground truth."""

    all_frames = load_dataset_frames(args.ground_truth, args.results_root, threshold=0.5)
    calibration = read_manifest(args.calibration_manifest)
    selected = validation_frames(all_frames, calibration)
    if not selected:
        raise ValueError("Validation set is empty after excluding calibration sources")
    image_index = index_source_images(args.dataset_root)
    args.output.mkdir(parents=True, exist_ok=True)
    shape = (len(selected), *INPUT_SHAPE[1:])
    inputs = np.lib.format.open_memmap(args.output / "input.npy", "w+", np.float32, shape=shape)
    outputs = np.lib.format.open_memmap(
        args.output / "pytorch_outputs.npy", "w+", np.float32, shape=shape
    )
    runner = ModelRunner(ModelConfig(
        "ALCNet", args.checkpoint, "IRSTD-1K", args.dataset_dir, 0.5, args.device
    ))
    for index, frame in enumerate(selected):
        image = next(iter(ImageSource(resolve_source(image_index, frame.source_file)))).image
        tensor, _ = runner.preprocess(image)
        if tuple(tensor.shape) != INPUT_SHAPE:
            raise ValueError(f"Expected preprocessed shape {INPUT_SHAPE}, got {tuple(tensor.shape)}")
        probability = runner.run_preprocessed(tensor)
        if tuple(probability.shape) != INPUT_SHAPE:
            raise ValueError(f"Expected output shape {INPUT_SHAPE}, got {tuple(probability.shape)}")
        inputs[index] = tensor.detach().cpu().numpy()[0]
        outputs[index] = probability.detach().cpu().numpy()[0]
    inputs.flush()
    outputs.flush()
    write_manifest(args.output / "validation_manifest.csv", selected)
    _copy_validation_ground_truth(
        args.ground_truth, args.output, {frame.source_file for frame in selected}
    )
    return args.output


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Запускает CLI и сообщает ожидаемые ошибки пользователю."""

    try:
        path = run(build_argument_parser().parse_args(argv))
        print(f"Validation saved: {path}")
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
