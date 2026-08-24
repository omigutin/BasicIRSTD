"""Готовит representative calibration tensor для CixBuilder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np


# Скрипт запускается напрямую из iwt_tools/scripts/conversion,
# а ModelRunner использует upstream net.py, utils.py и model/
# из корня BasicIRSTD.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from iwt_tools.cix.cix_dataset import (
    index_source_images,
    load_dataset_frames,
    resolve_source,
    select_calibration_frames,
    write_manifest,
)
from iwt_tools.evaluation.config import ModelConfig
from iwt_tools.evaluation.model_runner import ModelRunner
from iwt_tools.evaluation.sources import ImageSource


INPUT_SHAPE = (1, 1, 640, 512)


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI подготовки calibration dataset."""

    parser = argparse.ArgumentParser(
        description="Prepare BasicIRSTD CIX INT8 calibration"
    )

    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)

    parser.add_argument(
        "--dataset-dir",
        default=Path("datasets"),
        type=Path,
    )
    parser.add_argument("--count", default=128, type=int)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--device", default="cpu")

    return parser


def run(args: argparse.Namespace) -> Path:
    """Выбирает кадры и сохраняет float32 tensor и manifest."""

    frames = load_dataset_frames(
        args.ground_truth,
        args.results_root,
        threshold=0.5,
    )

    selected = select_calibration_frames(
        frames,
        args.count,
        args.seed,
    )

    image_index = index_source_images(args.dataset_root)

    args.output.mkdir(parents=True, exist_ok=True)

    output_path = args.output / "calibration.npy"

    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected), *INPUT_SHAPE[1:]),
    )

    runner = ModelRunner(
        ModelConfig(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint,
            train_dataset_name="IRSTD-1K",
            dataset_dir=args.dataset_dir,
            threshold=0.5,
            device=args.device,
        )
    )

    for index, frame in enumerate(selected):
        source_path = resolve_source(
            image_index,
            frame.source_file,
        )

        image = next(
            iter(ImageSource(source_path))
        ).image

        tensor, _ = runner.preprocess(image)

        array = tensor.detach().cpu().numpy()

        if array.shape != INPUT_SHAPE:
            raise ValueError(
                f"Expected preprocessed shape {INPUT_SHAPE}, "
                f"got {array.shape} for {source_path}"
            )

        output[index] = array[0]

    output.flush()

    write_manifest(
        args.output / "calibration_manifest.csv",
        selected,
    )

    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Запускает CLI и сообщает ожидаемые ошибки пользователю."""

    try:
        args = build_argument_parser().parse_args(argv)

        print(f"Model       : {args.model_name}")
        print(f"Checkpoint  : {args.checkpoint.resolve()}")
        print(f"Dataset     : {args.dataset_root.resolve()}")
        print(f"Results     : {args.results_root.resolve()}")
        print(f"Ground truth: {args.ground_truth.resolve()}")
        print(f"Output      : {args.output.resolve()}")
        print(f"Count       : {args.count}")
        print(f"Seed        : {args.seed}")

        path = run(args)

        print(f"\nCalibration saved: {path.resolve()}")

    except (
        FileNotFoundError,
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
