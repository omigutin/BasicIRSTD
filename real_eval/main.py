"""Единая CLI-точка запуска real_eval для изображений и видео."""

import argparse
import csv
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Union

import cv2

from .config import ModelConfig, RunConfig
from .model_runner import ModelRunner, PredictionResult
from .sources import FrameData, FrameSource, create_frame_source
from .visualization import create_overlay, save_frame_artifacts


LOGGER = logging.getLogger("real_eval")
CsvValue = Union[str, int, float, bool, None]


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI с одной явно выбранной моделью на запуск."""

    parser = argparse.ArgumentParser(description="Run BasicIRSTD on real images or video")
    parser.add_argument("--input", required=True, type=Path, help="Image, recursive image directory, or video")
    parser.add_argument("--output", required=True, type=Path, help="Results root directory")
    parser.add_argument("--model", required=True, help="BasicIRSTD model name, for example RDIAN")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint containing state_dict")
    parser.add_argument("--train-dataset", required=True, help="Dataset used to train the checkpoint")
    parser.add_argument("--dataset-dir", default=Path("datasets"), type=Path, help="BasicIRSTD datasets root")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--device", default="cuda", help="PyTorch device, for example cuda, cuda:0, or cpu")
    return parser


def _row(frame: FrameData, result: PredictionResult, config: ModelConfig) -> Dict[str, CsvValue]:
    """Формирует общую CSV-строку для изображения или видеокадра."""

    return {
        "source": frame.source_name,
        "frame": frame.index,
        "timestamp_sec": frame.timestamp_sec,
        "model": config.model_name,
        "train_dataset": config.train_dataset_name,
        "threshold": config.threshold,
        "max_score": result.max_score,
        "target_found": result.target_found,
        "foreground_pixels": result.foreground_pixels,
        "component_count": result.component_count,
        "inference_ms": result.inference_ms,
    }


def _open_video_writer(path: Path, fps: Optional[float], frame: FrameData) -> cv2.VideoWriter:
    """Создаёт MP4 writer по геометрии первого обработанного кадра."""

    height, width = frame.image.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"Cannot create annotated video: {path}")
    return writer


def run(run_config: RunConfig, model_config: ModelConfig) -> Path:
    """Выполняет единый потоковый цикл и сохраняет CSV с результатами."""

    run_config.validate()
    source: FrameSource = create_frame_source(run_config.input_path)
    runner = ModelRunner(model_config)
    model_output = run_config.output_dir / model_config.model_name
    model_output.mkdir(parents=True, exist_ok=True)
    csv_path = model_output / "results.csv"
    fields = [
        "source", "frame", "timestamp_sec", "model", "train_dataset", "threshold",
        "max_score", "target_found", "foreground_pixels", "component_count", "inference_ms",
    ]
    writer: Optional[cv2.VideoWriter] = None
    processed = 0
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
            csv_writer.writeheader()
            for frame in source:
                result = runner.predict(frame.image)
                csv_writer.writerow(_row(frame, result, model_config))
                if source.is_video:
                    if writer is None:
                        video_path = model_output / f"annotated_{run_config.input_path.stem}.mp4"
                        writer = _open_video_writer(video_path, source.fps, frame)
                    overlay = create_overlay(
                        frame.image, result, model_config.model_name, model_config.threshold
                    )
                    writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                else:
                    save_frame_artifacts(
                        model_output,
                        Path(frame.source_name),
                        frame.image,
                        result,
                        model_config.model_name,
                        model_config.threshold,
                    )
                processed += 1
    finally:
        if writer is not None:
            writer.release()
    if processed == 0:
        raise RuntimeError("Input source produced no frames")
    LOGGER.info("Processed %d frame(s); results: %s", processed, model_output)
    return model_output


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Преобразует CLI-параметры в конфигурации и показывает ясные ошибки."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_argument_parser().parse_args(argv)
    try:
        run(
            RunConfig(input_path=args.input, output_dir=args.output),
            ModelConfig(
                model_name=args.model,
                checkpoint_path=args.checkpoint,
                train_dataset_name=args.train_dataset,
                dataset_dir=args.dataset_dir,
                threshold=args.threshold,
                device=args.device,
            ),
        )
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, TypeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
