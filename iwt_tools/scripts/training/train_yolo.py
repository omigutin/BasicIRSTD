"""Запускает обучение любой Detect-модели Ultralytics YOLO.

CPU-обучение запрещено по умолчанию, чтобы случайно не запустить долгую задачу.
Для осознанного запуска без CUDA необходимо передать ``--allow-cpu``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional


@dataclass(frozen=True)
class RuntimeInfo:
    """Описывает версии библиотек и доступное устройство обучения."""

    ultralytics_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    gpu_vram_gib: Optional[float]


@dataclass(frozen=True)
class TrainingSummary:
    """Хранит основные результаты завершённого обучения."""

    epochs_completed: int
    best_epoch: Optional[int]
    precision: Optional[float]
    recall: Optional[float]
    map50: Optional[float]
    map50_95: Optional[float]


def load_training_modules() -> tuple[ModuleType, ModuleType]:
    """Загружает установленные Ultralytics и PyTorch перед обучением."""
    ultralytics = importlib.import_module("ultralytics")
    torch = importlib.import_module("torch")
    return ultralytics, torch


def inspect_runtime(ultralytics: ModuleType, torch: ModuleType) -> RuntimeInfo:
    """Собирает сведения о версиях, CUDA и первой видеокарте."""
    cuda_available = bool(torch.cuda.is_available())
    gpu_name: Optional[str] = None
    gpu_vram_gib: Optional[float] = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu_name = str(properties.name)
        gpu_vram_gib = float(properties.total_memory) / (1024**3)
    return RuntimeInfo(
        ultralytics_version=str(ultralytics.__version__),
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
        cuda_version=str(torch.version.cuda) if torch.version.cuda else None,
        gpu_name=gpu_name,
        gpu_vram_gib=gpu_vram_gib,
    )


def print_runtime(info: RuntimeInfo) -> None:
    """Печатает сведения о программной и аппаратной среде."""
    print(f"Ultralytics version: {info.ultralytics_version}")
    print(f"PyTorch version: {info.torch_version}")
    print(f"CUDA available: {info.cuda_available}")
    print(f"CUDA version: {info.cuda_version or 'not available'}")
    print(f"GPU name: {info.gpu_name or 'not available'}")
    vram = f"{info.gpu_vram_gib:.2f} GiB" if info.gpu_vram_gib is not None else "not available"
    print(f"GPU VRAM: {vram}")


def read_best_metrics(results_csv: Path) -> TrainingSummary:
    """Читает лучшую эпоху и Detect-метрики из results.csv."""
    if not results_csv.is_file():
        raise FileNotFoundError(f"Training results not found: {results_csv}")
    with results_csv.open(newline="", encoding="utf-8-sig") as file:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(file)]
    if not rows:
        raise ValueError(f"Training results are empty: {results_csv}")

    map_key = "metrics/mAP50-95(B)"
    best_index = max(range(len(rows)), key=lambda index: float(rows[index].get(map_key, "-inf")))
    best = rows[best_index]

    def optional_float(key: str) -> Optional[float]:
        """Возвращает числовую метрику, если колонка существует."""
        value = best.get(key)
        return float(value) if value not in (None, "") else None

    epoch_value = best.get("epoch")
    best_epoch = int(float(epoch_value)) + 1 if epoch_value not in (None, "") else best_index + 1
    return TrainingSummary(
        epochs_completed=len(rows),
        best_epoch=best_epoch,
        precision=optional_float("metrics/precision(B)"),
        recall=optional_float("metrics/recall(B)"),
        map50=optional_float("metrics/mAP50(B)"),
        map50_95=optional_float(map_key),
    )


def count_dataset_images(dataset_yaml: Path) -> tuple[int, int]:
    """Считает PNG-изображения рядом с переносимым dataset.yaml."""
    root = dataset_yaml.parent
    return (
        len(list((root / "images" / "train").glob("*.png"))),
        len(list((root / "images" / "val").glob("*.png"))),
    )


def format_metric(value: Optional[float]) -> str:
    """Форматирует доступную метрику либо понятную заглушку."""
    return f"{value:.6f}" if value is not None else "not available"


def print_summary(
    model: str,
    dataset: Path,
    train_images: int,
    val_images: int,
    epochs_requested: int,
    summary: TrainingSummary,
    best_checkpoint: Path,
    training_dir: Path,
) -> None:
    """Печатает краткий итог завершённого обучения."""
    print("\nTraining summary")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Train images: {train_images}")
    print(f"Val images: {val_images}")
    print(f"Epochs requested: {epochs_requested}")
    print(f"Epochs completed: {summary.epochs_completed}")
    print(f"Best epoch: {summary.best_epoch or 'not available'}")
    print(f"Precision: {format_metric(summary.precision)}")
    print(f"Recall: {format_metric(summary.recall)}")
    print(f"mAP50: {format_metric(summary.map50)}")
    print(f"mAP50-95: {format_metric(summary.map50_95)}")
    print(f"best.pt: {best_checkpoint}")
    print(f"Training artifacts: {training_dir}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт небольшой набор практических параметров обучения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint или имя модели Ultralytics.")
    parser.add_argument("--dataset", required=True, type=Path, help="Путь к dataset.yaml.")
    parser.add_argument("--output", required=True, type=Path, help="Каталог модели и артефактов.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="auto", help="Размер batch либо auto.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def parse_batch(value: str) -> int:
    """Преобразует batch в целое число либо значение auto Ultralytics."""
    if value == "auto":
        return -1
    batch = int(value)
    if batch <= 0:
        raise ValueError("--batch must be 'auto' or a positive integer")
    return batch


def main() -> int:
    """Проверяет среду, запускает обучение и сохраняет лучший checkpoint."""
    args = build_argument_parser().parse_args()
    if args.epochs <= 0 or args.imgsz <= 0 or args.patience < 0 or args.workers < 0:
        raise ValueError("epochs/imgsz must be positive; patience/workers must be non-negative")
    dataset = args.dataset.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset config not found: {dataset}")
    output = args.output.resolve()
    training_dir = output / "training"
    if training_dir.exists() or (output / "best.pt").exists():
        raise FileExistsError(f"Training output already exists: {output}")

    ultralytics, torch = load_training_modules()
    runtime = inspect_runtime(ultralytics, torch)
    print_runtime(runtime)
    if not runtime.cuda_available and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. CPU training was not started. "
            "Pass --allow-cpu only if slow CPU training is intentional."
        )
    device = 0 if runtime.cuda_available else "cpu"
    train_images, val_images = count_dataset_images(dataset)
    if train_images == 0 or val_images == 0:
        raise ValueError("Dataset train/val image directories are empty or unsupported")

    model = ultralytics.YOLO(args.model)
    model.train(
        data=str(dataset),
        project=str(output),
        name="training",
        exist_ok=False,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=parse_batch(args.batch),
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        mosaic=args.mosaic,
        mixup=0.0,
        copy_paste=0.0,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        device=device,
    )
    source_best = training_dir / "weights" / "best.pt"
    if not source_best.is_file():
        raise FileNotFoundError(f"Best checkpoint not produced: {source_best}")
    destination_best = output / "best.pt"
    shutil.copy2(source_best, destination_best)
    summary = read_best_metrics(training_dir / "results.csv")
    print_summary(
        args.model,
        dataset,
        train_images,
        val_images,
        args.epochs,
        summary,
        destination_best,
        training_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
