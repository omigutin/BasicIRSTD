"""Общие операции с manifest для калибровки и проверки ALCNet/CIX."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from .evaluation_core import GroundTruthFrame, SIZE_CLASSES, load_ground_truth_index


@dataclass(frozen=True)
class DatasetFrame:
    """Описывает один исходный кадр и причину его включения в набор."""

    source_file: str
    frame_type: str
    source_set: str
    size_class: str
    selection_reason: str


@dataclass(frozen=True)
class PositiveResult:
    """Содержит доступные признаки сложности из прежнего запуска ALCNet."""

    missed: bool
    lowest_score: Optional[float]


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Читает CSV и возвращает строки с гарантированными строковыми значениями."""

    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _join_size_classes(frame: GroundTruthFrame) -> str:
    """Формирует стабильный список размерных классов объектов кадра."""

    present = {record.size_class for record in frame.scored}
    return "|".join(size_class for size_class in SIZE_CLASSES if size_class in present)


def load_positive_results(results_root: Path) -> Mapping[str, PositiveResult]:
    """Загружает признаки сложности из существующих CSV результата ALCNet."""

    image_path = results_root / "positive_images.csv"
    object_path = results_root / "positive_objects.csv"
    if not image_path.is_file() or not object_path.is_file():
        return {}

    image_rows = _read_csv(image_path)
    object_rows = _read_csv(object_path)
    missed_by_image: dict[str, bool] = {}
    for row in image_rows:
        name = Path(row["image"]).as_posix()
        missed_by_image[name] = int(row.get("fn", "0") or 0) > 0

    scores: dict[str, list[float]] = {}
    for row in object_rows:
        value = row.get("prediction_score", "").strip()
        if value:
            scores.setdefault(Path(row["image"]).as_posix(), []).append(float(value))
    return {
        name: PositiveResult(missed, min(scores[name]) if scores.get(name) else None)
        for name, missed in missed_by_image.items()
    }


def load_dataset_frames(
    ground_truth_root: Path,
    results_root: Optional[Path] = None,
    threshold: float = 0.5,
) -> tuple[DatasetFrame, ...]:
    """Собирает positive и negative кадры из штатных ground-truth CSV."""

    gt_index = load_ground_truth_index(ground_truth_root)
    results = load_positive_results(results_root) if results_root is not None else {}
    frames: list[DatasetFrame] = []
    for name, ground_truth in sorted(gt_index.items()):
        result = results.get(name)
        if result is not None and result.missed:
            reason = "missed_target"
        elif result is not None and result.lowest_score is not None:
            distance = abs(result.lowest_score - threshold)
            reason = "near_threshold" if distance <= 0.1 else "low_detection_score"
        else:
            reason = "size_class_coverage"
        frames.append(DatasetFrame(
            name, "positive", "positive", _join_size_classes(ground_truth), reason
        ))

    negative_path = ground_truth_root / "negative_images.csv"
    for row in _read_csv(negative_path):
        source_set = row["source_set"]
        if source_set not in {"clear_sky", "clear_horizon"}:
            raise ValueError(f"Unsupported negative source set: {source_set}")
        frames.append(DatasetFrame(
            Path(row["image"]).as_posix(), "negative", source_set, "", source_set
        ))
    names = [frame.source_file for frame in frames]
    if len(names) != len(set(names)):
        raise ValueError("Source file names must be unique across positive and negative frames")
    return tuple(frames)


def _stable_rank(frame: DatasetFrame, seed: int) -> str:
    """Возвращает воспроизводимый псевдослучайный ключ без зависимости от Python hash."""

    return hashlib.sha256(f"{seed}:{frame.source_file}".encode("utf-8")).hexdigest()


def select_calibration_frames(
    frames: Sequence[DatasetFrame], count: int = 128, seed: int = 2024
) -> tuple[DatasetFrame, ...]:
    """Выбирает representative calibration с покрытием размеров и backgrounds."""

    if count <= 0:
        raise ValueError("Calibration count must be positive")
    if len(frames) < count:
        raise ValueError(f"Need at least {count} source frames, found {len(frames)}")

    chosen: list[DatasetFrame] = []
    chosen_names: set[str] = set()

    def add(frame: DatasetFrame) -> None:
        if frame.source_file not in chosen_names and len(chosen) < count:
            chosen.append(frame)
            chosen_names.add(frame.source_file)

    positives = [frame for frame in frames if frame.frame_type == "positive"]
    for size_class in SIZE_CLASSES:
        candidates = [frame for frame in positives if size_class in frame.size_class.split("|")]
        if not candidates:
            raise ValueError(f"No positive frame represents size class: {size_class}")
        candidates.sort(key=lambda item: (
            {"missed_target": 0, "near_threshold": 1, "low_detection_score": 2}.get(
                item.selection_reason, 3
            ),
            _stable_rank(item, seed),
        ))
        add(candidates[0])

    # По 1/8 calibration на каждый штатный negative source при наличии данных.
    negative_quota = max(1, count // 8)
    for source_set in ("clear_sky", "clear_horizon"):
        candidates = sorted(
            (frame for frame in frames if frame.source_set == source_set),
            key=lambda item: _stable_rank(item, seed),
        )
        if not candidates:
            raise ValueError(f"No negative frames found for source set: {source_set}")
        for frame in candidates[:negative_quota]:
            add(frame)

    reason_priority = {
        "missed_target": 0,
        "near_threshold": 1,
        "low_detection_score": 2,
        "size_class_coverage": 3,
        "clear_sky": 4,
        "clear_horizon": 4,
    }
    remainder = sorted(
        frames,
        key=lambda item: (reason_priority[item.selection_reason], _stable_rank(item, seed)),
    )
    for frame in remainder:
        add(frame)
    if len(chosen) != count:
        raise ValueError(f"Could not select exactly {count} calibration frames")
    return tuple(chosen)


def validation_frames(
    frames: Sequence[DatasetFrame], calibration: Sequence[DatasetFrame]
) -> tuple[DatasetFrame, ...]:
    """Возвращает все оставшиеся кадры и строго проверяет непересечение."""

    calibration_sources = {frame.source_file for frame in calibration}
    selected = tuple(frame for frame in frames if frame.source_file not in calibration_sources)
    validation_sources = {frame.source_file for frame in selected}
    if calibration_sources & validation_sources:
        raise ValueError("Calibration and validation sources overlap")
    return selected


def write_manifest(path: Path, rows: Sequence[DatasetFrame]) -> None:
    """Записывает manifest CSV с фиксированной и читаемой схемой."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(DatasetFrame)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def read_manifest(path: Path) -> tuple[DatasetFrame, ...]:
    """Читает manifest CSV и проверяет отсутствие повторов."""

    rows = tuple(DatasetFrame(**row) for row in _read_csv(path))
    names = [row.source_file for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Manifest contains duplicate sources: {path}")
    return rows


def index_source_images(dataset_root: Path) -> Mapping[str, Path]:
    """Индексирует изображения и разрешает пути manifest без угадывания."""

    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {dataset_root}")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        path for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    relative = {path.relative_to(dataset_root).as_posix(): path for path in paths}
    by_name: dict[str, list[Path]] = {}
    for path in paths:
        by_name.setdefault(path.name, []).append(path)

    index = dict(relative)
    for name, candidates in by_name.items():
        if len(candidates) == 1:
            index.setdefault(name, candidates[0])
    return index


def resolve_source(index: Mapping[str, Path], source_file: str) -> Path:
    """Находит исходный кадр по точному относительному пути или уникальному имени."""

    path = index.get(Path(source_file).as_posix()) or index.get(Path(source_file).name)
    if path is None:
        raise FileNotFoundError(f"Source image is missing or ambiguous: {source_file}")
    return path
