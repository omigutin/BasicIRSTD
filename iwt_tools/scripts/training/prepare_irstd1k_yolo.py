"""Проверяет IRSTD-1K и создаёт из масок датасет YOLO Detect.

Скрипт не изменяет исходный датасет и выполняет полный аудит до создания
выходных файлов. Объектом считается ненулевая 8-связная область, содержащая
хотя бы один пиксель со значением 255.
"""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Хранит прямоугольник объекта в пикселях."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        """Возвращает площадь прямоугольника в пикселях."""
        return self.width * self.height


@dataclass(frozen=True)
class AuditedImage:
    """Хранит проверенные сведения об изображении и его объектах."""

    split: str
    image_id: str
    image_path: Path
    mask_path: Path
    width: int
    height: int
    mask_values: tuple[int, ...]
    boxes: tuple[BoundingBox, ...]
    component_areas: tuple[int, ...]
    gray_pixel_count: int
    rejected_component_count: int


def read_split_ids(path: Path) -> list[str]:
    """Читает непустые идентификаторы и запрещает дубликаты."""
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    ids = [image_id for image_id in ids if image_id]
    if not ids:
        raise ValueError(f"Split file is empty: {path}")
    duplicates = sorted(image_id for image_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate IDs in {path}: {', '.join(duplicates[:20])}")
    return ids


def validate_splits(train_ids: list[str], val_ids: list[str]) -> None:
    """Проверяет, что обучающая и проверочная выборки не пересекаются."""
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(f"Train/val overlap detected: {', '.join(overlap[:20])}")


def load_mask(path: Path) -> np.ndarray:
    """Читает маску без неявного преобразования цветовых каналов."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Mask is unreadable: {path}")
    if mask.ndim == 3:
        channels = cv2.split(mask)
        if not all(np.array_equal(channels[0], channel) for channel in channels[1:]):
            raise ValueError(f"Mask has different color channels: {path}")
        mask = channels[0]
    if mask.ndim != 2:
        raise ValueError(f"Mask must be two-dimensional: {path}")
    return mask


def extract_components(
    mask: np.ndarray,
) -> tuple[tuple[BoundingBox, ...], tuple[int, ...], int]:
    """Возвращает только ненулевые области, подтверждённые пикселем 255."""
    binary_mask = np.asarray(mask != 0, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    anchored_indices = tuple(
        int(index) for index in np.unique(labels[mask == 255]) if index != 0
    )
    boxes = tuple(
        BoundingBox(
            x=int(stats[index, cv2.CC_STAT_LEFT]),
            y=int(stats[index, cv2.CC_STAT_TOP]),
            width=int(stats[index, cv2.CC_STAT_WIDTH]),
            height=int(stats[index, cv2.CC_STAT_HEIGHT]),
        )
        for index in anchored_indices
    )
    areas = tuple(
        int(stats[index, cv2.CC_STAT_AREA]) for index in anchored_indices
    )
    rejected_count = component_count - 1 - len(anchored_indices)
    return boxes, areas, rejected_count


def audit_image(
    dataset: Path,
    split: str,
    image_id: str,
) -> AuditedImage:
    """Проверяет одну пару image/mask и извлекает все компоненты."""
    image_path = dataset / "images" / f"{image_id}.png"
    mask_path = dataset / "masks" / f"{image_id}.png"
    missing = [str(path) for path in (image_path, mask_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required files not found: {', '.join(missing)}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Image is unreadable: {image_path}")
    mask = load_mask(mask_path)
    image_height, image_width = image.shape[:2]
    if (image_height, image_width) != mask.shape:
        raise ValueError(
            f"Image/mask size mismatch for {image_id}: "
            f"image={image_width}x{image_height}, mask={mask.shape[1]}x{mask.shape[0]}"
        )

    values = tuple(int(value) for value in np.unique(mask))
    gray_pixel_count = int(np.count_nonzero((mask != 0) & (mask != 255)))
    boxes, component_areas, rejected_component_count = extract_components(mask)
    return AuditedImage(
        split=split,
        image_id=image_id,
        image_path=image_path,
        mask_path=mask_path,
        width=image_width,
        height=image_height,
        mask_values=values,
        boxes=boxes,
        component_areas=component_areas,
        gray_pixel_count=gray_pixel_count,
        rejected_component_count=rejected_component_count,
    )


def run_audit(
    dataset: Path,
    train_ids: list[str],
    val_ids: list[str],
) -> list[AuditedImage]:
    """Выполняет полный аудит датасета до записи результата."""
    audited: list[AuditedImage] = []
    for split, ids in (("train", train_ids), ("val", val_ids)):
        for image_id in ids:
            audited.append(
                audit_image(dataset, split, image_id)
            )
    return audited


def print_audit(audited: list[AuditedImage], train_count: int, val_count: int) -> None:
    """Печатает статистику масок, компонент и рамок."""
    areas = [box.area for item in audited for box in item.boxes]
    component_areas = [area for item in audited for area in item.component_areas]
    empty_count = sum(not item.boxes for item in audited)
    multiple_count = sum(len(item.boxes) > 1 for item in audited)
    unique_sets = Counter(item.mask_values for item in audited)
    multivalue_count = sum(item.gray_pixel_count > 0 for item in audited)
    gray_pixel_count = sum(item.gray_pixel_count for item in audited)
    rejected_count = sum(item.rejected_component_count for item in audited)
    problem_files = [
        item.mask_path.name for item in audited if item.rejected_component_count > 0
    ]

    print("\nRuntime audit")
    print(f"Train IDs: {train_count}")
    print(f"Val IDs: {val_count}")
    print(f"Readable image/mask pairs: {len(audited)}")
    print(f"Empty masks: {empty_count}")
    print(f"Connected components: {len(areas)}")
    print(f"Images with multiple components: {multiple_count}")
    print(f"Bounding boxes: {len(areas)}")
    print(f"Multivalue masks: {multivalue_count}")
    print(f"Gray pixels: {gray_pixel_count}")
    print(f"Nonzero components without 255: {rejected_count}")
    print(f"Rejected components: {rejected_count}")
    print(f"Final objects: {len(areas)}")
    print("Files with nonzero components without 255:")
    if problem_files:
        for filename in problem_files:
            print(f"  {filename}")
    else:
        print("  none")
    print("Mask unique-value patterns:")
    for values, count in sorted(unique_sets.items(), key=lambda item: str(item[0])):
        print(f"  {values}: {count} mask(s)")
    if areas:
        print("BBox area (pixels):")
        print(f"  min: {min(areas)}")
        print(f"  median: {statistics.median(areas):.2f}")
        print(f"  mean: {statistics.fmean(areas):.2f}")
        print(f"  max: {max(areas)}")
    tiny_count = sum(area <= 2 for area in component_areas)
    if tiny_count:
        print(
            "WARNING: found "
            f"{tiny_count} one- or two-pixel component(s). They are preserved "
            "without area filtering."
        )
    if rejected_count:
        print(
            "WARNING: nonzero components without a 255 anchor were treated as "
            "artifacts and did not create bounding boxes."
        )


def yolo_label(box: BoundingBox, image_width: int, image_height: int) -> str:
    """Форматирует пиксельную рамку как нормализованную строку YOLO."""
    center_x = (box.x + box.width / 2) / image_width
    center_y = (box.y + box.height / 2) / image_height
    width = box.width / image_width
    height = box.height / image_height
    return f"0 {center_x:.10f} {center_y:.10f} {width:.10f} {height:.10f}"


def write_dataset(audited: list[AuditedImage], output: Path) -> None:
    """Копирует изображения и записывает YOLO-разметку без изменения исходника."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for item in audited:
        shutil.copy2(item.image_path, output / "images" / item.split / item.image_path.name)
        labels = [yolo_label(box, item.width, item.height) for box in item.boxes]
        label_path = output / "labels" / item.split / f"{item.image_id}.txt"
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    (output / "dataset.yaml").write_text(
        "train: images/train\nval: images/val\n\nnames:\n  0: bpla\n",
        encoding="utf-8",
    )


def select_check_images(
    audited: list[AuditedImage], random_count: int, seed: int
) -> list[AuditedImage]:
    """Выбирает крайние, типичные, многокомпонентные и случайные примеры."""
    nonempty = [item for item in audited if item.boxes]
    if not nonempty:
        return random.Random(seed).sample(audited, min(random_count, len(audited)))
    by_area = sorted(
        ((box.area, item) for item in nonempty for box in item.boxes),
        key=lambda pair: pair[0],
    )
    selected = [by_area[0][1], by_area[len(by_area) // 2][1], by_area[-1][1]]
    multiple = [item for item in nonempty if len(item.boxes) > 1]
    selected.extend(multiple[:2])
    remaining = [item for item in audited if item not in selected]
    selected.extend(random.Random(seed).sample(remaining, min(random_count, len(remaining))))
    return list(dict.fromkeys(selected))


def render_checks(items: list[AuditedImage], output: Path) -> None:
    """Рисует проверочные рамки и сохраняет изображения рядом с датасетом."""
    check_dir = output / "dataset_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        image = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image became unreadable: {item.image_path}")
        for box in item.boxes:
            cv2.rectangle(
                image,
                (box.x, box.y),
                (box.x + box.width - 1, box.y + box.height - 1),
                (0, 0, 255),
                1,
            )
        destination = check_dir / f"{item.split}_{item.image_id}.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Failed to write check image: {destination}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-параметры подготовки датасета."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/IRSTD-1K"))
    parser.add_argument("--output", type=Path, default=Path("datasets/IRSTD-1K-YOLO"))
    parser.add_argument("--check-random", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    """Проверяет аргументы, выполняет аудит и создаёт датасет."""
    args = build_argument_parser().parse_args()
    if args.check_random < 0:
        raise ValueError("--check-random must be non-negative")
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")
    train_ids = read_split_ids(dataset / "img_idx" / "train_IRSTD-1K.txt")
    val_ids = read_split_ids(dataset / "img_idx" / "test_IRSTD-1K.txt")
    validate_splits(train_ids, val_ids)
    audited = run_audit(dataset, train_ids, val_ids)
    print_audit(audited, len(train_ids), len(val_ids))
    output = args.output.resolve()
    write_dataset(audited, output)
    render_checks(select_check_images(audited, args.check_random, args.seed), output)
    print(f"\nYOLO dataset: {output}")
    print(f"Visual checks: {output / 'dataset_check'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
