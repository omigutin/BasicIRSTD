"""Объединяет готовый IRSTD-1K YOLO с проверенной COCO-разметкой IWT.

Исходные наборы только читаются. Результат записывается после полного аудита
COCO, изображений и готовых YOLO-разметок IRSTD.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2


EXPECTED_IWT_IMAGES = 164
EXPECTED_IWT_OBJECTS = 168
SOURCE_CATEGORY_ID = 1
YOLO_CLASS_ID = 0


@dataclass(frozen=True)
class BoundingBox:
    """Хранит прямоугольник COCO в пикселях."""

    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        """Возвращает площадь прямоугольника в пикселях."""
        return self.width * self.height


@dataclass(frozen=True)
class CocoImage:
    """Хранит проверенные метаданные изображения COCO."""

    image_id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoAnnotation:
    """Хранит проверенную аннотацию целевой категории."""

    annotation_id: int
    image_id: int
    category_id: int
    box: BoundingBox


@dataclass(frozen=True)
class AuditedIwtImage:
    """Связывает IWT-изображение с проверенными рамками."""

    metadata: CocoImage
    source_path: Path
    boxes: tuple[BoundingBox, ...]


@dataclass(frozen=True)
class SplitAudit:
    """Хранит число изображений и объектов готового YOLO split."""

    images: int
    objects: int


def require_mapping(value: object, context: str) -> Mapping[str, object]:
    """Проверяет, что JSON-значение является объектом."""
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object for {context}")
    return value


def require_sequence(value: object, context: str) -> Sequence[object]:
    """Проверяет, что JSON-значение является массивом."""
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array for {context}")
    return value


def require_int(mapping: Mapping[str, object], key: str, context: str) -> int:
    """Читает обязательное целое поле без неявного преобразования."""
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Expected integer {key} in {context}")
    return value


def require_string(mapping: Mapping[str, object], key: str, context: str) -> str:
    """Читает обязательную непустую строку."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string {key} in {context}")
    return value


def parse_box(value: object, context: str) -> BoundingBox:
    """Преобразует COCO bbox в типизированный прямоугольник."""
    coordinates = require_sequence(value, f"{context}.bbox")
    if len(coordinates) != 4 or any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in coordinates
    ):
        raise ValueError(f"Expected four numeric bbox values in {context}")
    x, y, width, height = (float(item) for item in coordinates)
    if not all(math.isfinite(item) for item in (x, y, width, height)):
        raise ValueError(f"BBox values must be finite in {context}")
    return BoundingBox(x=x, y=y, width=width, height=height)


def validate_box(box: BoundingBox, image: CocoImage, context: str) -> None:
    """Запрещает неположительные и выходящие за изображение рамки."""
    if box.width <= 0 or box.height <= 0:
        raise ValueError(f"BBox must have positive size in {context}")
    if box.x < 0 or box.y < 0 or box.x + box.width > image.width or box.y + box.height > image.height:
        raise ValueError(f"BBox is outside image bounds in {context}")


def yolo_label(box: BoundingBox, image_width: int, image_height: int) -> str:
    """Переводит пиксельную COCO-рамку в строку YOLO Detect."""
    center_x = (box.x + box.width / 2) / image_width
    center_y = (box.y + box.height / 2) / image_height
    return (
        f"{YOLO_CLASS_ID} {center_x:.10f} {center_y:.10f} "
        f"{box.width / image_width:.10f} {box.height / image_height:.10f}"
    )


def audit_coco_dataset(
    annotation_path: Path,
    image_dir: Path,
    expected_images: int | None = EXPECTED_IWT_IMAGES,
    expected_objects: int | None = EXPECTED_IWT_OBJECTS,
) -> list[AuditedIwtImage]:
    """Читает COCO и проверяет категории, файлы, размеры и рамки."""
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
    root = require_mapping(json.loads(annotation_path.read_text(encoding="utf-8")), "COCO root")
    raw_categories = require_sequence(root.get("categories"), "categories")
    categories: dict[int, str] = {}
    for index, raw_category in enumerate(raw_categories):
        category = require_mapping(raw_category, f"categories[{index}]")
        category_id = require_int(category, "id", f"categories[{index}]")
        name = require_string(category, "name", f"categories[{index}]")
        categories[category_id] = name
    if categories.get(SOURCE_CATEGORY_ID) != "bpla":
        raise ValueError("COCO category 1 must be named bpla")

    images: dict[int, CocoImage] = {}
    file_names: set[str] = set()
    for index, raw_image in enumerate(require_sequence(root.get("images"), "images")):
        item = require_mapping(raw_image, f"images[{index}]")
        image = CocoImage(
            image_id=require_int(item, "id", f"images[{index}]"),
            file_name=require_string(item, "file_name", f"images[{index}]"),
            width=require_int(item, "width", f"images[{index}]"),
            height=require_int(item, "height", f"images[{index}]"),
        )
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"Image dimensions must be positive: {image.file_name}")
        if image.image_id in images or image.file_name in file_names:
            raise ValueError(f"Duplicate COCO image id or filename: {image.file_name}")
        images[image.image_id] = image
        file_names.add(image.file_name)

    annotations: defaultdict[int, list[BoundingBox]] = defaultdict(list)
    seen_annotation_ids: set[int] = set()
    for index, raw_annotation in enumerate(require_sequence(root.get("annotations"), "annotations")):
        item = require_mapping(raw_annotation, f"annotations[{index}]")
        context = f"annotations[{index}]"
        annotation = CocoAnnotation(
            annotation_id=require_int(item, "id", context),
            image_id=require_int(item, "image_id", context),
            category_id=require_int(item, "category_id", context),
            box=parse_box(item.get("bbox"), context),
        )
        if annotation.annotation_id in seen_annotation_ids:
            raise ValueError(f"Duplicate COCO annotation id: {annotation.annotation_id}")
        seen_annotation_ids.add(annotation.annotation_id)
        if annotation.category_id != SOURCE_CATEGORY_ID:
            name = categories.get(annotation.category_id, "unknown")
            raise ValueError(f"Unsupported used category: id={annotation.category_id}, name={name}")
        image = images.get(annotation.image_id)
        if image is None:
            raise ValueError(f"Annotation references unknown image id: {annotation.image_id}")
        validate_box(annotation.box, image, context)
        annotations[annotation.image_id].append(annotation.box)

    if expected_images is not None and len(images) != expected_images:
        raise ValueError(f"Expected {expected_images} COCO images, found {len(images)}")
    object_count = sum(len(boxes) for boxes in annotations.values())
    if expected_objects is not None and object_count != expected_objects:
        raise ValueError(f"Expected {expected_objects} IWT boxes, found {object_count}")

    audited: list[AuditedIwtImage] = []
    for image in images.values():
        source_path = image_dir / image.file_name
        if not source_path.is_file():
            raise FileNotFoundError(f"COCO image not found: {source_path}")
        pixels = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise ValueError(f"COCO image is unreadable: {source_path}")
        actual_height, actual_width = pixels.shape[:2]
        if (actual_width, actual_height) != (image.width, image.height):
            raise ValueError(
                f"COCO image size mismatch for {image.file_name}: "
                f"metadata={image.width}x{image.height}, actual={actual_width}x{actual_height}"
            )
        audited.append(AuditedIwtImage(image, source_path, tuple(annotations[image.image_id])))
    return audited


def audit_yolo_split(dataset: Path, split: str) -> SplitAudit:
    """Проверяет соответствие PNG-изображений и YOLO label-файлов."""
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"YOLO split directories not found: {split}")
    images = sorted(image_dir.glob("*.png"))
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if image_stems != label_stems:
        raise ValueError(f"YOLO image/label mismatch in split: {split}")
    objects = 0
    for path in labels:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError(f"Invalid YOLO label at {path}:{line_number}")
            try:
                values = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise ValueError(f"Invalid YOLO numbers at {path}:{line_number}") from error
            if any(value < 0 or value > 1 for value in values) or values[2] <= 0 or values[3] <= 0:
                raise ValueError(f"Invalid normalized bbox at {path}:{line_number}")
            objects += 1
    return SplitAudit(images=len(images), objects=objects)


def prefixed_iwt_name(file_name: str) -> str:
    """Добавляет безопасный префикс к имени IWT-файла."""
    name = Path(file_name)
    if name.name != file_name or name.suffix == "":
        raise ValueError(f"COCO file_name must be a plain filename with suffix: {file_name}")
    return f"iwt_{name.name}"


def write_combined_dataset(irstd: Path, iwt: list[AuditedIwtImage], output: Path) -> None:
    """Копирует исходники и создаёт объединённую YOLO-разметку."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for source in (irstd / "images" / split).glob("*.png"):
            shutil.copy2(source, output / "images" / split / source.name)
        for source in (irstd / "labels" / split).glob("*.txt"):
            shutil.copy2(source, output / "labels" / split / source.name)
    existing_names = {path.name for path in (output / "images" / "train").iterdir()}
    for item in iwt:
        image_name = prefixed_iwt_name(item.metadata.file_name)
        if image_name in existing_names:
            raise ValueError(f"Combined image filename collision: {image_name}")
        existing_names.add(image_name)
        shutil.copy2(item.source_path, output / "images" / "train" / image_name)
        labels = [yolo_label(box, item.metadata.width, item.metadata.height) for box in item.boxes]
        (output / "labels" / "train" / f"{Path(image_name).stem}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
        )
    (output / "dataset.yaml").write_text(
        "train: images/train\nval: images/val\n\nnames:\n  0: bpla\n", encoding="utf-8"
    )


def select_check_images(items: list[AuditedIwtImage], random_count: int, seed: int) -> list[AuditedIwtImage]:
    """Выбирает IWT-примеры с крайними, множественными и случайными рамками."""
    with_boxes = [item for item in items if item.boxes]
    ranked = sorted(
        ((box.area, item) for item in with_boxes for box in item.boxes),
        key=lambda pair: pair[0],
    )
    selected = [ranked[0][1], ranked[-1][1]] if ranked else []
    multiple = next((item for item in with_boxes if len(item.boxes) > 1), None)
    if multiple is not None:
        selected.append(multiple)
    unique = list(dict.fromkeys(selected))
    remaining = [item for item in items if item not in unique]
    unique.extend(random.Random(seed).sample(remaining, min(random_count, len(remaining))))
    return unique


def render_dataset_checks(items: list[AuditedIwtImage], output: Path) -> None:
    """Рисует рамки на отдельных копиях для ручной проверки."""
    check_dir = output / "dataset_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        image = cv2.imread(str(item.source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"IWT image became unreadable: {item.source_path}")
        for box in item.boxes:
            cv2.rectangle(
                image,
                (round(box.x), round(box.y)),
                (round(box.x + box.width), round(box.y + box.height)),
                (0, 0, 255),
                2,
            )
        destination = check_dir / f"check_{Path(prefixed_iwt_name(item.metadata.file_name)).stem}.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Failed to write check image: {destination}")


def print_audit(train: SplitAudit, val: SplitAudit, iwt: list[AuditedIwtImage]) -> None:
    """Печатает требуемые количества и статистику IWT-рамок."""
    areas = [box.area for item in iwt for box in item.boxes]
    iwt_objects = len(areas)
    print("\nCombined dataset audit")
    print(f"IRSTD train images/objects: {train.images}/{train.objects}")
    print(f"IWT train images/objects: {len(iwt)}/{iwt_objects}")
    print(f"Combined train images/objects: {train.images + len(iwt)}/{train.objects + iwt_objects}")
    print(f"Val images/objects: {val.images}/{val.objects}")
    print("IWT bbox area (pixels):")
    print(f"  min: {min(areas):.2f}")
    print(f"  median: {statistics.median(areas):.2f}")
    print(f"  mean: {statistics.fmean(areas):.2f}")
    print(f"  max: {max(areas):.2f}")
    print(f"Multi-object IWT images: {sum(len(item.boxes) > 1 for item in iwt)}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт параметры путей и визуальной выборки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--irstd", type=Path, default=Path("datasets/IRSTD-1K-YOLO"))
    parser.add_argument(
        "--iwt", type=Path, default=Path("__DATASETS__/iwt_device_all_bpla_MARKED/train")
    )
    parser.add_argument(
        "--annotations", type=Path, default=None, help="По умолчанию <iwt>/_annotations.coco.json."
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/IRSTD-1K-IWT-YOLO"))
    parser.add_argument("--check-random", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    """Выполняет аудит, запись и контроль результата."""
    args = build_argument_parser().parse_args()
    if args.check_random < 0:
        raise ValueError("--check-random must be non-negative")
    irstd = args.irstd.resolve()
    iwt_dir = args.iwt.resolve()
    annotations = (args.annotations or iwt_dir / "_annotations.coco.json").resolve()
    train = audit_yolo_split(irstd, "train")
    val = audit_yolo_split(irstd, "val")
    if (train.images, val.images) != (800, 201):
        raise ValueError(f"Expected IRSTD train/val image counts 800/201, found {train.images}/{val.images}")
    iwt = audit_coco_dataset(annotations, iwt_dir)
    output = args.output.resolve()
    write_combined_dataset(irstd, iwt, output)
    render_dataset_checks(select_check_images(iwt, args.check_random, args.seed), output)
    combined_train = audit_yolo_split(output, "train")
    combined_val = audit_yolo_split(output, "val")
    if combined_train.images != 964 or combined_val.images != 201:
        raise RuntimeError("Unexpected combined dataset image counts after writing")
    print_audit(train, val, iwt)
    print(f"\nYOLO dataset: {output}")
    print(f"Visual checks: {output / 'dataset_check'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
