"""Аудит COCO-разметки и подготовка таблиц ground truth для real_eval."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

from PIL import Image


IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
ROBOFLOW_SUFFIX = re.compile(
    r"\.rf\.[^.]+\.(?:png|jpe?g|bmp|tiff?)$",
    re.IGNORECASE,
)


class SizeClass(str, Enum):
    """Размер объекта относительно полной площади изображения."""

    TINY = "Tiny"
    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"


class MatchStatus(str, Enum):
    """Результат сопоставления COCO-записи с исходным изображением."""

    MATCH = "MATCH"
    ROTATED_DIMENSIONS = "ROTATED_DIMENSIONS"
    RESIZED = "RESIZED"
    MISSING = "MISSING"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class GroundTruthObject:
    """Один размеченный объект с геометрией и размерным классом."""

    image: str
    annotation_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    image_width: int
    image_height: int
    image_area: int
    area_ratio: float
    size_class: str


@dataclass(frozen=True)
class GroundTruthImage:
    """Количество объектов каждого размера на одном positive-кадре."""

    image: str
    gt_target_count: int
    tiny_count: int
    small_count: int
    medium_count: int
    large_count: int


@dataclass(frozen=True)
class ImageMatch:
    """Итог проверки одной COCO image-записи."""

    image_id: int
    coco_name: str
    status: MatchStatus
    path: Optional[Path]
    relative_path: Optional[str]
    coco_width: int
    coco_height: int
    actual_width: Optional[int] = None
    actual_height: Optional[int] = None
    detail: str = ""


@dataclass(frozen=True)
class AuditResult:
    """Проверенный индекс, достаточный для безопасной генерации CSV."""

    coco_path: Path
    positive_frames: int
    images: tuple[Mapping[str, Any], ...]
    annotations: tuple[Mapping[str, Any], ...]
    matches: Mapping[int, ImageMatch]
    annotations_by_image: Mapping[int, tuple[Mapping[str, Any], ...]]
    images_without_annotations: tuple[str, ...]


class AuditError(ValueError):
    """Ошибка данных, при которой ground truth нельзя безопасно создать."""


def classify_size(area_ratio: float) -> SizeClass:
    """Определяет фиксированный размерный класс отдельного объекта."""

    if area_ratio < 0.00075:
        return SizeClass.TINY
    if area_ratio < 0.0015:
        return SizeClass.SMALL
    if area_ratio < 0.01:
        return SizeClass.MEDIUM
    return SizeClass.LARGE


def find_coco_json(coco_export: Path) -> Path:
    """Находит единственный `_annotations.coco.json` в COCO-export."""

    if not coco_export.exists():
        raise FileNotFoundError(f"COCO export does not exist: {coco_export}")
    if coco_export.is_file():
        candidates = [coco_export] if coco_export.name == "_annotations.coco.json" else []
    else:
        candidates = sorted(coco_export.rglob("_annotations.coco.json"))
    if len(candidates) != 1:
        listed = ", ".join(str(path) for path in candidates) or "none"
        raise AuditError(
            "Expected exactly one _annotations.coco.json in COCO export, "
            f"found {len(candidates)}: {listed}"
        )
    return candidates[0]


def index_images(root: Path) -> tuple[Path, ...]:
    """Рекурсивно индексирует поддерживаемые изображения dataset root."""

    if not root.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {root}")
    return tuple(sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ))


def _require_list(data: Mapping[str, Any], section: str) -> list[Any]:
    """Возвращает обязательную COCO-секцию с проверкой типа."""

    value = data.get(section)
    if not isinstance(value, list):
        raise AuditError(f"COCO section '{section}' must be a list")
    return value


def _load_coco(path: Path) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Читает основные COCO-секции без зависимости pycocotools."""

    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"Cannot read COCO JSON: {error}") from error
    if not isinstance(data, dict):
        raise AuditError("COCO root must be an object")
    images = _require_list(data, "images")
    annotations = _require_list(data, "annotations")
    _require_list(data, "categories")
    if not all(isinstance(item, dict) for item in images + annotations):
        raise AuditError("COCO images and annotations must contain objects")
    return images, annotations


def _validate_unique_ids(records: Iterable[Mapping[str, Any]], name: str) -> None:
    """Проверяет наличие и уникальность числовых COCO id."""

    ids = [record.get("id") for record in records]
    if any(not isinstance(value, int) for value in ids):
        raise AuditError(f"Every COCO {name} record must have an integer id")
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise AuditError(f"Duplicate COCO {name} ids: {duplicates}")


def _validate_annotations(
    annotations: Sequence[Mapping[str, Any]],
    images_by_id: Mapping[int, Mapping[str, Any]],
) -> dict[int, tuple[Mapping[str, Any], ...]]:
    """Проверяет ссылки, bbox, segmentation и category каждой annotation."""

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    for annotation in annotations:
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        image = images_by_id.get(image_id) if isinstance(image_id, int) else None
        if image is None:
            errors.append(f"annotation {annotation_id}: unknown image_id {image_id}")
            continue
        if "category_id" not in annotation:
            errors.append(f"annotation {annotation_id}: missing category_id")
        if "segmentation" not in annotation:
            errors.append(f"annotation {annotation_id}: missing segmentation")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) for value in bbox
        ):
            errors.append(f"annotation {annotation_id}: bbox must be [x, y, width, height]")
            continue
        x, y, width, height = (float(value) for value in bbox)
        if width <= 0 or height <= 0:
            errors.append(f"annotation {annotation_id}: bbox width and height must be positive")
        image_width = image.get("width")
        image_height = image.get("height")
        if not isinstance(image_width, int) or not isinstance(image_height, int):
            errors.append(f"image {image_id}: width and height must be integers")
        elif x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            errors.append(f"annotation {annotation_id}: bbox is outside COCO image bounds")
        grouped[image_id].append(annotation)
    if errors:
        raise AuditError("COCO annotation audit failed:\n- " + "\n- ".join(errors))
    return {image_id: tuple(items) for image_id, items in grouped.items()}


def _coco_roboflow_key(file_name: str) -> Optional[str]:
    """Извлекает строгий ключ перед `.rf.<hash>.<export_extension>`."""

    basename = Path(file_name).name
    match = ROBOFLOW_SUFFIX.search(basename)
    if match is None:
        return None
    return basename[:match.start()].casefold()


def _source_roboflow_key(path: Path) -> str:
    """Кодирует исходное имя по правилу Roboflow без обратных замен."""

    extension = path.suffix.removeprefix(".").casefold()
    encoded_stem = path.stem.replace(".", "-").casefold()
    return f"{encoded_stem}_{extension}"


def _match_candidates(
    coco_name: str,
    positive_root: Path,
    paths: Sequence[Path],
) -> tuple[Path, ...]:
    """Ищет exact match, затем уникальное стандартное Roboflow-имя без fuzzy matching."""

    normalized_name = coco_name.replace("\\", "/")
    exact_relative = tuple(
        path for path in paths
        if path.relative_to(positive_root).as_posix().casefold() == normalized_name.casefold()
    )
    if exact_relative:
        return exact_relative
    exact_basename = tuple(path for path in paths if path.name.casefold() == Path(coco_name).name.casefold())
    if exact_basename:
        return exact_basename
    coco_key = _coco_roboflow_key(coco_name)
    if coco_key is None:
        return ()
    return tuple(path for path in paths if _source_roboflow_key(path) == coco_key)


def _audit_matches(
    images: Sequence[Mapping[str, Any]],
    positive_root: Path,
    positive_paths: Sequence[Path],
) -> dict[int, ImageMatch]:
    """Сопоставляет COCO images и строго проверяет размеры исходных файлов."""

    results: dict[int, ImageMatch] = {}
    for image in images:
        image_id = image.get("id")
        file_name = image.get("file_name")
        width = image.get("width")
        height = image.get("height")
        if not isinstance(image_id, int) or not isinstance(file_name, str) or not file_name:
            raise AuditError("Every COCO image must have integer id and non-empty file_name")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise AuditError(f"COCO image {image_id} must have positive integer width and height")
        candidates = _match_candidates(file_name, positive_root, positive_paths)
        if not candidates:
            results[image_id] = ImageMatch(
                image_id, file_name, MatchStatus.MISSING, None, None, width, height,
                detail="No exact or unambiguous Roboflow-name match",
            )
            continue
        if len(candidates) != 1:
            names = ", ".join(path.relative_to(positive_root).as_posix() for path in candidates)
            results[image_id] = ImageMatch(
                image_id, file_name, MatchStatus.AMBIGUOUS, None, None, width, height,
                detail=f"Candidates: {names}",
            )
            continue
        path = candidates[0]
        try:
            with Image.open(path) as opened:
                actual_width, actual_height = opened.size
        except (OSError, ValueError) as error:
            raise AuditError(f"Cannot read positive image {path}: {error}") from error
        if (actual_width, actual_height) == (width, height):
            status = MatchStatus.MATCH
        elif (actual_width, actual_height) == (height, width):
            status = MatchStatus.ROTATED_DIMENSIONS
        else:
            status = MatchStatus.RESIZED
        results[image_id] = ImageMatch(
            image_id=image_id,
            coco_name=file_name,
            status=status,
            path=path,
            relative_path=path.relative_to(positive_root).as_posix(),
            coco_width=width,
            coco_height=height,
            actual_width=actual_width,
            actual_height=actual_height,
        )
    return results


def audit_dataset(positive_root: Path, coco_export: Path) -> AuditResult:
    """Выполняет полный аудит COCO и сопоставления с positive PNG."""

    positive_paths = index_images(positive_root)
    coco_path = find_coco_json(coco_export)
    images, annotations = _load_coco(coco_path)
    _validate_unique_ids(images, "image")
    _validate_unique_ids(annotations, "annotation")

    file_names = [image.get("file_name") for image in images]
    duplicate_names = sorted(name for name, count in Counter(file_names).items() if count > 1)
    basenames = [Path(name).name.casefold() for name in file_names if isinstance(name, str)]
    duplicate_basenames = sorted(name for name, count in Counter(basenames).items() if count > 1)
    if duplicate_names or duplicate_basenames:
        raise AuditError(
            f"Duplicate COCO filenames: {duplicate_names}; duplicate basenames: {duplicate_basenames}"
        )

    images_by_id = {int(image["id"]): image for image in images}
    annotations_by_image = _validate_annotations(annotations, images_by_id)
    without_annotations = tuple(
        str(image["file_name"]) for image in images
        if int(image["id"]) not in annotations_by_image
    )
    matches = _audit_matches(images, positive_root, positive_paths)
    return AuditResult(
        coco_path=coco_path,
        positive_frames=len(positive_paths),
        images=tuple(images),
        annotations=tuple(annotations),
        matches=matches,
        annotations_by_image=annotations_by_image,
        images_without_annotations=without_annotations,
    )


def print_audit_report(audit: AuditResult, positive_frames: int) -> None:
    """Печатает сводку и все небезопасные результаты сопоставления."""

    counts = Counter(match.status for match in audit.matches.values())
    print("Audit report")
    print(f"COCO file: {audit.coco_path}")
    print(f"Positive frames: {positive_frames}")
    print(f"COCO images: {len(audit.images)}")
    print(f"COCO annotations: {len(audit.annotations)}")
    print(f"Images without annotation: {len(audit.images_without_annotations)}")
    for status in MatchStatus:
        print(f"{status.value}: {counts[status]}")
    for match in audit.matches.values():
        if match.status is MatchStatus.MATCH:
            continue
        print(
            f"[{match.status.value}] {match.coco_name}: "
            f"COCO={match.coco_width}x{match.coco_height}, "
            f"actual={match.actual_width}x{match.actual_height}; {match.detail}"
        )
    if audit.images_without_annotations:
        print("COCO images without annotation:")
        for name in audit.images_without_annotations:
            print(f"- {name}")


def _build_rows(audit: AuditResult) -> tuple[list[GroundTruthObject], list[GroundTruthImage]]:
    """Создаёт объектные и покадровые строки после успешного аудита."""

    object_rows: list[GroundTruthObject] = []
    image_rows: list[GroundTruthImage] = []
    for image in sorted(audit.images, key=lambda item: str(item["file_name"])):
        image_id = int(image["id"])
        match = audit.matches[image_id]
        if match.relative_path is None:
            raise AuditError(f"Successful audit has no path for COCO image {image_id}")
        image_width = int(image["width"])
        image_height = int(image["height"])
        image_area = image_width * image_height
        size_counts: Counter[SizeClass] = Counter()
        annotations = audit.annotations_by_image.get(image_id, ())
        for annotation in sorted(annotations, key=lambda item: int(item["id"])):
            x, y, width, height = (float(value) for value in annotation["bbox"])
            bbox_area = width * height
            area_ratio = bbox_area / image_area
            size_class = classify_size(area_ratio)
            size_counts[size_class] += 1
            object_rows.append(GroundTruthObject(
                image=match.relative_path,
                annotation_id=int(annotation["id"]),
                x1=x,
                y1=y,
                x2=x + width,
                y2=y + height,
                bbox_width=width,
                bbox_height=height,
                bbox_area=bbox_area,
                image_width=image_width,
                image_height=image_height,
                image_area=image_area,
                area_ratio=area_ratio,
                size_class=size_class.value,
            ))
        image_rows.append(GroundTruthImage(
            image=match.relative_path,
            gt_target_count=len(annotations),
            tiny_count=size_counts[SizeClass.TINY],
            small_count=size_counts[SizeClass.SMALL],
            medium_count=size_counts[SizeClass.MEDIUM],
            large_count=size_counts[SizeClass.LARGE],
        ))
    return object_rows, image_rows


def _write_dataclasses(path: Path, rows: Sequence[Any], fieldnames: Sequence[str]) -> None:
    """Записывает типизированные строки в CSV с фиксированным порядком колонок."""

    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_negative_images(
    path: Path,
    negative_roots: Sequence[Path],
) -> Mapping[str, int]:
    """Записывает относительные пути negative-кадров и возвращает счётчики."""

    allowed_names = {"clear_horizon", "clear_sky"}
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    seen_sets: set[str] = set()
    for root in negative_roots:
        source_set = root.name
        if source_set not in allowed_names:
            raise AuditError(
                f"Negative directory name must be clear_horizon or clear_sky: {root}"
            )
        if source_set in seen_sets:
            raise AuditError(f"Duplicate negative source set: {source_set}")
        seen_sets.add(source_set)
        for image_path in index_images(root):
            rows.append({
                "image": image_path.relative_to(root).as_posix(),
                "source_set": source_set,
            })
            counts[source_set] += 1
    missing_sets = allowed_names - seen_sets
    if missing_sets:
        raise AuditError(f"Missing negative source sets: {sorted(missing_sets)}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "source_set"])
        writer.writeheader()
        writer.writerows(rows)
    return counts


def generate_csv(
    audit: AuditResult,
    output_dir: Path,
    negative_roots: Sequence[Path],
) -> None:
    """Создаёт три CSV только для полностью безопасного audit result."""

    bad_matches = [match for match in audit.matches.values() if match.status is not MatchStatus.MATCH]
    if bad_matches:
        raise AuditError("Ground truth CSV was not generated because image matching audit failed")
    if audit.images_without_annotations:
        raise AuditError("Ground truth CSV was not generated because COCO has images without annotation")

    object_rows, image_rows = _build_rows(audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataclasses(
        output_dir / "ground_truth_objects.csv",
        object_rows,
        tuple(GroundTruthObject.__dataclass_fields__),
    )
    _write_dataclasses(
        output_dir / "ground_truth_images.csv",
        image_rows,
        tuple(GroundTruthImage.__dataclass_fields__),
    )
    negative_counts = _write_negative_images(
        output_dir / "negative_images.csv", negative_roots
    )

    sizes = Counter(row.size_class for row in object_rows)
    print("\nGeneration summary")
    print(f"Positive frames: {audit.positive_frames}")
    print(f"Annotated positive frames: {len(image_rows)}")
    print(f"GT objects: {len(object_rows)}")
    print(f"Tiny: {sizes[SizeClass.TINY.value]}")
    print(f"Small: {sizes[SizeClass.SMALL.value]}")
    print(f"Medium: {sizes[SizeClass.MEDIUM.value]}")
    print(f"Large: {sizes[SizeClass.LARGE.value]}")
    print(f"clear_horizon frames: {negative_counts['clear_horizon']}")
    print(f"clear_sky frames: {negative_counts['clear_sky']}")
    print(f"Total negative frames: {sum(negative_counts.values())}")
    print(f"COCO images: {len(audit.images)}")
    print(f"COCO annotations: {len(audit.annotations)}")
    print(f"Images without annotation: {len(audit.images_without_annotations)}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI для локального Windows-аудита и генерации CSV."""

    parser = argparse.ArgumentParser(description="Audit COCO ground truth and create real_eval CSV files")
    parser.add_argument("--positive", required=True, type=Path, help="Positive image directory")
    parser.add_argument(
        "--negative", required=True, action="append", type=Path,
        help="Negative directory; pass clear_horizon and clear_sky separately",
    )
    parser.add_argument("--coco", required=True, type=Path, help="Roboflow COCO export directory")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for three CSV files")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Сначала проводит аудит и прекращает работу при любом несоответствии."""

    args = build_argument_parser().parse_args(argv)
    try:
        positive_paths = index_images(args.positive)
        for negative_root in args.negative:
            index_images(negative_root)
        audit = audit_dataset(args.positive, args.coco)
        print_audit_report(audit, len(positive_paths))
        generate_csv(audit, args.output, args.negative)
    except (AuditError, FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
