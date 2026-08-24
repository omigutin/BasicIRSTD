"""Проверяет строгую конвертацию COCO-разметки IWT в YOLO."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PREPARE_SCRIPT = REPOSITORY_ROOT / "iwt_tools/scripts/training/prepare_irstd1k_iwt_yolo.py"


def load_module() -> ModuleType:
    """Загружает CLI-скрипт для проверки его чистых функций."""
    cv2 = ModuleType("cv2")
    cv2.IMREAD_UNCHANGED = -1

    class TestPixels:
        """Имитирует только размер изображения, нужный аудиту."""

        shape = (50, 100, 3)

    def imread(_path: str, _mode: int) -> TestPixels:
        """Возвращает фиксированный размер синтетического изображения."""
        return TestPixels()

    cv2.imread = imread
    sys.modules["cv2"] = cv2
    spec = importlib.util.spec_from_file_location("prepare_irstd1k_iwt_yolo_test", PREPARE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test module: {PREPARE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_fixture(
    root: Path,
    annotations: list[dict[str, object]],
    categories: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    """Создаёт минимальный COCO dataset с одним изображением."""
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "sample.png").write_bytes(b"synthetic image placeholder")
    coco = {
        "images": [{"id": 7, "file_name": "sample.png", "width": 100, "height": 50}],
        "annotations": annotations,
        "categories": categories or [{"id": 1, "name": "bpla"}],
    }
    annotation_path = root / "annotations.json"
    annotation_path.write_text(json.dumps(coco), encoding="utf-8")
    return annotation_path, image_dir


def annotation(
    annotation_id: int,
    bbox: list[float],
    category_id: int = 1,
) -> dict[str, object]:
    """Создаёт тестовую COCO-аннотацию."""
    return {
        "id": annotation_id,
        "image_id": 7,
        "category_id": category_id,
        "bbox": bbox,
    }


def audit_fixture(module: ModuleType, annotation_path: Path, image_dir: Path) -> list[object]:
    """Запускает аудит без production-ограничений количества."""
    return module.audit_coco_dataset(
        annotation_path, image_dir, expected_images=None, expected_objects=None
    )


def test_category_one_becomes_yolo_class_zero_and_bbox_is_converted(tmp_path: Path) -> None:
    """Проверяет отображение класса и нормализацию COCO bbox."""
    module = load_module()
    annotation_path, image_dir = write_fixture(tmp_path, [annotation(1, [10, 5, 20, 10])])
    audited = audit_fixture(module, annotation_path, image_dir)
    assert module.yolo_label(audited[0].boxes[0], 100, 50) == (
        "0 0.2000000000 0.2000000000 0.2000000000 0.2000000000"
    )


def test_duplicate_named_unused_category_zero_does_not_interfere(tmp_path: Path) -> None:
    """Проверяет допустимую неиспользуемую запись bpla с ID 0."""
    module = load_module()
    categories = [{"id": 0, "name": "bpla"}, {"id": 1, "name": "bpla"}]
    annotation_path, image_dir = write_fixture(
        tmp_path, [annotation(1, [10, 5, 20, 10])], categories
    )
    assert len(audit_fixture(module, annotation_path, image_dir)) == 1


def test_unknown_used_category_is_rejected(tmp_path: Path) -> None:
    """Проверяет ошибку для реально используемой неизвестной категории."""
    module = load_module()
    annotation_path, image_dir = write_fixture(
        tmp_path, [annotation(1, [10, 5, 20, 10], category_id=2)]
    )
    with pytest.raises(ValueError, match="Unsupported used category"):
        audit_fixture(module, annotation_path, image_dir)


def test_multi_object_image_creates_multiple_label_lines(tmp_path: Path) -> None:
    """Проверяет отдельную YOLO-строку для каждой рамки изображения."""
    module = load_module()
    annotation_path, image_dir = write_fixture(
        tmp_path,
        [annotation(1, [10, 5, 20, 10]), annotation(2, [40, 20, 10, 5])],
    )
    audited = audit_fixture(module, annotation_path, image_dir)
    lines = [module.yolo_label(box, 100, 50) for box in audited[0].boxes]
    assert len(lines) == 2
    assert all(line.startswith("0 ") for line in lines)


@pytest.mark.parametrize(
    "bbox,error",
    [
        ([1, 1, 0, 3], "positive size"),
        ([95, 1, 10, 3], "outside image bounds"),
        ([-1, 1, 2, 3], "outside image bounds"),
    ],
)
def test_invalid_or_out_of_bounds_bbox_is_rejected(
    tmp_path: Path, bbox: list[float], error: str
) -> None:
    """Проверяет аудит неположительных и выходящих за границы рамок."""
    module = load_module()
    annotation_path, image_dir = write_fixture(tmp_path, [annotation(1, bbox)])
    with pytest.raises(ValueError, match=error):
        audit_fixture(module, annotation_path, image_dir)
