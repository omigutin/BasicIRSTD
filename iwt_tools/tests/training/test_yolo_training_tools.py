"""Проверяет подготовку YOLO-разметки и чтение итогов обучения."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PREPARE_SCRIPT = REPOSITORY_ROOT / "iwt_tools/scripts/training/prepare_irstd1k_yolo.py"
TRAIN_SCRIPT = REPOSITORY_ROOT / "iwt_tools/scripts/training/train_yolo.py"


def load_script(path: Path, name: str) -> ModuleType:
    """Загружает CLI-скрипт как модуль для проверки чистых функций."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_png(path: Path, image: object, cv2_module: ModuleType) -> None:
    """Создаёт PNG для синтетического датасета."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2_module.imwrite(str(path), image)


def test_prepare_dataset_preserves_tiny_and_separate_components(tmp_path: Path) -> None:
    """Проверяет split, Tiny-цель и отдельные рамки независимых объектов."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    dataset = tmp_path / "IRSTD-1K"
    (dataset / "img_idx").mkdir(parents=True)
    (dataset / "img_idx/train_IRSTD-1K.txt").write_text("train_one\n", encoding="utf-8")
    (dataset / "img_idx/test_IRSTD-1K.txt").write_text("val_one\n", encoding="utf-8")
    image = np.full((10, 12), 80, dtype=np.uint8)
    train_mask = np.zeros((10, 12), dtype=np.uint8)
    train_mask[1, 1] = 255
    train_mask[5:7, 8:11] = 255
    val_mask = np.zeros((10, 12), dtype=np.uint8)
    for image_id, mask in (("train_one", train_mask), ("val_one", val_mask)):
        write_png(dataset / "images" / f"{image_id}.png", image, cv2)
        write_png(dataset / "masks" / f"{image_id}.png", mask, cv2)

    output = tmp_path / "IRSTD-1K-YOLO"
    result = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), "--dataset", str(dataset), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    labels = (output / "labels/train/train_one.txt").read_text(encoding="utf-8").splitlines()
    assert len(labels) == 2
    assert labels[0].endswith("0.0833333333 0.1000000000")
    assert (output / "labels/val/val_one.txt").read_text(encoding="utf-8") == ""
    assert (output / "images/train/train_one.png").read_bytes() == (
        dataset / "images/train_one.png"
    ).read_bytes()
    assert (output / "dataset_check/train_train_one.png").is_file()
    assert "one- or two-pixel bbox(es)" in result.stdout


def test_multivalue_mask_requires_explicit_consent(tmp_path: Path) -> None:
    """Проверяет остановку на неоднозначных значениях маски."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    module = load_script(PREPARE_SCRIPT, "prepare_irstd1k_yolo_test")
    dataset = tmp_path / "dataset"
    image = np.zeros((3, 3), dtype=np.uint8)
    mask = image.copy()
    mask[0, 0] = 1
    mask[2, 2] = 255
    write_png(dataset / "images/sample.png", image, cv2)
    write_png(dataset / "masks/sample.png", mask, cv2)
    with pytest.raises(ValueError, match="Suspicious multivalue mask"):
        module.audit_image(dataset, "train", "sample", False)


def test_training_summary_selects_best_map_epoch(tmp_path: Path) -> None:
    """Проверяет выбор лучшей эпохи и Detect-метрик из CSV."""
    module = load_script(TRAIN_SCRIPT, "train_yolo_test")
    results = tmp_path / "results.csv"
    results.write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.2,0.3,0.4,0.1\n"
        "1,0.6,0.7,0.8,0.5\n",
        encoding="utf-8",
    )
    summary = module.read_best_metrics(results)
    assert summary.epochs_completed == 2
    assert summary.best_epoch == 2
    assert summary.precision == pytest.approx(0.6)
    assert summary.map50_95 == pytest.approx(0.5)
