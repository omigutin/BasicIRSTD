"""Сохранение масок и понятных диагностических overlay."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .model_runner import PredictionResult


BoundingBox = Tuple[int, int, int, int]


def to_display_image(image: np.ndarray) -> np.ndarray:
    """Создаёт RGB uint8-копию только для визуализации, не для модели."""

    if image.ndim == 3:
        rgb = image[..., :3]
    elif image.ndim == 2:
        minimum = float(image.min())
        maximum = float(image.max())
        if maximum > minimum:
            gray = ((image.astype(np.float32) - minimum) * 255.0 / (maximum - minimum)).astype(np.uint8)
        else:
            gray = np.zeros(image.shape, dtype=np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image shape for visualization: {image.shape}")
    if rgb.dtype != np.uint8:
        rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def find_components(binary_mask: np.ndarray) -> List[BoundingBox]:
    """Возвращает прямоугольники всех компонентов без фильтрации размера."""

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    return [tuple(int(value) for value in stats[index, :4]) for index in range(1, count)]


def create_overlay(
    image: np.ndarray,
    result: PredictionResult,
    model_name: str,
    threshold: float,
) -> np.ndarray:
    """Подсвечивает маску, компоненты и основные параметры запуска."""

    overlay = to_display_image(image)
    tinted = overlay.copy()
    tinted[result.binary_mask] = (255, 0, 0)
    overlay = cv2.addWeighted(overlay, 0.7, tinted, 0.3, 0.0)
    for x, y, width, height in find_components(result.binary_mask):
        cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (0, 255, 0), 1)
    label = f"{model_name}  thr={threshold:.3f}  max={result.max_score:.4f}"
    cv2.putText(overlay, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(overlay, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return overlay


def save_frame_artifacts(
    output_root: Path,
    relative_path: Path,
    image: np.ndarray,
    result: PredictionResult,
    model_name: str,
    threshold: float,
) -> None:
    """Сохраняет маску и overlay с относительной структурой источника."""

    mask_path = (output_root / "masks" / relative_path).with_suffix(".png")
    overlay_path = (output_root / "overlays" / relative_path).with_suffix(".png")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    mask = result.binary_mask.astype(np.uint8) * 255
    overlay = create_overlay(image, result, model_name, threshold)
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Cannot write mask: {mask_path}")
    if not cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Cannot write overlay: {overlay_path}")
