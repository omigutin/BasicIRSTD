"""Сохранение масок и понятных диагностических overlay."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .model_runner import PredictionResult


BoundingBox = Tuple[int, int, int, int]
BBOX_PADDING = 4


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
    bbox_padding: int = BBOX_PADDING,
) -> np.ndarray:
    """Подсвечивает маску, компоненты и основные параметры запуска."""

    if bbox_padding < 0:
        raise ValueError("Bounding box padding must not be negative")

    overlay = to_display_image(image)
    tinted = overlay.copy()
    tinted[result.binary_mask] = (255, 0, 0)
    overlay = cv2.addWeighted(overlay, 0.7, tinted, 0.3, 0.0)
    image_height, image_width = result.binary_mask.shape
    for component_index, (x, y, width, height) in enumerate(
        find_components(result.binary_mask), start=1
    ):
        x1 = max(0, x - bbox_padding)
        y1 = max(0, y - bbox_padding)
        x2 = min(image_width - 1, x + width - 1 + bbox_padding)
        y2 = min(image_height - 1, y + height - 1 + bbox_padding)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Подпись размещается у рамки, а не поверх маленькой ИК-цели.
        label_y = y1 - 3 if y1 >= 10 else min(image_height - 2, y1 + 9)
        cv2.putText(
            overlay,
            f"#{component_index}",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
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
    """Сохраняет одну диагностическую PNG с относительной структурой."""

    output_path = (output_root / relative_path).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = create_overlay(image, result, model_name, threshold)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Cannot write diagnostic image: {output_path}")
