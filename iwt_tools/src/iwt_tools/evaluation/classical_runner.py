"""CPU-адаптер классических методов обнаружения малых ИК-целей."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter

import cv2
import numpy as np


class ClassicalMethod(str, Enum):
    """Перечисляет реализованные классические методы."""

    TOP_HAT = "top_hat"


class StructuringElementShape(str, Enum):
    """Задаёт форму структурного элемента OpenCV."""

    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    CROSS = "cross"


class ThresholdStrategy(str, Enum):
    """Задаёт способ преобразования карты отклика в бинарную маску."""

    FIXED = "fixed"
    OTSU = "otsu"


class BorderType(str, Enum):
    """Задаёт обработку границ изображения морфологической операцией."""

    REFLECT101 = "reflect101"


@dataclass(frozen=True)
class TopHatConfig:
    """Хранит воспроизводимые параметры морфологического Top-Hat."""

    method: ClassicalMethod = ClassicalMethod.TOP_HAT
    kernel_size: int = 9
    kernel_shape: StructuringElementShape = StructuringElementShape.ELLIPSE
    threshold_strategy: ThresholdStrategy = ThresholdStrategy.FIXED
    threshold: float | None = 10.0
    border_type: BorderType = BorderType.REFLECT101
    connectivity: int = 8
    minimum_component_area: int = 1

    def validate(self) -> None:
        """Проверяет параметры без автоматического исправления значений."""

        if self.method is not ClassicalMethod.TOP_HAT:
            raise ValueError(f"Unsupported classical method: {self.method.value}")
        if self.kernel_size <= 1 or self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be an odd integer greater than one")
        if self.threshold_strategy is ThresholdStrategy.FIXED:
            if self.threshold is None:
                raise ValueError("Fixed threshold strategy requires a threshold value")
            if not np.isfinite(self.threshold) or self.threshold < 0:
                raise ValueError("Threshold must be a finite non-negative number")
        elif self.threshold_strategy is ThresholdStrategy.OTSU:
            if self.threshold is not None:
                raise ValueError("Otsu threshold strategy does not accept a threshold value")
        else:
            raise ValueError(
                f"Unsupported threshold strategy: {self.threshold_strategy.value}"
            )
        if self.border_type is not BorderType.REFLECT101:
            raise ValueError(f"Unsupported border type: {self.border_type.value}")
        if self.connectivity != 8:
            raise ValueError("Only 8-connectivity is supported")
        if self.minimum_component_area != 1:
            raise ValueError("Minimum component area must remain one")


@dataclass(frozen=True)
class ClassicalDetection:
    """Описывает одну связную компоненту бинарной маски."""

    index: int
    label: int
    x1: int
    y1: int
    x2: int
    y2: int
    centroid_x: float
    centroid_y: float
    area_pixels: int
    max_response: float


@dataclass(frozen=True)
class ClassicalPredictionResult:
    """Возвращает отклик, маску, компоненты и раздельный timing."""

    response_map: np.ndarray
    binary_mask: np.ndarray
    labels: np.ndarray
    detections: tuple[ClassicalDetection, ...]
    foreground_pixels: int
    applied_threshold: float
    algorithm_ms: float
    processing_ms: float


class TopHatRunner:
    """Выполняет white Top-Hat и извлекает все связные компоненты."""

    _SHAPES = {
        StructuringElementShape.RECTANGLE: cv2.MORPH_RECT,
        StructuringElementShape.ELLIPSE: cv2.MORPH_ELLIPSE,
        StructuringElementShape.CROSS: cv2.MORPH_CROSS,
    }

    def __init__(self, config: TopHatConfig) -> None:
        """Валидирует конфигурацию и один раз создаёт kernel."""

        config.validate()
        self.config = config
        self._kernel = cv2.getStructuringElement(
            self._SHAPES[config.kernel_shape],
            (config.kernel_size, config.kernel_size),
        )

    @staticmethod
    def _prepare_grayscale(image: np.ndarray) -> np.ndarray:
        """Возвращает исходный grayscale без нормализации и смены dtype."""

        if image.ndim == 2:
            grayscale = image
        elif image.ndim == 3 and image.shape[2] == 1:
            grayscale = image[:, :, 0]
        else:
            raise ValueError(f"Expected HxW or HxWx1 grayscale image, got {image.shape}")
        if grayscale.size == 0:
            raise ValueError("Input image must not be empty")
        supported_dtypes = (
            np.dtype(np.uint8),
            np.dtype(np.uint16),
            np.dtype(np.int16),
            np.dtype(np.float32),
            np.dtype(np.float64),
        )
        if grayscale.dtype not in supported_dtypes:
            raise ValueError(f"Unsupported grayscale dtype: {grayscale.dtype}")
        if np.issubdtype(grayscale.dtype, np.floating) and not np.isfinite(grayscale).all():
            raise ValueError("Floating-point image must contain only finite values")
        return np.ascontiguousarray(grayscale)

    def _build_binary_mask(self, response_map: np.ndarray) -> tuple[np.ndarray, float]:
        """Применяет fixed или Otsu без нормализации карты отклика."""

        if self.config.threshold_strategy is ThresholdStrategy.FIXED:
            if self.config.threshold is None:
                raise RuntimeError("Validated fixed threshold is missing")
            return response_map > self.config.threshold, self.config.threshold

        if response_map.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            raise ValueError(
                "OpenCV Otsu requires an unsigned 8-bit or 16-bit response map; "
                f"got {response_map.dtype}. Automatic normalization is disabled."
            )
        applied_threshold, thresholded = cv2.threshold(
            response_map,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        return thresholded > 0, float(applied_threshold)

    def predict(self, image: np.ndarray) -> ClassicalPredictionResult:
        """Строит отклик, бинарную маску и компоненты на CPU."""

        processing_started = perf_counter()
        grayscale = self._prepare_grayscale(image)

        algorithm_started = perf_counter()
        response_map = cv2.morphologyEx(
            grayscale,
            cv2.MORPH_TOPHAT,
            self._kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        algorithm_ms = (perf_counter() - algorithm_started) * 1000.0

        binary_mask, applied_threshold = self._build_binary_mask(response_map)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask.astype(np.uint8), connectivity=self.config.connectivity
        )
        detections: list[ClassicalDetection] = []
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            component_response = response_map[labels == label]
            detections.append(ClassicalDetection(
                index=label,
                label=label,
                x1=x,
                y1=y,
                x2=x + width,
                y2=y + height,
                centroid_x=float(centroids[label, 0]),
                centroid_y=float(centroids[label, 1]),
                area_pixels=area,
                max_response=float(component_response.max()),
            ))
        processing_ms = (perf_counter() - processing_started) * 1000.0
        return ClassicalPredictionResult(
            response_map=response_map,
            binary_mask=binary_mask,
            labels=labels,
            detections=tuple(detections),
            foreground_pixels=int(np.count_nonzero(binary_mask)),
            applied_threshold=applied_threshold,
            algorithm_ms=algorithm_ms,
            processing_ms=processing_ms,
        )
