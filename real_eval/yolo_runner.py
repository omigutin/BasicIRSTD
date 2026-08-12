"""Ленивый адаптер Ultralytics YOLO для объективного benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class YoloModelConfig:
    """Настройки пользовательской Ultralytics YOLO-модели."""

    checkpoint_path: Path
    device: str = "cuda"
    confidence_threshold: float = 0.25
    nms_iou: float = 0.7
    imgsz: int = 640
    class_id: Optional[int] = None

    def validate(self) -> None:
        """Проверяет параметры до загрузки модели."""

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")
        if not self.device.strip():
            raise ValueError("Device must not be empty")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if not 0.0 <= self.nms_iou <= 1.0:
            raise ValueError("NMS IoU must be between 0 and 1")
        if self.imgsz <= 0:
            raise ValueError("Image size must be positive")
        if self.class_id is not None and self.class_id < 0:
            raise ValueError("Class id must not be negative")


@dataclass(frozen=True)
class YoloDetection:
    """Один YOLO bbox в координатах исходного изображения."""

    index: int
    x1: float
    y1: float
    x2: float
    y2: float
    centroid_x: float
    centroid_y: float
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class YoloPredictionResult:
    """YOLO detections и чистое время model inference."""

    detections: tuple[YoloDetection, ...]
    inference_ms: float


class YoloModelRunner:
    """Загружает Ultralytics YOLO один раз и возвращает общие bbox-данные."""

    def __init__(self, config: YoloModelConfig) -> None:
        """Лениво импортирует Ultralytics и загружает checkpoint."""

        config.validate()
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required for YOLO evaluation. "
                "Install it in the active environment."
            ) from error
        self.config = config
        self._model = YOLO(str(config.checkpoint_path))

    @staticmethod
    def _prepare_image(image: np.ndarray) -> np.ndarray:
        """Переводит RGB/RGBA source в ожидаемый Ultralytics BGR без resize."""

        if image.ndim == 2:
            return image
        if image.ndim != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        raise ValueError(f"Unsupported image channel count: {image.shape[2]}")

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        """Возвращает имя класса из mapping или sequence Ultralytics."""

        if isinstance(names, Mapping):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def predict(self, image: np.ndarray) -> YoloPredictionResult:
        """Выполняет detection без искусственной segmentation mask."""

        height, width = image.shape[:2]
        classes = None if self.config.class_id is None else [self.config.class_id]
        results = self._model.predict(
            source=self._prepare_image(image),
            conf=self.config.confidence_threshold,
            iou=self.config.nms_iou,
            imgsz=self.config.imgsz,
            device=self.config.device,
            classes=classes,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError(f"Expected one YOLO result, got {len(results)}")
        result = results[0]
        boxes = result.boxes
        detections: list[YoloDetection] = []
        if boxes is not None:
            coordinates = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy()
            names = getattr(result, "names", getattr(self._model, "names", {}))
            for index, (coordinate, confidence, raw_class_id) in enumerate(
                zip(coordinates, confidences, class_ids), start=1
            ):
                x1 = float(np.clip(coordinate[0], 0, width))
                y1 = float(np.clip(coordinate[1], 0, height))
                x2 = float(np.clip(coordinate[2], 0, width))
                y2 = float(np.clip(coordinate[3], 0, height))
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = int(raw_class_id)
                detections.append(YoloDetection(
                    index=index, x1=x1, y1=y1, x2=x2, y2=y2,
                    centroid_x=(x1 + x2) / 2.0,
                    centroid_y=(y1 + y2) / 2.0,
                    confidence=float(confidence), class_id=class_id,
                    class_name=self._class_name(names, class_id),
                ))
        speed = getattr(result, "speed", {}) or {}
        inference_ms = float(speed.get("inference", 0.0))
        if inference_ms < 0:
            raise RuntimeError("YOLO inference time must not be negative")
        return YoloPredictionResult(tuple(detections), inference_ms)
