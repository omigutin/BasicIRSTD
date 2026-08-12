"""Потоковые источники изображений и видеокадров без ML-preprocessing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol

import cv2
import numpy as np


IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".avi", ".mp4", ".mkv", ".mov"})


@dataclass(frozen=True)
class FrameData:
    """Исходный кадр и его положение во входном источнике."""

    image: np.ndarray
    index: int
    source_name: str
    timestamp_sec: Optional[float] = None


class FrameSource(Protocol):
    """Общий интерфейс последовательного чтения кадров."""

    is_video: bool
    fps: Optional[float]

    def __iter__(self) -> Iterator[FrameData]:
        """Возвращает кадры по одному, не накапливая их в памяти."""


def _read_image(path: Path) -> np.ndarray:
    """Читает изображение с сохранением исходной глубины интенсивности."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            raise ValueError(f"Unsupported image channel count in {path}: {image.shape[2]}")
    return image


class ImageSource:
    """Источник из одного файла изображения."""

    is_video = False
    fps: Optional[float] = None

    def __init__(self, path: Path) -> None:
        """Сохраняет проверенный путь к изображению."""

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        self._path = path

    def __iter__(self) -> Iterator[FrameData]:
        """Возвращает единственный исходный кадр."""

        yield FrameData(_read_image(self._path), 0, self._path.name)


class ImageDirectorySource:
    """Рекурсивный источник изображений с относительными именами."""

    is_video = False
    fps: Optional[float] = None

    def __init__(self, root: Path) -> None:
        """Находит поддерживаемые файлы в стабильном порядке."""

        self._root = root
        self._paths = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self._paths:
            raise ValueError(f"No supported images found in directory: {root}")

    def __iter__(self) -> Iterator[FrameData]:
        """Читает файлы по одному и сохраняет путь относительно root."""

        for index, path in enumerate(self._paths):
            yield FrameData(_read_image(path), index, path.relative_to(self._root).as_posix())


class VideoSource:
    """Потоковый источник кадров OpenCV-видео."""

    is_video = True

    def __init__(self, path: Path) -> None:
        """Открывает метаданные видео, не загружая кадры в память."""

        self._path = path
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        self.fps: Optional[float] = fps if fps > 0 else None

    def __iter__(self) -> Iterator[FrameData]:
        """Декодирует кадры последовательно и освобождает video handle."""

        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {self._path}")
        index = 0
        try:
            while True:
                ok, bgr_image = capture.read()
                if not ok:
                    break
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                yield FrameData(rgb_image, index, self._path.name, timestamp)
                index += 1
        finally:
            capture.release()


def create_frame_source(path: Path) -> FrameSource:
    """Выбирает источник по типу пути и расширению файла."""

    if path.is_dir():
        return ImageDirectorySource(path)
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return ImageSource(path)
    if suffix in VIDEO_EXTENSIONS:
        return VideoSource(path)
    raise ValueError(f"Unsupported input type: {path}")
