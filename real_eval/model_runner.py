"""Адаптер штатной модели и preprocessing BasicIRSTD."""

from dataclasses import dataclass
from time import perf_counter
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from net import Net
from utils import Normalized, PadImg, get_img_norm_cfg

from .config import ModelConfig


@dataclass(frozen=True)
class PredictionResult:
    """Сырая карта модели и простая статистика smoke-test."""

    probability_map: np.ndarray
    binary_mask: np.ndarray
    max_score: float
    target_found: bool
    foreground_pixels: int
    component_count: int
    inference_ms: float


class ModelRunner:
    """Загружает одну модель и выполняет совместимый штатный inference."""

    def __init__(self, config: ModelConfig) -> None:
        """Валидирует device, normalization и формат checkpoint."""

        config.validate()
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self._norm_cfg = get_img_norm_cfg(
            config.train_dataset_name,
            str(config.dataset_dir),
        )
        self._model = Net(model_name=config.model_name, mode="test").to(self.device)
        checkpoint = torch.load(str(config.checkpoint_path), map_location=self.device)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Checkpoint must be a mapping containing 'state_dict'")
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint 'state_dict' must be a mapping")
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()

    @staticmethod
    def _to_pil_integer(image: np.ndarray) -> Image.Image:
        """Приводит вход к эквиваленту штатного PIL ``convert('I')``."""

        if image.ndim == 2:
            pil_image = Image.fromarray(image)
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            pil_image = Image.fromarray(image, mode="RGB" if image.shape[2] == 3 else "RGBA")
        else:
            raise ValueError(f"Expected HxW, HxWx3 or HxWx4 image, got {image.shape}")
        return pil_image.convert("I")

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Повторяет Normalized/PadImg и формирует tensor [1, 1, H, W]."""

        grayscale = np.asarray(self._to_pil_integer(image), dtype=np.float32)
        height, width = grayscale.shape
        normalized = Normalized(grayscale, self._norm_cfg)
        padded = PadImg(normalized)
        tensor = torch.from_numpy(
            np.ascontiguousarray(padded[np.newaxis, np.newaxis, ...])
        ).to(self.device)
        return tensor, (height, width)

    def run_preprocessed(self, tensor: torch.Tensor) -> torch.Tensor:
        """Выполняет только inference для готового tensor и сохраняет padded output."""

        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 1:
            raise ValueError(
                "Preprocessed input must have shape [1, 1, H, W], "
                f"got {tuple(tensor.shape)}"
            )
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            prediction = self._model(tensor)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Model output must be a torch.Tensor")
        if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[1] != 1:
            raise ValueError(
                "Model output must have shape [1, 1, H, W], "
                f"got {tuple(prediction.shape)}"
            )
        return prediction

    def predict(self, image: np.ndarray) -> PredictionResult:
        """Выполняет inference без дополнительного sigmoid и считает метрики."""

        tensor, (height, width) = self.preprocess(image)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        prediction = self.run_preprocessed(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (perf_counter() - started) * 1000.0

        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Model output must be a torch.Tensor")
        if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[1] != 1:
            raise ValueError(
                "Model output must have shape [1, 1, H, W], "
                f"got {tuple(prediction.shape)}"
            )
        if prediction.shape[2] < height or prediction.shape[3] < width:
            raise ValueError("Model output is smaller than the original image")

        probability_map = prediction[0, 0, :height, :width].detach().cpu().numpy().copy()
        binary_mask = probability_map > self.config.threshold
        foreground_pixels = int(np.count_nonzero(binary_mask))
        components, _ = cv2.connectedComponents(binary_mask.astype(np.uint8), connectivity=8)
        return PredictionResult(
            probability_map=probability_map,
            binary_mask=binary_mask,
            max_score=float(probability_map.max()),
            target_found=foreground_pixels > 0,
            foreground_pixels=foreground_pixels,
            component_count=components - 1,
            inference_ms=inference_ms,
        )
