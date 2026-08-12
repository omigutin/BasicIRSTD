"""Локальный запуск pretrained-моделей BasicIRSTD на реальных данных."""

from .config import ModelConfig, RunConfig
from .model_runner import ModelRunner, PredictionResult
from .sources import FrameData, FrameSource, create_frame_source

__all__ = [
    "FrameData",
    "FrameSource",
    "ModelConfig",
    "ModelRunner",
    "PredictionResult",
    "RunConfig",
    "create_frame_source",
]
