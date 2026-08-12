"""Локальные утилиты BasicIRSTD с ленивым inference API."""

from importlib import import_module
from typing import Any

from .config import ModelConfig, RunConfig

__all__ = [
    "FrameData",
    "FrameSource",
    "ModelConfig",
    "ModelRunner",
    "PredictionResult",
    "RunConfig",
    "create_frame_source",
]


def __getattr__(name: str) -> Any:
    """Лениво импортирует inference API только при явном обращении."""

    modules = {
        "FrameData": ".sources",
        "FrameSource": ".sources",
        "create_frame_source": ".sources",
        "ModelRunner": ".model_runner",
        "PredictionResult": ".model_runner",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
