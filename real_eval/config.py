"""Типизированная конфигурация локального inference-запуска."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Настройки модели и соответствующего ей pretrained checkpoint."""

    model_name: str
    checkpoint_path: Path
    train_dataset_name: str
    dataset_dir: Path = Path("datasets")
    threshold: float = 0.5
    device: str = "cuda"

    def validate(self) -> None:
        """Проверяет пользовательские параметры до загрузки тяжёлой модели."""

        if not self.model_name.strip():
            raise ValueError("Model name must not be empty")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")
        if not self.train_dataset_name.strip():
            raise ValueError("Train dataset name must not be empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        if not self.device.strip():
            raise ValueError("Device must not be empty")


@dataclass(frozen=True)
class RunConfig:
    """Настройки входных данных и каталога результатов."""

    input_path: Path
    output_dir: Path

    def validate(self) -> None:
        """Проверяет существование входного пути."""

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input does not exist: {self.input_path}")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {self.output_dir}")
