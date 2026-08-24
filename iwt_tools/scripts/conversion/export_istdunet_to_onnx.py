#!/usr/bin/env python3
"""
Экспорт ISTDU-Net / IRSTD-1K в статический ONNX и проверка PyTorch ↔ ONNX Runtime.

Что делает скрипт:
1. Загружает ISTDU-Net через штатный iwt_tools.evaluation.ModelRunner.
2. Использует штатный preprocessing BasicIRSTD.
3. Проверяет реальный вход [1, 1, 640, 512].
4. Запускает PyTorch reference inference.
5. Экспортирует static ONNX opset 17 без dynamic axes и без export-adapter.
6. Проверяет ONNX через onnx.checker.
7. Запускает ONNX Runtime.
8. Сравнивает PyTorch ↔ ONNX:
   - MAE;
   - RMSE;
   - max absolute error;
   - Mask IoU @ threshold 0.5;
   - threshold flips;
   - foreground pixels;
   - connected components.
9. Сохраняет:
   - istdunet_irstd1k.onnx;
   - export_report.json.

Запускать из корня BasicIRSTD после:
    python -m pip install -e ./iwt_tools
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import onnx
except ImportError as exc:
    raise SystemExit(
        "Не установлен пакет 'onnx'. Установите:\n"
        "  python -m pip install onnx onnxruntime"
    ) from exc

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit(
        "Не установлен пакет 'onnxruntime'. Установите:\n"
        "  python -m pip install onnx onnxruntime"
    ) from exc

try:
    from iwt_tools.evaluation import ModelConfig, ModelRunner
except ImportError as exc:
    raise SystemExit(
        "Не удалось импортировать iwt_tools.\n"
        "Из корня BasicIRSTD выполните:\n"
        "  python -m pip install -e ./iwt_tools"
    ) from exc


MODEL_NAME = "ISTDU-Net"
TRAIN_DATASET_NAME = "IRSTD-1K"

INPUT_HEIGHT = 640
INPUT_WIDTH = 512
EXPECTED_SHAPE = (1, 1, INPUT_HEIGHT, INPUT_WIDTH)

DEFAULT_OPSET = 17
DEFAULT_THRESHOLD = 0.5
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4


def project_root() -> Path:
    """Возвращает корень BasicIRSTD для iwt_tools/scripts/conversion/*.py."""
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description="Экспорт ISTDU-Net / IRSTD-1K в static ONNX opset 17."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "log" / "IRSTD-1K" / "ISTDU-Net_400.pth.tar",
        help="PyTorch checkpoint ISTDU-Net.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=(
            root
            / "__DATASETS__"
            / "iwt_device_all_bpla_RAW"
            / "fpv"
            / "03_097_az_283.310_el_3.500.png"
        ),
        help="Реальный positive-кадр для smoke-проверки.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "iwt_tools" / "models" / "istdunet_irstd1k",
        help="Каталог ONNX-артефактов.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=root / "datasets",
        help="Upstream datasets BasicIRSTD для normalization config.",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Тестовый кадр не найден: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV не смог прочитать изображение: {path}")

    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    return image


def count_components(mask_4d: np.ndarray) -> int:
    if mask_4d.ndim != 4 or tuple(mask_4d.shape[:2]) != (1, 1):
        raise ValueError(f"Ожидалась маска [1,1,H,W], получено {mask_4d.shape}")

    mask = np.ascontiguousarray(mask_4d[0, 0].astype(np.uint8))
    count, _ = cv2.connectedComponents(mask, connectivity=8)
    return int(count - 1)


def output_stats(output: np.ndarray, threshold: float) -> dict[str, Any]:
    if output.ndim != 4 or tuple(output.shape[:2]) != (1, 1):
        raise ValueError(f"Ожидался output [1,1,H,W], получено {output.shape}")

    mask = output > threshold
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "min": float(output.min()),
        "max": float(output.max()),
        "mean": float(output.mean()),
        "foreground_pixels": int(mask.sum()),
        "connected_components": count_components(mask),
    }


def run_pytorch(
    runner: ModelRunner,
    tensor: torch.Tensor,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    output = runner.run_preprocessed(tensor)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    array = np.ascontiguousarray(
        output.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    return array, elapsed_ms


def export_onnx(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    model.eval()

    export_kwargs: dict[str, Any] = {
        "opset_version": opset,
        "input_names": ["input"],
        "output_names": ["probability_map"],
        "do_constant_folding": True,
    }

    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        tensor,
        str(output_path),
        **export_kwargs,
    )


def validate_onnx(path: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    operator_counts = dict(
        sorted(Counter(node.op_type for node in model.graph.node).items())
    )
    opsets = [
        {
            "domain": item.domain or "ai.onnx",
            "version": int(item.version),
        }
        for item in model.opset_import
    ]
    return operator_counts, opsets


def run_onnx(
    path: Path,
    input_array: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"Ожидался ONNX с 1 input / 1 output, получено "
            f"{len(inputs)} / {len(outputs)}"
        )

    input_meta = inputs[0]
    output_meta = outputs[0]

    started = time.perf_counter()
    output = session.run(
        [output_meta.name],
        {input_meta.name: input_array},
    )[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    metadata = {
        "input_name": input_meta.name,
        "input_shape": list(input_meta.shape),
        "input_type": input_meta.type,
        "output_name": output_meta.name,
        "output_shape": list(output_meta.shape),
        "output_type": output_meta.type,
    }

    return (
        np.ascontiguousarray(np.asarray(output, dtype=np.float32)),
        elapsed_ms,
        metadata,
    )


def compare_outputs(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    threshold: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"PyTorch / ONNX shape mismatch: "
            f"{reference.shape} vs {candidate.shape}"
        )

    diff = np.abs(reference - candidate)
    reference_mask = reference > threshold
    candidate_mask = candidate > threshold

    intersection = int(np.logical_and(reference_mask, candidate_mask).sum())
    union = int(np.logical_or(reference_mask, candidate_mask).sum())
    threshold_flips = int(
        np.logical_xor(reference_mask, candidate_mask).sum()
    )

    return {
        "allclose": bool(
            np.allclose(reference, candidate, atol=atol, rtol=rtol)
        ),
        "atol": float(atol),
        "rtol": float(rtol),
        "mae": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(reference - candidate)))),
        "max_abs_error": float(diff.max()),
        "mask_iou": 1.0 if union == 0 else float(intersection / union),
        "threshold_flip_pixels": threshold_flips,
        "pytorch": output_stats(reference, threshold),
        "onnx": output_stats(candidate, threshold),
    }


def package_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "onnx": str(onnx.__version__),
        "onnxruntime": str(ort.__version__),
        "opencv": str(cv2.__version__),
        "numpy": str(np.__version__),
    }


def print_stats(title: str, stats: dict[str, Any], elapsed_ms: float) -> None:
    print(f"\n{title}")
    print(f"  shape:                {stats['shape']}")
    print(f"  dtype:                {stats['dtype']}")
    print(f"  min:                  {stats['min']:.9f}")
    print(f"  max:                  {stats['max']:.9f}")
    print(f"  mean:                 {stats['mean']:.9f}")
    print(f"  foreground pixels:    {stats['foreground_pixels']}")
    print(f"  connected components: {stats['connected_components']}")
    print(f"  inference:            {elapsed_ms:.3f} ms")


def main() -> int:
    args = parse_args()

    checkpoint_path = args.checkpoint.resolve()
    image_path = args.image.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint не найден: {checkpoint_path}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"datasets не найден: {dataset_dir}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold должен быть в диапазоне [0, 1]")

    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "istdunet_irstd1k.onnx"
    report_path = output_dir / "export_report.json"

    print("=" * 78)
    print("ISTDU-Net / IRSTD-1K: PyTorch -> static ONNX")
    print("=" * 78)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Image      : {image_path}")
    print(f"Output dir : {output_dir}")
    print(f"Expected   : {EXPECTED_SHAPE}")
    print(f"ONNX opset : {args.opset}")

    config = ModelConfig(
        model_name=MODEL_NAME,
        checkpoint_path=checkpoint_path,
        train_dataset_name=TRAIN_DATASET_NAME,
        dataset_dir=dataset_dir,
        threshold=args.threshold,
        device="cpu",
    )

    print("\n[1/6] Загружаем ISTDU-Net через ModelRunner...")
    runner = ModelRunner(config)

    print("[2/6] Штатный preprocessing реального кадра...")
    image = load_image(image_path)
    tensor, original_hw = runner.preprocess(image)

    print(f"       source HxW : {original_hw}")
    print(f"       tensor     : {tuple(tensor.shape)}, {tensor.dtype}")
    print(
        f"       input      : min={tensor.min().item():.6f}; "
        f"max={tensor.max().item():.6f}; mean={tensor.mean().item():.6f}"
    )

    if tuple(original_hw) != (INPUT_HEIGHT, INPUT_WIDTH):
        raise RuntimeError(
            f"Ожидался исходный кадр {(INPUT_HEIGHT, INPUT_WIDTH)}, "
            f"получено {original_hw}"
        )
    if tuple(tensor.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Ожидался tensor {EXPECTED_SHAPE}, "
            f"получено {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.float32:
        raise RuntimeError(f"Ожидался torch.float32, получено {tensor.dtype}")

    print("[3/6] PyTorch reference inference...")
    pytorch_output, pytorch_ms = run_pytorch(runner, tensor)

    if pytorch_output.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Ожидался PyTorch output {EXPECTED_SHAPE}, "
            f"получено {pytorch_output.shape}"
        )

    pytorch_stats = output_stats(pytorch_output, args.threshold)
    print_stats("PyTorch", pytorch_stats, pytorch_ms)

    if not hasattr(runner, "model"):
        raise RuntimeError(
            "В ModelRunner отсутствует публичное свойство 'model'.\n"
            "Добавьте read-only property, показанный в инструкции."
        )

    print("\n[4/6] Экспортируем static ONNX...")
    export_onnx(
        runner.model,
        tensor,
        onnx_path,
        args.opset,
    )

    operator_counts, opsets = validate_onnx(onnx_path)
    print(f"       ONNX        : {onnx_path}")
    print("       onnx.checker: OK")
    print("       operators:")
    for name, count in operator_counts.items():
        print(f"         {name:<24} {count}")

    print("\n[5/6] ONNX Runtime inference...")
    input_array = np.ascontiguousarray(
        tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    onnx_output, onnx_ms, onnx_metadata = run_onnx(
        onnx_path,
        input_array,
    )

    if onnx_output.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Ожидался ONNX output {EXPECTED_SHAPE}, "
            f"получено {onnx_output.shape}"
        )

    onnx_stats = output_stats(onnx_output, args.threshold)
    print_stats("ONNX Runtime", onnx_stats, onnx_ms)

    print("\n[6/6] Сравниваем PyTorch ↔ ONNX...")
    comparison = compare_outputs(
        pytorch_output,
        onnx_output,
        threshold=args.threshold,
        atol=args.atol,
        rtol=args.rtol,
    )

    print(f"  allclose:            {comparison['allclose']}")
    print(f"  MAE:                 {comparison['mae']:.12f}")
    print(f"  RMSE:                {comparison['rmse']:.12f}")
    print(f"  max abs error:       {comparison['max_abs_error']:.12f}")
    print(f"  Mask IoU @ 0.5:     {comparison['mask_iou']:.9f}")
    print(
        f"  threshold flips:     "
        f"{comparison['threshold_flip_pixels']}"
    )
    print(
        "  foreground pixels:  "
        f"{comparison['pytorch']['foreground_pixels']} -> "
        f"{comparison['onnx']['foreground_pixels']}"
    )
    print(
        "  components:         "
        f"{comparison['pytorch']['connected_components']} -> "
        f"{comparison['onnx']['connected_components']}"
    )

    report = {
        "model": {
            "name": MODEL_NAME,
            "train_dataset": TRAIN_DATASET_NAME,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "input": {
            "source_image": str(image_path),
            "source_hw": list(original_hw),
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "mean": float(tensor.mean().item()),
        },
        "export": {
            "onnx_path": str(onnx_path),
            "onnx_sha256": sha256_file(onnx_path),
            "opset_requested": int(args.opset),
            "opsets": opsets,
            "dynamic_axes": False,
            "operators": operator_counts,
        },
        "onnx_metadata": onnx_metadata,
        "timing_ms": {
            "pytorch_cpu": float(pytorch_ms),
            "onnxruntime_cpu": float(onnx_ms),
        },
        "comparison": comparison,
        "versions": package_versions(),
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport: {report_path}")

    success = (
        comparison["allclose"]
        and comparison["mask_iou"] == 1.0
        and comparison["threshold_flip_pixels"] == 0
    )

    if success:
        print("\nRESULT: OK — PyTorch и ONNX практически идентичны.")
        return 0

    print(
        "\nRESULT: CHECK — ONNX создан, но сравнение требует анализа. "
        "Не переходите к CixBuilder."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
