#!/usr/bin/env python3
"""
Экспорт ALCNet / IRSTD-1K в статический ONNX с export-only адаптацией Resize.

Главная идея:
- исходные файлы BasicIRSTD НЕ изменяются;
- reference-модель работает как есть;
- для export-копии временно подменяется torchvision.transforms.Resize
  на F.interpolate(..., bilinear, align_corners=False, antialias=False);
- ДО ONNX обязательно сравниваются:
    original PyTorch ↔ export-friendly PyTorch;
- затем сравниваются:
    export-friendly PyTorch ↔ ONNX Runtime.

Это нужно из-за torchvision Resize, который в текущем окружении
экспортируется как aten::_upsample_bilinear2d_aa и не поддерживается
legacy ONNX exporter для opset 17.

Запускать из корня BasicIRSTD.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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

from net import Net
import importlib

# В BasicIRSTD имя `model.ACM` затенено алиасом класса в model/__init__.py,
# поэтому обычный dotted import может разрешиться не в пакет.
alcnet_module = importlib.import_module("model.ACM.model_ALCnet")


MODEL_NAME = "ALCNet"
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 640
DEFAULT_MEAN = 87.4661865
DEFAULT_STD = 39.7195320
DEFAULT_OPSET = 17
DEFAULT_THRESHOLD = 0.5
DEFAULT_SEED = 42


class ExportResize:
    """
    Drop-in замена torchvision.transforms.Resize для tensor input.

    В ALCNet Resize вызывается как:
        transforms.Resize([height, width])(tensor)

    Здесь сохраняем bilinear resize, но отключаем antialias,
    чтобы получить обычный ONNX Resize вместо
    aten::_upsample_bilinear2d_aa.
    """

    def __init__(self, size: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError(f"Ожидался Resize size из 2 элементов, получено: {size}")
            self.size = (int(size[0]), int(size[1]))

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "ExportResize предназначен только для torch.Tensor внутри ALCNet"
            )

        # Identity resize не создаём вообще.
        if tuple(tensor.shape[-2:]) == self.size:
            return tensor

        return F.interpolate(
            tensor,
            size=self.size,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Экспорт ALCNet / IRSTD-1K в ONNX opset 17."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("log") / "IRSTD-1K" / "ALCNet_400.pth.tar",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iwt_tools") / "models" / "alcnet_irstd1k",
    )
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--input-npy", type=Path, default=None)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--mean", type=float, default=DEFAULT_MEAN)
    parser.add_argument("--std", type=float, default=DEFAULT_STD)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Original ↔ export-friendly проверяем строже/отдельно.
    parser.add_argument("--adapter-atol", type=float, default=5e-4)
    parser.add_argument("--adapter-rtol", type=float, default=5e-3)

    # Export-friendly ↔ ONNX должно совпасть значительно точнее.
    parser.add_argument("--onnx-atol", type=float, default=1e-5)
    parser.add_argument("--onnx-rtol", type=float, default=1e-4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False

    checkpoint = torch.load(str(path), **kwargs)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint должен быть dict с ключом 'state_dict'")
    if not isinstance(checkpoint["state_dict"], dict):
        raise TypeError("checkpoint['state_dict'] должен быть dict")
    return checkpoint


def build_model(checkpoint_path: Path) -> torch.nn.Module:
    model = Net(model_name=MODEL_NAME, mode="test")
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model.cpu()


def normalize(gray: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (gray.astype(np.float32) - np.float32(mean)) / np.float32(std)


def prepare_input(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    expected = (1, 1, args.height, args.width)

    if args.image is not None and args.input_npy is not None:
        raise ValueError("Укажите либо --image, либо --input-npy")

    if args.input_npy is not None:
        path = args.input_npy.resolve()
        arr = np.asarray(np.load(path), dtype=np.float32)
        if tuple(arr.shape) != expected:
            raise ValueError(f"Ожидалось {expected}, получено {arr.shape}")
        return np.ascontiguousarray(arr), f"npy:{path}"

    if args.image is not None:
        path = args.image.resolve()
        with Image.open(path) as img:
            gray = np.asarray(img.convert("I"), dtype=np.float32)

        if tuple(gray.shape) != (args.height, args.width):
            raise ValueError(
                f"Кадр должен быть {args.height}x{args.width}, "
                f"получено {gray.shape}"
            )

        arr = normalize(gray, args.mean, args.std)[None, None, ...]
        return np.ascontiguousarray(arr, dtype=np.float32), f"image:{path}"

    rng = np.random.default_rng(args.seed)
    gray = rng.integers(
        0, 256, size=(args.height, args.width), dtype=np.uint8
    )
    arr = normalize(gray, args.mean, args.std)[None, None, ...]
    return np.ascontiguousarray(arr, dtype=np.float32), f"synthetic:seed={args.seed}"


def run_torch(model: torch.nn.Module, arr: np.ndarray) -> tuple[np.ndarray, float]:
    x = torch.from_numpy(arr)
    started = time.perf_counter()
    with torch.inference_mode():
        y = model(x)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"Ожидался Tensor, получено {type(y).__name__}")
    return (
        np.ascontiguousarray(y.detach().cpu().numpy().astype(np.float32, copy=False)),
        elapsed_ms,
    )


def compare(
    ref: np.ndarray,
    got: np.ndarray,
    *,
    threshold: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if ref.shape != got.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {got.shape}")

    diff = np.abs(ref - got)
    ref_mask = ref > threshold
    got_mask = got > threshold
    union = int(np.logical_or(ref_mask, got_mask).sum())
    intersection = int(np.logical_and(ref_mask, got_mask).sum())
    flips = np.logical_xor(ref_mask, got_mask)

    return {
        "allclose": bool(np.allclose(ref, got, atol=atol, rtol=rtol)),
        "atol": float(atol),
        "rtol": float(rtol),
        "mae": float(diff.mean()),
        "max_abs_error": float(diff.max()),
        "rmse": float(np.sqrt(np.mean(np.square(ref - got)))),
        "mask_iou": 1.0 if union == 0 else float(intersection / union),
        "threshold_flip_pixels": int(flips.sum()),
        "ref": {
            "min": float(ref.min()),
            "max": float(ref.max()),
            "mean": float(ref.mean()),
            "foreground_pixels": int(ref_mask.sum()),
        },
        "got": {
            "min": float(got.min()),
            "max": float(got.max()),
            "mean": float(got.mean()),
            "foreground_pixels": int(got_mask.sum()),
        },
    }


def make_export_friendly_model(reference_model: torch.nn.Module) -> torch.nn.Module:
    # Deepcopy гарантирует, что reference и export-модель независимы.
    export_model = copy.deepcopy(reference_model)
    export_model.eval()
    return export_model


def run_export_friendly(
    model: torch.nn.Module,
    arr: np.ndarray,
) -> tuple[np.ndarray, float]:
    original_resize = alcnet_module.transforms.Resize
    alcnet_module.transforms.Resize = ExportResize
    try:
        return run_torch(model, arr)
    finally:
        alcnet_module.transforms.Resize = original_resize


def export_onnx(
    model: torch.nn.Module,
    arr: np.ndarray,
    path: Path,
    *,
    opset: int,
) -> None:
    x = torch.from_numpy(arr)
    original_resize = alcnet_module.transforms.Resize
    alcnet_module.transforms.Resize = ExportResize
    try:
        torch.onnx.export(
            model,
            x,
            str(path),
            opset_version=opset,
            input_names=["input"],
            output_names=["probability_map"],
            do_constant_folding=True,
            dynamo=False,
        )
    finally:
        alcnet_module.transforms.Resize = original_resize


def validate_onnx(path: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    counts = dict(sorted(Counter(n.op_type for n in model.graph.node).items()))
    opsets = [
        {"domain": item.domain or "ai.onnx", "version": int(item.version)}
        for item in model.opset_import
    ]
    return counts, opsets


def run_ort(
    path: Path,
    arr: np.ndarray,
) -> tuple[np.ndarray, float, str, str]:
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"Ожидался 1 input/1 output, получено {len(inputs)}/{len(outputs)}"
        )

    started = time.perf_counter()
    y = session.run([outputs[0].name], {inputs[0].name: arr})[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return (
        np.ascontiguousarray(np.asarray(y, dtype=np.float32)),
        elapsed_ms,
        inputs[0].name,
        outputs[0].name,
    )


def versions() -> dict[str, Any]:
    try:
        import torchvision
        tv = torchvision.__version__
    except Exception:
        tv = None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": tv,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    }


def main() -> int:
    args = parse_args()

    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    onnx_path = output_dir / "alcnet_irstd1k.onnx"
    input_path = output_dir / "input.npy"
    original_path = output_dir / "pytorch_original_output.npy"
    export_torch_path = output_dir / "pytorch_export_friendly_output.npy"
    onnx_output_path = output_dir / "onnx_output.npy"
    report_path = output_dir / "export_report.json"

    print("=" * 76)
    print("ALCNet / IRSTD-1K: export-friendly PyTorch -> static ONNX")
    print("=" * 76)
    print(f"Checkpoint : {checkpoint}")
    print(f"Output dir : {output_dir}")
    print(f"Input      : [1, 1, {args.height}, {args.width}]")
    print(f"ONNX opset : {args.opset}")

    print("\n[1/7] Загружаем reference ALCNet...")
    reference_model = build_model(checkpoint)
    export_model = make_export_friendly_model(reference_model)

    print("[2/7] Готовим input...")
    arr, input_source = prepare_input(args)
    np.save(input_path, arr)
    print(
        f"       {input_source}; min={arr.min():.6f}; "
        f"max={arr.max():.6f}; mean={arr.mean():.6f}"
    )

    print("[3/7] Original PyTorch inference...")
    original, original_ms = run_torch(reference_model, arr)
    np.save(original_path, original)
    print(
        f"       min={original.min():.8f}; max={original.max():.8f}; "
        f"mean={original.mean():.8f}; {original_ms:.2f} ms"
    )

    print("[4/7] Export-friendly PyTorch inference...")
    export_torch, export_torch_ms = run_export_friendly(export_model, arr)
    np.save(export_torch_path, export_torch)

    adapter_cmp = compare(
        original,
        export_torch,
        threshold=args.threshold,
        atol=args.adapter_atol,
        rtol=args.adapter_rtol,
    )
    print(f"       MAE           : {adapter_cmp['mae']:.10g}")
    print(f"       Max abs error : {adapter_cmp['max_abs_error']:.10g}")
    print(f"       Mask IoU @0.5 : {adapter_cmp['mask_iou']:.10f}")
    print(f"       Threshold flips: {adapter_cmp['threshold_flip_pixels']}")

    if not adapter_cmp["allclose"]:
        print(
            "\nSTOP: export-friendly PyTorch заметно отличается от original."
        )
        print(
            "ONNX не экспортируем. Нужен реальный кадр и отдельный анализ Resize."
        )
        return 3

    print("[5/7] Экспорт ONNX opset 17...")
    export_onnx(export_model, arr, onnx_path, opset=args.opset)
    print(f"       Создан: {onnx_path}")

    print("[6/7] ONNX checker + operator inventory...")
    operator_counts, opsets = validate_onnx(onnx_path)
    print("       ONNX checker: OK")
    for name, count in operator_counts.items():
        print(f"         {name}: {count}")

    print("[7/7] ONNX Runtime inference...")
    onnx_output, onnx_ms, input_name, output_name = run_ort(onnx_path, arr)
    np.save(onnx_output_path, onnx_output)

    onnx_cmp = compare(
        export_torch,
        onnx_output,
        threshold=args.threshold,
        atol=args.onnx_atol,
        rtol=args.onnx_rtol,
    )

    print(f"       MAE           : {onnx_cmp['mae']:.10g}")
    print(f"       Max abs error : {onnx_cmp['max_abs_error']:.10g}")
    print(f"       Mask IoU @0.5 : {onnx_cmp['mask_iou']:.10f}")
    print(f"       Threshold flips: {onnx_cmp['threshold_flip_pixels']}")

    report = {
        "status": "ok" if adapter_cmp["allclose"] and onnx_cmp["allclose"] else "mismatch",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "input": {
            "source": input_source,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "mean": float(args.mean),
            "std": float(args.std),
            "npy": str(input_path),
        },
        "export_adapter": {
            "reason": "torchvision Resize antialias exports as aten::_upsample_bilinear2d_aa",
            "implementation": "F.interpolate bilinear align_corners=False antialias=False",
            "original_vs_export_friendly": adapter_cmp,
        },
        "onnx": {
            "path": str(onnx_path),
            "sha256": sha256_file(onnx_path),
            "opset_requested": int(args.opset),
            "opset_imports": opsets,
            "input_name": input_name,
            "output_name": output_name,
            "operator_counts": operator_counts,
            "export_friendly_vs_onnx": onnx_cmp,
        },
        "timing_ms": {
            "original_pytorch": float(original_ms),
            "export_friendly_pytorch": float(export_torch_ms),
            "onnxruntime": float(onnx_ms),
        },
        "versions": versions(),
        "artifacts": {
            "input": str(input_path),
            "pytorch_original_output": str(original_path),
            "pytorch_export_friendly_output": str(export_torch_path),
            "onnx_output": str(onnx_output_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("РЕЗУЛЬТАТ")
    print("=" * 76)
    print(f"ONNX       : {onnx_path}")
    print(f"Input name : {input_name}")
    print(f"Output name: {output_name}")
    print(f"Report     : {report_path}")

    if onnx_cmp["allclose"]:
        print("\nOK: ONNX готов к следующему этапу cixbuild.")
        return 0

    print("\nSTOP: ONNX отличается от export-friendly PyTorch.")
    return 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print("\nОШИБКА:", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
