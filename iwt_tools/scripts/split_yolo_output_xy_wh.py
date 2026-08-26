from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, shape_inference


XY_OUTPUT_NAME = "boxes_xy"
WH_OUTPUT_NAME = "boxes_wh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace final YOLO bbox output with independent XY, WH and "
            "confidence outputs before common bbox quantization."
        )
    )
    parser.add_argument("--input-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument(
        "--evaluation-tensor",
        type=Path,
        help=(
            "Optional eval_images_rgb_u8.npy for bit-identical "
            "ONNX Runtime verification."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def value_shape(
    value_info: onnx.ValueInfoProto,
) -> list[int | str]:
    dimensions: list[int | str] = []

    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        else:
            dimensions.append(dimension.dim_param or "?")

    return dimensions


def collect_values(
    model: onnx.ModelProto,
) -> dict[str, onnx.ValueInfoProto]:
    values = (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    )
    return {value.name: value for value in values}


def producer_by_output(
    model: onnx.ModelProto,
) -> dict[str, onnx.NodeProto]:
    return {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
    }


def get_axis(node: onnx.NodeProto) -> int | None:
    for attribute in node.attribute:
        if attribute.name == "axis":
            return int(helper.get_attribute_value(attribute))
    return None


def create_xy_wh_model(
    input_path: Path,
    output_path: Path,
) -> tuple[
    str,
    str,
    str,
    str,
    list[int | str],
    list[int | str],
]:
    model = onnx.load(input_path)
    inferred = shape_inference.infer_shapes(model)
    values = collect_values(inferred)
    producers = producer_by_output(model)

    if len(model.graph.output) != 1:
        raise RuntimeError(
            f"Expected one graph output, got {len(model.graph.output)}"
        )

    original_output_name = model.graph.output[0].name
    final_concat = producers.get(original_output_name)

    if final_concat is None or final_concat.op_type != "Concat":
        raise RuntimeError(
            "Original output producer must be final Concat"
        )

    if get_axis(final_concat) != 1:
        raise RuntimeError(
            f"Expected final Concat axis=1, got {get_axis(final_concat)}"
        )

    if len(final_concat.input) != 2:
        raise RuntimeError(
            f"Expected two final Concat inputs, got "
            f"{len(final_concat.input)}"
        )

    bbox_output_name, scores_name = final_concat.input
    bbox_mul = producers.get(bbox_output_name)

    if bbox_mul is None or bbox_mul.op_type != "Mul":
        raise RuntimeError(
            f"Bbox producer must be Mul, got "
            f"{None if bbox_mul is None else bbox_mul.op_type}"
        )

    if len(bbox_mul.input) != 2:
        raise RuntimeError(
            f"Expected bbox Mul with two inputs, got "
            f"{len(bbox_mul.input)}"
        )

    bbox_concat_name, stride_name = bbox_mul.input
    bbox_concat = producers.get(bbox_concat_name)

    if bbox_concat is None or bbox_concat.op_type != "Concat":
        raise RuntimeError(
            "Input of bbox Mul must be bbox Concat"
        )

    if get_axis(bbox_concat) != 1:
        raise RuntimeError(
            f"Expected bbox Concat axis=1, got {get_axis(bbox_concat)}"
        )

    if len(bbox_concat.input) != 2:
        raise RuntimeError(
            f"Expected bbox Concat with two inputs, got "
            f"{len(bbox_concat.input)}"
        )

    xy_grid_name, wh_grid_name = bbox_concat.input

    for tensor_name in (
        xy_grid_name,
        wh_grid_name,
        scores_name,
    ):
        if tensor_name not in values:
            raise RuntimeError(
                f"Missing inferred shape for {tensor_name}"
            )

    xy_shape = value_shape(values[xy_grid_name])
    wh_shape = value_shape(values[wh_grid_name])
    scores_shape = value_shape(values[scores_name])

    if len(xy_shape) != 3 or xy_shape[1] != 2:
        raise RuntimeError(
            f"Unexpected XY shape: {xy_shape}"
        )

    if len(wh_shape) != 3 or wh_shape[1] != 2:
        raise RuntimeError(
            f"Unexpected WH shape: {wh_shape}"
        )

    if len(scores_shape) != 3 or scores_shape[1] != 1:
        raise RuntimeError(
            f"Unexpected scores shape: {scores_shape}"
        )

    removed_output_names = {
        original_output_name,
        bbox_output_name,
        bbox_concat_name,
    }

    remaining_nodes = [
        node
        for node in model.graph.node
        if not any(
            output_name in removed_output_names
            for output_name in node.output
        )
    ]

    del model.graph.node[:]
    model.graph.node.extend(remaining_nodes)

    model.graph.node.extend(
        [
            helper.make_node(
                "Mul",
                inputs=[xy_grid_name, stride_name],
                outputs=[XY_OUTPUT_NAME],
                name="DecodeBoxesXY",
            ),
            helper.make_node(
                "Mul",
                inputs=[wh_grid_name, stride_name],
                outputs=[WH_OUTPUT_NAME],
                name="DecodeBoxesWH",
            ),
        ]
    )

    del model.graph.output[:]
    model.graph.output.extend(
        [
            helper.make_tensor_value_info(
                XY_OUTPUT_NAME,
                TensorProto.FLOAT,
                xy_shape,
            ),
            helper.make_tensor_value_info(
                WH_OUTPUT_NAME,
                TensorProto.FLOAT,
                wh_shape,
            ),
            helper.make_tensor_value_info(
                scores_name,
                TensorProto.FLOAT,
                scores_shape,
            ),
        ]
    )

    checker.check_model(model)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    onnx.save(model, output_path)

    return (
        original_output_name,
        xy_grid_name,
        wh_grid_name,
        scores_name,
        xy_shape,
        wh_shape,
        scores_shape,
    )


def verify_equivalence(
    original_model: Path,
    transformed_model: Path,
    tensor_path: Path,
) -> None:
    images_u8 = np.load(
        tensor_path,
        mmap_mode="r",
    )

    original_session = ort.InferenceSession(
        str(original_model),
        providers=["CPUExecutionProvider"],
    )
    transformed_session = ort.InferenceSession(
        str(transformed_model),
        providers=["CPUExecutionProvider"],
    )

    original_input = original_session.get_inputs()[0]
    input_shape = original_input.shape

    if (
        len(input_shape) != 4
        or not isinstance(input_shape[0], int)
    ):
        raise RuntimeError(
            f"Unexpected input shape: {input_shape}"
        )

    batch_size = input_shape[0]
    batch = np.ascontiguousarray(
        images_u8[:batch_size].astype(np.float32)
        / 255.0
    )

    original_output = original_session.run(
        None,
        {original_input.name: batch},
    )[0]

    transformed_input_name = (
        transformed_session.get_inputs()[0].name
    )
    xy, wh, scores = transformed_session.run(
        None,
        {transformed_input_name: batch},
    )

    reconstructed = np.concatenate(
        (xy, wh, scores),
        axis=1,
    )

    max_abs = float(
        np.max(
            np.abs(
                reconstructed - original_output
            )
        )
    )

    if not np.array_equal(
        reconstructed,
        original_output,
    ):
        raise RuntimeError(
            f"Outputs are not bit-identical; max_abs={max_abs}"
        )

    print("Verification:")
    print(
        f"  original shape:      {list(original_output.shape)}"
    )
    print(f"  XY shape:            {list(xy.shape)}")
    print(f"  WH shape:            {list(wh.shape)}")
    print(f"  scores shape:        {list(scores.shape)}")
    print(
        f"  reconstructed shape: {list(reconstructed.shape)}"
    )
    print(f"  max_abs:             {max_abs:.10f}")
    print("  result:              BIT-IDENTICAL")


def main() -> int:
    args = parse_args()

    input_path = args.input_model.expanduser().resolve()
    output_path = args.output_model.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input model not found: {input_path}"
        )

    (
        original_output_name,
        xy_grid_name,
        wh_grid_name,
        scores_name,
        xy_shape,
        wh_shape,
        scores_shape,
    ) = create_xy_wh_model(
        input_path=input_path,
        output_path=output_path,
    )

    print(f"Input model:       {input_path}")
    print(f"Output model:      {output_path}")
    print(f"Removed output:    {original_output_name}")
    print(f"XY source:         {xy_grid_name}")
    print(f"WH source:         {wh_grid_name}")
    print(f"XY output:         {XY_OUTPUT_NAME} {xy_shape}")
    print(f"WH output:         {WH_OUTPUT_NAME} {wh_shape}")
    print(f"Scores output:     {scores_name} {scores_shape}")
    print(f"Output SHA-256:    {sha256(output_path)}")

    if args.evaluation_tensor is not None:
        tensor_path = (
            args.evaluation_tensor.expanduser().resolve()
        )

        if not tensor_path.is_file():
            raise FileNotFoundError(
                f"Evaluation tensor not found: {tensor_path}"
            )

        verify_equivalence(
            original_model=input_path,
            transformed_model=output_path,
            tensor_path=tensor_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
