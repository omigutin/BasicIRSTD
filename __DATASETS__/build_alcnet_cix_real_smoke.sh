#!/usr/bin/env bash
set -euo pipefail

# ALCNet real-frame smoke build: ONNX [1,1,640,512] -> CIX.
# Calibration uses one real positive IWT frame only.
# This is still NOT the final production calibration dataset.

SOURCE_DIR="/mnt/w/PycharmProjects/_IWT_/BasicIRSTD/npu_artifacts/alcnet_real_03_097"
SOURCE_ONNX="$SOURCE_DIR/alcnet_irstd1k.onnx"
SOURCE_INPUT="$SOURCE_DIR/input.npy"

WORKSPACE="$HOME/cix/alcnet_irstd1k_real_smoke"

if [[ ! -f "$SOURCE_ONNX" ]]; then
  echo "ERROR: ONNX not found: $SOURCE_ONNX" >&2
  exit 1
fi

if [[ ! -f "$SOURCE_INPUT" ]]; then
  echo "ERROR: input.npy not found: $SOURCE_INPUT" >&2
  exit 1
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate cix-noe

mkdir -p \
  "$WORKSPACE/model" \
  "$WORKSPACE/datasets" \
  "$WORKSPACE/cfg" \
  "$WORKSPACE/out" \
  "$WORKSPACE/logs"

cp -f "$SOURCE_ONNX" "$WORKSPACE/model/alcnet_irstd1k.onnx"
cp -f "$SOURCE_INPUT" "$WORKSPACE/datasets/calibration_real_03_097.npy"

cat > "$WORKSPACE/cfg/alcnet_irstd1k_build.cfg" <<'CFG'
[Common]
mode = build

[Parser]
model_type = ONNX
model_name = alcnet_irstd1k
input_data_format = NCHW
input_model = model/alcnet_irstd1k.onnx
input = input
output = probability_map
input_shape = [1, 1, 640, 512]
output_dir = ./out

[Optimizer]
dataset = numpydataset
calibration_data = datasets/calibration_real_03_097.npy
calibration_batch_size = 1
output_dir = ./out
dump_dir = ./out
quantize_method_for_activation = per_tensor_asymmetric
quantize_method_for_weight = per_channel_symmetric_restricted_range
save_statistic_info = True
cast_dtypes_for_lib = True

[GBuilder]
target = X2_1204MP3
outputs = alcnet_irstd1k_real_smoke.cix
tiling = fps
CFG

cd "$WORKSPACE"

python - <<'PY'
import numpy as np
p = "datasets/calibration_real_03_097.npy"
x = np.load(p)
print("Calibration:", p)
print("shape:", x.shape)
print("dtype:", x.dtype)
print("min/max/mean:", float(x.min()), float(x.max()), float(x.mean()))
if x.shape != (1, 1, 640, 512):
    raise SystemExit(f"Wrong calibration shape: {x.shape}")
if x.dtype != np.float32:
    raise SystemExit(f"Wrong calibration dtype: {x.dtype}")
PY

LOG_FILE="logs/build_$(date +%Y%m%d_%H%M%S).log"

echo
echo "============================================================"
echo "ALCNet real-frame smoke build [1,1,640,512]"
echo "Workspace: $WORKSPACE"
echo "============================================================"

set +e
/usr/bin/time -v \
  cixbuild cfg/alcnet_irstd1k_build.cfg \
  2>&1 | tee "$LOG_FILE"
CIX_EXIT=${PIPESTATUS[0]}
set -e

echo
echo "============================================================"
echo "Build summary"
echo "============================================================"
echo "cixbuild exit code: $CIX_EXIT"
echo "log: $WORKSPACE/$LOG_FILE"

grep -nEi \
  'build success|total errors|error|unsupported|cosine|similarity|MSE|quantize.*accuracy|output tensors' \
  "$LOG_FILE" | tail -n 100 || true

CIX_FILE="$(find "$WORKSPACE" -maxdepth 2 -type f -name 'alcnet_irstd1k_real_smoke.cix' -print -quit)"

if [[ "$CIX_EXIT" -ne 0 ]]; then
  echo "BUILD FAILED"
  exit "$CIX_EXIT"
fi

if [[ -z "$CIX_FILE" ]]; then
  echo "ERROR: .cix not found" >&2
  exit 2
fi

echo
echo "CIX created:"
ls -lh "$CIX_FILE"
sha256sum "$CIX_FILE"

echo
echo "REAL-FRAME SMOKE BUILD OK."
echo "NOTE: calibration contains only one positive frame; final calibration is still required."
