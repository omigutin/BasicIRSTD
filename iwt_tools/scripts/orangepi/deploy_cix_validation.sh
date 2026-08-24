#!/usr/bin/env bash
# Копирует автономный CIX validation runner и его входные артефакты на Orange Pi.

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <ssh-host> <remote-dir> <model.cix> <input.npy>" >&2
    exit 2
fi

ssh_host=$1
remote_dir=$2
model_path=$3
input_path=$4
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

[[ -f "$model_path" ]] || { echo "Model file not found: $model_path" >&2; exit 1; }
[[ -f "$input_path" ]] || { echo "Input file not found: $input_path" >&2; exit 1; }

ssh "$ssh_host" mkdir -p -- "$remote_dir"
scp "$script_dir/run_cix_validation.py" "$model_path" "$input_path" \
    "$ssh_host:$remote_dir/"
