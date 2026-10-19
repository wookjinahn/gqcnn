#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_dir="$repo_dir/.ai/local"
env_dir="$local_dir/dexnet3"
model_dir="$local_dir/models/GQCNN-3.0"
model_zip="$local_dir/models/GQCNN-3.0.zip"
model_id="1XT5iy22NWFUGI6_TowOawXQvQQ2w8GjN"
model_sha256="d048afb9974d4c3170ccb89289302ea871b795d421ee3910805675f35da7b1f6"

mkdir -p "$local_dir/conda-pkgs" "$local_dir/cache" "$local_dir/pip-cache" "$local_dir/models"
export CONDA_PKGS_DIRS="$local_dir/conda-pkgs"
export XDG_CACHE_HOME="$local_dir/cache"
export PIP_CACHE_DIR="$local_dir/pip-cache"

if [[ ! -x "$env_dir/bin/python" ]]; then
    conda env create --prefix "$env_dir" --file "$repo_dir/envs/dexnet3.yml"
fi

if [[ ! -f "$model_dir/config.json" ]]; then
    if [[ ! -f "$model_zip" ]]; then
        "$env_dir/bin/gdown" "$model_id" -O "$model_zip"
    fi
    printf '%s  %s\n' "$model_sha256" "$model_zip" | sha256sum -c -
    unzip -q "$model_zip" -d "$local_dir/models"
    mv "$local_dir/models/GQ-Suction" "$model_dir"
fi

echo "Conda environment: $env_dir"
echo "Dex-Net 3.0 model: $model_dir"
