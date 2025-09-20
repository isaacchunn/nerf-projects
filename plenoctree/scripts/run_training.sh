#!/usr/bin/env bash
# set -euo pipefail

# EDIT THESE PATHS
DATA_ROOT="/mnt/d/GitHub/plenoctree/data/NeRF/nerf_synthetic"  # e.g., /mnt/d/data/NeRF/nerf_synthetic
CKPT_ROOT="/mnt/d/GitHub/plenoctree/data/Plenoctree/checkpoints/syn_sh16"
SCENE="chair"
CONFIG_FILE="nerf_sh/config/blender"            # no .yaml
EXTRA_FLAGS=""                                  # e.g., for mic: "--lr_delay_steps 50000 --lr_delay_mult 0.01"

# Note: Make sure you have the appropriate conda environment activated before running this script
# Use 'plenoctree' environment for training and 'plenoctree_eval' environment for evaluation

python -m nerf_sh.train \
  --train_dir "${CKPT_ROOT}/${SCENE}" \
  --config "${CONFIG_FILE}" \
  --data_dir "${DATA_ROOT}/${SCENE}" \
  ${EXTRA_FLAGS}

echo "Training completed. Please switch to 'plenoctree_eval' environment for evaluation."
echo "Then run:"
echo "python -m nerf_sh.eval --chunk 4096 --train_dir '${CKPT_ROOT}/${SCENE}' --config '${CONFIG_FILE}' --data_dir '${DATA_ROOT}/${SCENE}'"


