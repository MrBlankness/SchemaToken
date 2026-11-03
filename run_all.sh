#!/usr/bin/env bash
# Runs the full SchemaToken pipeline end-to-end with the small debug budgets
# defined in config.yaml. All stages share the same conda environment and
# config; environment variables override which GPU is used.
set -euo pipefail

cd "$(dirname "$0")"

GPU=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES=$GPU

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nl2sql_bugs

echo "==============================="
echo "Step 1/4 - data pipeline"
echo "==============================="
python data_pipeline.py

echo
echo "==============================="
echo "Step 2/4 - stage 1 (Semantic Alignment SFT)"
echo "==============================="
python stage1_train.py

echo
echo "==============================="
echo "Step 3/4 - stage 2 (Pseudo-task GRPO)"
echo "==============================="
python stage2_train.py

echo
echo "==============================="
echo "Step 4/4 - evaluation (BIRD-dev, schema linking)"
echo "==============================="
python evaluate.py --stage stage2

echo
echo "Done. Artefacts in /apdcephfs_nj7/share_303382070/weibinliao/SchemaToken/runs/"
