#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
CKPT="data/outputs/2026.01.26/23.10.30_train_diffusion_transformer_vla_tokens_vla_nav_224/checkpoints/epoch=0030-val_loss=0.0202.ckpt"
BASE_OUT="/media/dragon_llm/linux_ssd/RSS_2026_exp/benckmark/k_exp_"
NUM_INFER_STEPS=5
BATCH_SIZE=128
SEED=42

# Sweep k
KS=(1 2 4 5 10 20 50)

# -----------------------------
# Run
# -----------------------------
mkdir -p "${BASE_OUT}"

echo "[INFO] Checkpoint: ${CKPT}"
echo "[INFO] Base output dir: ${BASE_OUT}"
echo "[INFO] num_inference_steps=${NUM_INFER_STEPS} batch_size=${BATCH_SIZE} seed=${SEED}"
echo

for k in "${KS[@]}"; do
  OUT_DIR="${BASE_OUT}/k_${k}"
  mkdir -p "${OUT_DIR}"

  echo "[RUN] k=${k} -> ${OUT_DIR}"
  python evaluate_dp_dataset.py \
    --checkpoint "${CKPT}" \
    --output_dir "${OUT_DIR}" \
    --num_inference_steps "${NUM_INFER_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --k "${k}"
  echo "[DONE] k=${k}"
  echo
done

echo "[ALL DONE]"
