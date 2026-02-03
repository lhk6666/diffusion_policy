#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
CKPT="data/outputs/2026.01.26/23.10.30_train_diffusion_transformer_vla_tokens_vla_nav_224/checkpoints/epoch=0030-val_loss=0.0202.ckpt"
BASE_OUT="/media/dragon_llm/linux_ssd/RSS_2026_exp/benckmark/k_exp_random"
NUM_INFER_STEPS=5
BATCH_SIZE=128
# If SEED is empty/None, we will omit --seed entirely (unseeded/random run).
# Otherwise, pass it through (supports single int or comma-separated list).
SEED=None

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

# Build optional seed args.
SEED_ARGS=()
if [[ -n "${SEED}" && "${SEED}" != "None" && "${SEED}" != "none" ]]; then
  SEED_ARGS=(--seed "${SEED}")
fi

for k in "${KS[@]}"; do
  OUT_DIR="${BASE_OUT}/k_${k}"
  mkdir -p "${OUT_DIR}"

  echo "[RUN] k=${k} -> ${OUT_DIR}"
  python evaluate_dp_dataset.py \
    --checkpoint "${CKPT}" \
    --output_dir "${OUT_DIR}" \
    --num_inference_steps "${NUM_INFER_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    "${SEED_ARGS[@]}" \
    --k "${k}"
  echo "[DONE] k=${k}"
  echo
done

echo "[ALL DONE]"
