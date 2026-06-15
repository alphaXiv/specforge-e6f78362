#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bebop (arXiv:2606.12370) minimal proof-of-concept.
#
# Core claim: training an EAGLE3 / MTP draft head with the paper's Total
# Variation loss (TV: minimize d_TV(p,q) = 1 - sum_v min(p_v,q_v), which is
# exactly the rejection-sampling acceptance rate) yields HIGHER acceptance
# than the conventional cross-entropy / KL objective, at matched training.
#
# This script trains a Qwen3-8B EAGLE3 draft head twice on a tiny ShareGPT
# slice (single H100), once with the default CE/KL loss and once with the
# TV loss, then compares per-MTP-step rejection-sampling acceptance.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

ART=.openresearch/artifacts
mkdir -p "$ART"
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/compiled_kernels"
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

echo "=================== [0/4] system deps ==================="
# sglang's sgl_kernel (sm90) dlopen fails without libnuma.so.1 on this image.
apt-get update -y >/dev/null 2>&1 && apt-get install -y libnuma1 libnuma-dev >/dev/null 2>&1 || true
python -c "import ctypes; ctypes.CDLL('libnuma.so.1'); print('libnuma OK')" || true

echo "=================== [1/4] install specforge ==================="
pip install -v . 2>&1 | tail -8

echo "=================== [2/4] prepare tiny ShareGPT slice ==================="
if [ ! -f cache/dataset/sharegpt_train.jsonl ]; then
  python scripts/prepare_data.py --dataset sharegpt --sample-size 3000 --split-eval
fi
ls -la cache/dataset/ || true

TARGET=Qwen/Qwen3-8B
STEPS=${STEPS:-300}
EVAL_EVERY=${EVAL_EVERY:-100}

run_one () {
  local tag=$1; shift
  echo "=================== train: $tag ==================="
  torchrun --standalone --nproc_per_node 1 scripts/train_eagle3.py \
    --target-model-path "$TARGET" \
    --draft-model-config configs/qwen3-8b-eagle3.json \
    --train-data-path cache/dataset/sharegpt_train.jsonl \
    --eval-data-path cache/dataset/sharegpt_test.jsonl \
    --output-dir "outputs/$tag" \
    --num-epochs 1 \
    --max-num-steps "$STEPS" \
    --total-steps "$STEPS" \
    --eval-interval "$EVAL_EVERY" \
    --save-interval 100000 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --ttt-length 3 \
    --chat-template qwen \
    --cache-dir cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 1 \
    --target-model-backend sglang \
    --report-to tensorboard \
    --shard-target-output \
    --sglang-mem-fraction-static 0.4 \
    "$@" 2>&1 | tee "$ART/train_$tag.log"
}

echo "=================== [3/4] train CE baseline, then TV ==================="
run_one ce
run_one tv --lk-loss-type tv

echo "=================== [4/4] assemble comparison report ==================="
python poc_report.py --ce "$ART/train_ce.log" --tv "$ART/train_tv.log" --out "$ART"
cat "$ART/EVAL.md"
