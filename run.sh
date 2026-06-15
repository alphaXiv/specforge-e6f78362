#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bebop (arXiv:2606.12370) minimal proof-of-concept.
#
# Core claim: training an EAGLE3 / MTP draft head with the paper's Total
# Variation loss (TV: minimize d_TV(p,q) = 1 - sum_v min(p_v,q_v), which is
# exactly the rejection-sampling acceptance rate) yields HIGHER acceptance
# than the conventional cross-entropy / KL objective, at matched training.
#
# The TV gradient is proportional to the draft prob q_j (paper Eq. 11), so it
# vanishes for a uniform (untrained) head: TV is a refinement objective, which
# matches the paper adapting an already-trained MTP module. We therefore:
#   1. CE-warm up a Qwen3-8B EAGLE3 draft head (single H100),
#   2. fork two matched-length continuations from that checkpoint:
#        - CE-continue  (baseline: keep the standard loss)
#        - TV-finetune  (ours: switch to the paper's TV loss)
#   3. compare per-MTP-step rejection-sampling acceptance on held-out eval.
# Both forks see identical total steps and data; only the final loss differs.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

ART=.openresearch/artifacts
mkdir -p "$ART"
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/compiled_kernels"
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

echo "=================== [0/5] system deps ==================="
# sglang's sgl_kernel (sm90) dlopen fails without libnuma.so.1 on this image.
apt-get update -y >/dev/null 2>&1 && apt-get install -y libnuma1 libnuma-dev >/dev/null 2>&1 || true
python -c "import ctypes; ctypes.CDLL('libnuma.so.1'); print('libnuma OK')" || true

echo "=================== [1/5] install specforge ==================="
pip install -v . 2>&1 | tail -8

echo "=================== [2/5] prepare tiny ShareGPT slice ==================="
if [ ! -f cache/dataset/sharegpt_train.jsonl ]; then
  python scripts/prepare_data.py --dataset sharegpt --sample-size 4000 --split-eval
fi
ls -la cache/dataset/ || true

TARGET=Qwen/Qwen3-8B
WARMUP_STEPS=${WARMUP_STEPS:-1000}
FORK_STEPS=${FORK_STEPS:-600}

# train_eagle3 with the shared minimal config. $1=output tag; rest=extra args.
train () {
  local tag=$1; shift
  echo "=================== train: $tag ($*) ==================="
  torchrun --standalone --nproc_per_node 1 scripts/train_eagle3.py \
    --target-model-path "$TARGET" \
    --draft-model-config configs/qwen3-8b-eagle3.json \
    --train-data-path cache/dataset/sharegpt_train.jsonl \
    --eval-data-path cache/dataset/sharegpt_test.jsonl \
    --output-dir "outputs/$tag" \
    --num-epochs 1 \
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
    --sglang-mem-fraction-static 0.4 \
    "$@" 2>&1 | tee "$ART/train_$tag.log"
}

echo "=================== [3/5] CE warmup ==================="
train warmup --max-num-steps "$WARMUP_STEPS" --total-steps "$WARMUP_STEPS" --eval-interval "$WARMUP_STEPS"
WARMUP_CKPT=$(ls -d outputs/warmup/epoch_*_step_* 2>/dev/null | sort -V | tail -1)
echo "warmup checkpoint: $WARMUP_CKPT"
test -f "$WARMUP_CKPT/config.json"

echo "=================== [4/5] forks: CE-continue vs TV-finetune ==================="
EVAL_EVERY=$(( FORK_STEPS / 2 ))
train ce --ckpt-dir "$WARMUP_CKPT" --max-num-steps "$FORK_STEPS" --total-steps "$FORK_STEPS" --eval-interval "$EVAL_EVERY"
train tv --ckpt-dir "$WARMUP_CKPT" --lk-loss-type tv --max-num-steps "$FORK_STEPS" --total-steps "$FORK_STEPS" --eval-interval "$EVAL_EVERY"

echo "=================== [5/5] assemble comparison report ==================="
cp -f outputs/ce/eval_acceptance.json "$ART/ce_eval_acceptance.json" 2>/dev/null || true
cp -f outputs/tv/eval_acceptance.json "$ART/tv_eval_acceptance.json" 2>/dev/null || true
python poc_report.py \
  --ce outputs/ce/eval_acceptance.json \
  --tv outputs/tv/eval_acceptance.json \
  --out "$ART"
cat "$ART/EVAL.md"
