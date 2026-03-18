#!/bin/bash
# Phase 1A: Continue β2 training from step 4000 to 10000
# Runs sequentially (MPS device shared)
set -e

cd /Users/yongzhongxu/mini_gpt

echo "============================================================"
echo "  Phase 1A: Continue training to 10K steps"
echo "  3 runs: β2={0.99, 0.95, 0.80}"
echo "============================================================"

COMMON_ARGS="--steps 10000 --lr 0.001 --wd 0.5 --lambda-probe 2.0 --p-probe 0.20 \
  --n-layer 8 --d-model 512 --n-head 16 --d-ff 2048 --eval-every 200"

echo ""
echo ">>> [1/3] β2=0.99"
echo "    Start: $(date)"
python training/pilot.py \
  --continue-from runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.99_s42/ckpt_004000.pt \
  --beta2 0.99 \
  --out-dir runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.99_s42 \
  $COMMON_ARGS
echo "    Done: $(date)"

echo ""
echo ">>> [2/3] β2=0.95"
echo "    Start: $(date)"
python training/pilot.py \
  --continue-from runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.95_s42/ckpt_004000.pt \
  --beta2 0.95 \
  --out-dir runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.95_s42 \
  $COMMON_ARGS
echo "    Done: $(date)"

echo ""
echo ">>> [3/3] β2=0.80"
echo "    Start: $(date)"
python training/pilot.py \
  --continue-from runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.80_s42/ckpt_004000.pt \
  --beta2 0.80 \
  --out-dir runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.80_s42 \
  $COMMON_ARGS
echo "    Done: $(date)"

echo ""
echo "============================================================"
echo "  Phase 1A complete! $(date)"
echo "============================================================"
