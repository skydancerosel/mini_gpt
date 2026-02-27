#!/usr/bin/env bash
#
# Beta2 ablation overnight runner.
#
# Trains 5 conditions, runs analysis, and reheating tests.
# ~8-12h on MPS (M-series Mac).
#
# Usage:
#   nohup bash run_beta2_overnight.sh > beta2_overnight.log 2>&1 &
#   tail -f beta2_overnight.log
#
set -euo pipefail
export PYTHONUNBUFFERED=1

BETA2_VALUES="0.99 0.95 0.90 0.80 0.0"
BASE_DIR="runs/beta2_ablation"
REHEAT_BETA2="0.95 0.80"
REHEAT_CKPTS="2000 4000"
REHEAT_LRS="0.001 0.0006 0.0003"

# Model architecture (must match paper)
N_LAYER=8
D_MODEL=512
N_HEAD=16
D_FF=2048

echo "========================================================================"
echo "  Beta2 Ablation Overnight Run"
echo "  Started: $(date)"
echo "  Beta2 values: ${BETA2_VALUES}"
echo "  Base dir: ${BASE_DIR}"
echo "========================================================================"

mkdir -p "${BASE_DIR}"

# ── Phase 1: Training ─────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 1: Training (5 beta2 conditions, 4000 steps each)          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

FAILED_RUNS=""
for b2 in ${BETA2_VALUES}; do
    OUT_DIR="${BASE_DIR}/pilot_wd0.5_lr0.001_lp2.0_b2${b2}_s42"

    # Skip if already completed
    if [ -f "${OUT_DIR}/pilot_metrics.json" ]; then
        LAST_STEP=$(python -c "import json; m=json.load(open('${OUT_DIR}/pilot_metrics.json')); print(m[-1]['step'])" 2>/dev/null || echo "0")
        if [ "${LAST_STEP}" -ge 4000 ]; then
            echo ""
            echo "  [SKIP] beta2=${b2}: already completed (step ${LAST_STEP})"
            continue
        fi
    fi

    echo ""
    echo "  [TRAIN] beta2=${b2} -> ${OUT_DIR}"
    echo "  $(date)"

    if python pilot.py \
        --seed 42 \
        --wd 0.5 \
        --lr 0.001 \
        --beta2 "${b2}" \
        --lambda-probe 2.0 \
        --lambda-probe2 4.0 \
        --lambda-step 4000 \
        --steps 4000 \
        --warmup 1500 \
        --eval-every 200 \
        --n-layer ${N_LAYER} \
        --d-model ${D_MODEL} \
        --n-head ${N_HEAD} \
        --d-ff ${D_FF} \
        --out-dir "${OUT_DIR}" \
        --ckpt-dense-from 600 \
        --ckpt-dense-to 2000 \
        --ckpt-dense-every 50 \
        --ckpt-sparse-from 2000 \
        --ckpt-sparse-every 100; then
        echo "  [OK] beta2=${b2} completed"
    else
        echo "  [FAIL] beta2=${b2} FAILED (exit code $?)"
        FAILED_RUNS="${FAILED_RUNS} ${b2}"
    fi
done

if [ -n "${FAILED_RUNS}" ]; then
    echo ""
    echo "  WARNING: Failed runs: ${FAILED_RUNS}"
fi

# ── Phase 2: Per-run analysis ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 2: Per-run analysis                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

for b2 in ${BETA2_VALUES}; do
    OUT_DIR="${BASE_DIR}/pilot_wd0.5_lr0.001_lp2.0_b2${b2}_s42"
    if [ ! -d "${OUT_DIR}" ]; then
        echo "  [SKIP] beta2=${b2}: run directory not found"
        continue
    fi
    if [ -f "${OUT_DIR}/analysis/backbone_metrics.json" ] && \
       [ -f "${OUT_DIR}/analysis/update_alignment.json" ]; then
        echo "  [SKIP] beta2=${b2}: analysis already exists"
        continue
    fi

    echo ""
    echo "  [ANALYSIS] beta2=${b2}"
    python beta2_analysis.py --run-dir "${OUT_DIR}" || \
        echo "  [WARN] analysis failed for beta2=${b2}"
done

# ── Phase 3: Cross-run comparison ─────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 3: Cross-run comparison + summary                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

python beta2_analysis.py --base-dir "${BASE_DIR}" --compare

# ── Phase 4: Reheating tests ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 4: Reheating tests                                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

for b2 in ${REHEAT_BETA2}; do
    RUN_DIR="${BASE_DIR}/pilot_wd0.5_lr0.001_lp2.0_b2${b2}_s42"
    if [ ! -d "${RUN_DIR}" ]; then
        echo "  [SKIP] beta2=${b2}: run directory not found"
        continue
    fi

    for ckpt in ${REHEAT_CKPTS}; do
        CKPT_FILE="${RUN_DIR}/ckpt_$(printf '%06d' ${ckpt}).pt"
        if [ ! -f "${CKPT_FILE}" ]; then
            echo "  [SKIP] beta2=${b2}, ckpt=${ckpt}: checkpoint not found"
            continue
        fi

        for rlr in ${REHEAT_LRS}; do
            REHEAT_DIR="${RUN_DIR}/reheat_ckpt${ckpt}_lr${rlr}"

            # Skip if already completed
            if [ -f "${REHEAT_DIR}/pilot_metrics.json" ]; then
                LAST_STEP=$(python -c "import json; m=json.load(open('${REHEAT_DIR}/pilot_metrics.json')); print(m[-1]['step'])" 2>/dev/null || echo "0")
                if [ "${LAST_STEP}" -ge 2000 ]; then
                    echo "  [SKIP] beta2=${b2}, ckpt=${ckpt}, lr=${rlr}: already done"
                    continue
                fi
            fi

            echo ""
            echo "  [REHEAT] beta2=${b2}, ckpt=${ckpt}, lr=${rlr}"
            echo "  $(date)"

            if python pilot.py \
                --resume-from "${CKPT_FILE}" \
                --seed 42 \
                --wd 0.5 \
                --lr "${rlr}" \
                --beta2 "${b2}" \
                --warmup 200 \
                --steps 2000 \
                --lambda-probe 4.0 \
                --eval-every 200 \
                --n-layer ${N_LAYER} \
                --d-model ${D_MODEL} \
                --n-head ${N_HEAD} \
                --d-ff ${D_FF} \
                --out-dir "${REHEAT_DIR}"; then
                echo "  [OK] reheat beta2=${b2}, ckpt=${ckpt}, lr=${rlr}"
            else
                echo "  [FAIL] reheat beta2=${b2}, ckpt=${ckpt}, lr=${rlr}"
            fi
        done
    done
done

# ── Phase 5: Reheating summary ───────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 5: Reheating summary + final comparison                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

python beta2_analysis.py --base-dir "${BASE_DIR}" --reheat-summary

echo ""
echo "========================================================================"
echo "  Beta2 Ablation COMPLETE"
echo "  Finished: $(date)"
echo "  Results: ${BASE_DIR}/summary/"
echo "========================================================================"
