#!/usr/bin/env bash
#
# Master pipeline for running the full experiment with a given seed.
#
# Usage:
#   ./run_seed.sh 137                          # full 10K-step run
#   ./run_seed.sh 137 --steps 2000 --eval-every 100   # mini dry run
#
# Features:
#   - Resume support via stage completion flags (23h runs WILL crash)
#   - Environment snapshot (pip freeze, uname, git hash)
#   - Config diff verification (only seed should differ from seed-42)
#
set -euo pipefail

# ── Parse arguments ────────────────────────────────────────────────────
SEED="${1:?Usage: ./run_seed.sh <seed> [--steps N] [--eval-every N]}"
shift

# Defaults (full run)
STEPS=10000
EVAL_EVERY=200
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)      STEPS="$2";      shift 2 ;;
        --eval-every) EVAL_EVERY="$2"; shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Derived paths ──────────────────────────────────────────────────────
WD=0.5
LR=0.001
LP=2.0
LP2=4.0
LP_STEP=4000
RUN_DIR="runs/pilot_wd${WD}_lr${LR}_lp${LP}_s${SEED}"
REF_DIR="runs/pilot_wd${WD}_lr${LR}_lp${LP}_s42"
STAGE_DIR="${RUN_DIR}/.stages"
ANALYSIS_DIR="${RUN_DIR}/analysis"

mkdir -p "$STAGE_DIR" "$ANALYSIS_DIR"

echo "========================================================================"
echo "  SEED PIPELINE: seed=${SEED}"
echo "  Run dir:       ${RUN_DIR}"
echo "  Steps:         ${STEPS}"
echo "  Eval every:    ${EVAL_EVERY}"
echo "  Stages:        ${STAGE_DIR}"
echo "========================================================================"

# ── Stage runner with resume support ───────────────────────────────────
run_stage() {
    local stage="$1"; shift
    if [ -f "${STAGE_DIR}/${stage}.done" ]; then
        echo ""
        echo ">>> Stage [${stage}] already complete, skipping"
        return 0
    fi
    echo ""
    echo ">>> Running stage [${stage}]..."
    echo ">>> Command: $*"
    "$@" && touch "${STAGE_DIR}/${stage}.done"
    echo ">>> Stage [${stage}] completed"
}

# ── Stage 1: Environment snapshot ──────────────────────────────────────
run_stage "01_env_snapshot" bash -c "
    pip freeze > '${RUN_DIR}/env.txt'
    uname -a > '${RUN_DIR}/system.txt'
    python --version >> '${RUN_DIR}/system.txt' 2>&1
    git rev-parse HEAD > '${RUN_DIR}/git_commit.txt' 2>/dev/null || echo 'not a git repo' > '${RUN_DIR}/git_commit.txt'
    echo 'Environment snapshot saved to ${RUN_DIR}/'
"

# ── Stage 2: Training ─────────────────────────────────────────────────
run_stage "02_training" python -u pilot.py \
    --seed "$SEED" \
    --wd "$WD" \
    --lr "$LR" \
    --lambda-probe "$LP" \
    --lambda-probe2 "$LP2" \
    --lambda-step "$LP_STEP" \
    --steps "$STEPS" \
    --eval-every "$EVAL_EVERY" \
    --p-probe 0.10 \
    --n-layer 8 \
    --d-model 512 \
    --n-head 16 \
    --d-ff 2048 \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

# ── Stage 3: Config diff verification ─────────────────────────────────
if [ -f "${REF_DIR}/config.json" ] && [ -f "${RUN_DIR}/config.json" ]; then
    run_stage "03_config_diff" python -c "
import json, sys
c_ref = json.load(open('${REF_DIR}/config.json'))
c_new = json.load(open('${RUN_DIR}/config.json'))
diffs = {k: (c_ref[k], c_new[k]) for k in c_ref if c_ref.get(k) != c_new.get(k)}
print('Config diffs vs seed-42:', diffs)
if not (set(diffs.keys()) <= {'seed'}):
    print('ERROR: Unexpected config differences beyond seed!', file=sys.stderr)
    sys.exit(1)
print('Config verification PASSED: only seed differs')
"
else
    echo ">>> Skipping config diff: reference config not found at ${REF_DIR}/config.json"
fi

# ── Stage 3b: Estimate eval noise floor ──────────────────────────────
run_stage "03b_eval_noise" python -u estimate_eval_noise.py \
    --run-dir "$RUN_DIR" \
    --n-boot 20

# ── Stage 4: Detect oscillations (reads eval_noise.json for tiering) ─
run_stage "04_detect_oscillations" python -u detect_oscillations.py \
    --run-dir "$RUN_DIR"

# Check that manifest was created
MANIFEST="${RUN_DIR}/oscillation_manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: oscillation_manifest.json not created. Aborting."
    exit 1
fi

# Extract switch-pairs from manifest
SWITCH_PAIRS=$(python -c "import json; m=json.load(open('${MANIFEST}')); print(m['switch_pairs_str'])")
echo ">>> Switch pairs from manifest: ${SWITCH_PAIRS}"

# ── Stage 5: Attractor analysis (B1-B3) ───────────────────────────────
run_stage "05_attractor_analysis" python -u attractor_analysis.py \
    --run-dir "$RUN_DIR" \
    --seed "$SEED" \
    --switch-pairs "$SWITCH_PAIRS"

# ── Stage 6: Basin geometry (B5-B7) ───────────────────────────────────
# This is the longest stage (~10h). Run with manifest for seed-agnostic markers.
run_stage "06_basin_geometry" python -u basin_geometry.py \
    --run-dir "$RUN_DIR" \
    --seed "$SEED" \
    --manifest "$MANIFEST" \
    --n-trials 4 \
    --dump-config \
    --resume

# ── Stage 7: Reheating experiments ────────────────────────────────────
LAST_CKPT="${RUN_DIR}/ckpt_$(printf '%06d' "$STEPS").pt"
if [ -f "$LAST_CKPT" ]; then
    for REHEAT_LR in 1e-3 6e-4 3e-4; do
        REHEAT_DIR="${RUN_DIR}_reheat_lr${REHEAT_LR}"
        STAGE_NAME="07_reheat_${REHEAT_LR}"

        run_stage "$STAGE_NAME" python -u pilot.py \
            --seed "$SEED" \
            --wd "$WD" \
            --lr "$REHEAT_LR" \
            --lambda-probe 4.0 \
            --steps 2000 \
            --warmup 200 \
            --eval-every "$EVAL_EVERY" \
            --p-probe 0.10 \
            --n-layer 8 \
            --d-model 512 \
            --n-head 16 \
            --d-ff 2048 \
            --resume-from "$LAST_CKPT"
    done

    # ── Stage 8: Reheating plots ──────────────────────────────────────
    REHEAT_PAIRS=""
    for REHEAT_LR in 3e-4 6e-4 1e-3; do
        # pilot.py formats floats via Python f-string, so 1e-3 → 0.001
        REHEAT_DIR_PY=$(python -c "print(f'runs/pilot_wd${WD}_lr{float(\"${REHEAT_LR}\")}_lp4.0_s${SEED}')")
        if [ -d "$REHEAT_DIR_PY" ]; then
            [ -n "$REHEAT_PAIRS" ] && REHEAT_PAIRS="${REHEAT_PAIRS},"
            REHEAT_PAIRS="${REHEAT_PAIRS}${REHEAT_LR}:${REHEAT_DIR_PY}"
        fi
    done

    if [ -n "$REHEAT_PAIRS" ]; then
        run_stage "08_reheat_plots" python -u attractor_analysis.py \
            --run-dir "$RUN_DIR" \
            --seed "$SEED" \
            --skip-basin \
            --skip-subspace \
            --switch-pairs "$SWITCH_PAIRS" \
            --reheat-dirs "$REHEAT_PAIRS"
    fi
else
    echo ">>> Skipping reheating: last checkpoint ${LAST_CKPT} not found"
fi

# ── Stage 9: Directional probing (logit lens + trajectory PCA + block probe) ──
run_stage "09_directional_probing" python -u directional_probing.py \
    --run-dir "$RUN_DIR" \
    --seed "$SEED" \
    --manifest "$MANIFEST" \
    --sigma-grid "0.0,0.5,1.0,1.5,2.0" \
    --n-random 3 \
    --max-pairs 3

# ── Stage 10: Fisher eigenvalue analysis ─────────────────────────────
run_stage "10_fisher_analysis" python -u fisher_analysis.py \
    --run-dir "$RUN_DIR" \
    --seed "$SEED" \
    --manifest "$MANIFEST" \
    --top-k 20 \
    --n-batches 32

echo ""
echo "========================================================================"
echo "  PIPELINE COMPLETE: seed=${SEED}"
echo "  All outputs in ${RUN_DIR}/"
echo "========================================================================"
