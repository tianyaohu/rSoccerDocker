#!/usr/bin/env bash
set -euo pipefail

# ── local_ci.sh ──────────────────────────────────────────────
# Run from repo root:  ./local_ci.sh
#
# Stages:
#   1. Unit tests   (test_resume, test_utils, test_train)
#   2. Smoke tests  (tiny real training runs — ~30s each)
#
# Requirements:
#   - Docker image built:  docker build -t rsoccer:latest .
#   - .env file exists (for wandb — smoke tests use offline mode)
# ─────────────────────────────────────────────────────────────

IMAGE="rsoccer:latest"
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
RESET='\033[0m'

pass=0
fail=0

run_stage() {
    local name="$1"
    shift
    echo -e "\n${CYAN}━━━ ${name} ━━━${RESET}"
    if "$@"; then
        echo -e "${GREEN}✓ ${name}${RESET}"
        ((pass++))
    else
        echo -e "${RED}✗ ${name}${RESET}"
        ((fail++))
    fi
}

# ─────────────────────────────────────────────────────────────
# Stage 1: Unit tests
# ─────────────────────────────────────────────────────────────

run_stage "Install pytest" \
    docker run --rm \
        -v "$PWD":/workspace \
        -w /workspace \
        "$IMAGE" \
        bash -lic "pip install -q pytest 2>&1 | tail -1"

run_stage "test_resume.py" \
    docker run --rm \
        -v "$PWD":/workspace \
        -w /workspace \
        "$IMAGE" \
        bash -lic "pip install -q pytest && python -m pytest tests/test_resume.py -v --tb=short"

run_stage "test_utils.py" \
    docker run --rm \
        -v "$PWD":/workspace \
        -w /workspace \
        "$IMAGE" \
        bash -lic "pip install -q pytest && python -m pytest tests/test_utils.py -v --tb=short"

run_stage "test_train.py" \
    docker run --rm \
        -v "$PWD":/workspace \
        -w /workspace \
        "$IMAGE" \
        bash -lic "pip install -q pytest && python -m pytest tests/test_train.py -v --tb=short"

# ─────────────────────────────────────────────────────────────
# Stage 2: Smoke tests (real training, tiny timesteps)
#
# Uses WANDB_MODE=offline so no API key is needed and nothing
# is uploaded. Just verifies the full pipeline runs without
# crashing and produces the expected outputs.
# ─────────────────────────────────────────────────────────────

SMOKE_DIR="$PWD/experiments/_smoke_test"
cleanup_smoke() {
    rm -rf "$SMOKE_DIR"
}
trap cleanup_smoke EXIT

run_stage "Smoke: fresh SAC (1k steps)" \
    docker run --rm \
        -v "$PWD":/workspace \
        -w /workspace \
        -e WANDB_MODE=offline \
        "$IMAGE" \
        bash -lic "python src/train.py \
            --env SSLDribbling-v0 --algo sac \
            --timesteps 1000 --save-freq 500 --max-checkpoints 2 \
            --seed 0 --root /workspace/experiments/_smoke_test/root"

# Verify outputs exist
run_stage "Smoke: verify SAC outputs" \
    docker run --rm \
        -v "$PWD":/workspace \
        -w /workspace \
        "$IMAGE" \
        bash -lic "
            ROOT=/workspace/experiments/_smoke_test/root/experiments/SAC
            RUN=\$(ls -1 \$ROOT | head -1)
            echo \"  Run dir: \$RUN\"

            # Must have these files
            test -f \$ROOT/\$RUN/command.txt          && echo '  ✓ command.txt'
            test -f \$ROOT/\$RUN/wandb_run_id.txt     && echo '  ✓ wandb_run_id.txt'
            ls \$ROOT/\$RUN/checkpoints/*_final.zip    && echo '  ✓ final checkpoint'
            ls \$ROOT/\$RUN/checkpoints/*.zip          && echo '  ✓ checkpoints exist'
            test -d \$ROOT/\$RUN/logs                  && echo '  ✓ logs dir'
        "

run_stage "Smoke: --resume latest" \
    docker run --rm \
        -v "$PWD":/workspace \
        -w /workspace \
        -e WANDB_MODE=offline \
        "$IMAGE" \
        bash -lic "
            # Remove _final.zip so the run looks unfinished
            find /workspace/experiments/_smoke_test/root -name '*_final.zip' -delete
            python src/train.py \
                --resume latest \
                --root /workspace/experiments/_smoke_test/root
        "

run_stage "Smoke: fresh PPO (1k steps)" \
    docker run --rm \
        -v "$PWD":/workspace \
        -w /workspace \
        -e WANDB_MODE=offline \
        "$IMAGE" \
        bash -lic "python src/train.py \
            --env VSS-v0 --algo ppo \
            --timesteps 1000 --save-freq 500 --max-checkpoints 2 \
            --n-envs 2 --seed 0 \
            --root /workspace/experiments/_smoke_test/root"

run_stage "Smoke: fresh DDPG (1k steps)" \
    docker run --rm \
        -v "$PWD":/workspace \
        -w /workspace \
        -e WANDB_MODE=offline \
        "$IMAGE" \
        bash -lic "python src/train.py \
            --env SSLDribbling-v0 --algo ddpg \
            --timesteps 1000 --save-freq 500 --max-checkpoints 2 \
            --seed 0 --root /workspace/experiments/_smoke_test/root"

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

echo ""
echo -e "${CYAN}━━━ Results ━━━${RESET}"
echo -e "  ${GREEN}Passed: ${pass}${RESET}"
if [ "$fail" -gt 0 ]; then
    echo -e "  ${RED}Failed: ${fail}${RESET}"
    exit 1
else
    echo -e "  ${DIM}Failed: 0${RESET}"
    echo -e "\n${GREEN}All stages passed.${RESET}"
fi