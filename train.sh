#!/usr/bin/env bash
# =============================================================================
# SPT Forest — training wrapper
# Usage: ./train.sh <mode> [options]
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PRETRAINED="$REPO_DIR/pretrained/spt-2_dales.ckpt"
EXPERIMENT="experiment=semantic/forest"
BASE_CMD="python $REPO_DIR/src/train.py $EXPERIMENT"

# ── colour helpers ────────────────────────────────────────────────────────────
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; NC=$'\033[0m'
info()    { echo -e "${CYAN}[spt]${NC} $*"; }
success() { echo -e "${GREEN}[spt]${NC} $*"; }
warn()    { echo -e "${YELLOW}[spt]${NC} $*"; }
die()     { echo -e "${RED}[spt] ERROR:${NC} $*" >&2; exit 1; }

# ── background launcher ───────────────────────────────────────────────────────
# Start command in background, stripping ANSI codes into a clean log file.
# Returns the PID of the background job.
launch_bg() {
  local logfile="$1"; shift
  mkdir -p "$(dirname "$logfile")"
  ( "$@" 2>&1 | sed 's/\x1b\[[0-9;]*m//g; s/\r/\n/g' >> "$logfile" ) &
  local pid=$!
  disown "$pid" 2>/dev/null || true
  echo "$pid"
}

# Follow a log file live; Ctrl+C detaches without killing the training process.
follow_log() {
  local logfile="$1"
  local pid="$2"
  echo ""
  info "━━━ Live output (${BOLD}Ctrl+C${NC}${CYAN} to detach — training keeps running) ━━━"
  trap "echo -e \"\n${YELLOW}[spt]${NC} Detached. Training still running (PID $pid).\n      Monitor : tail -f $logfile\n      Kill    : kill $pid\"; trap - INT; exit 0" INT
  tail -f "$logfile"
}

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
cat << EOF

${BOLD}SPT Forest — training wrapper${NC}

  ${CYAN}./train.sh preprocess${NC}
      Run preprocessing only (no training). Produces .h5 files under
      data/forest/processed/. Safe to re-run — already-processed tiles
      are skipped automatically.

  ${CYAN}./train.sh train${NC}  [--lr <lr>] [--epochs <n>]
      Train from random initialisation. Default: lr=0.01, epochs=500.
      Streams live output to the terminal. Ctrl+C detaches — training
      continues in background.

  ${CYAN}./train.sh resume <ckpt_path>${NC}
      Resume a previous run (restores weights, optimizer, epoch counter).

  ${CYAN}./train.sh eval <ckpt_path>${NC}
      Evaluate a checkpoint on the test split. Prints per-class mIoU.

  ${CYAN}./train.sh infer <ckpt_path>${NC}  [--split test|val] [--data-dir <dir>] [--out <dir>]
      Run inference on the test (or val) split. Writes predicted LAZ files
      with PredictedClass extra-byte field and per-class coloured RGB.
      Prints per-class IoU when ground-truth labels are present.
      --data-dir selects the dataset root (default: data/forest;
                 use data/Inference for newly prepared data).

  ${CYAN}./train.sh mini${NC}
      Quick sanity check — 2 plots per stage, 5 epochs (interactive).

  ${CYAN}Options (train / resume):${NC}
    --lr <float>    learning rate  (default: 0.01)
    --epochs <int>  max epochs     (default: 500)

  ${CYAN}Logs:${NC}  logs/train_forest_scratch.log  (ANSI-stripped, grep-friendly)
  ${CYAN}Ckpt:${NC}  logs/train/runs/<timestamp>/checkpoints/best.ckpt

EOF
}

# ── argument parsing ──────────────────────────────────────────────────────────
MODE="${1:-help}"
shift || true

LR=""
EPOCHS=500
POSITIONAL=""
SPLIT="test"
OUT_DIR=""
DATA_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lr)       LR="$2";       shift 2 ;;
    --epochs)   EPOCHS="$2";   shift 2 ;;
    --split)    SPLIT="$2";    shift 2 ;;
    --out)      OUT_DIR="$2";  shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    -*)         die "Unknown option: $1" ;;
    *)          POSITIONAL="$1"; shift ;;
  esac
done

# ── modes ─────────────────────────────────────────────────────────────────────
case "$MODE" in

  # ── preprocess ──────────────────────────────────────────────────────────────
  preprocess)
    LOG="$REPO_DIR/logs/preprocess_forest.log"
    info "Preprocessing all 14 plots (trainer.max_epochs=0)"
    PID=$(launch_bg "$LOG" $BASE_CMD trainer.max_epochs=0)
    success "Preprocessing started (PID $PID)"
    follow_log "$LOG" "$PID"
    ;;

  # ── train from scratch ───────────────────────────────────────────────────────
  train)
    LR="${LR:-0.01}"
    LOG="$REPO_DIR/logs/train_forest_scratch.log"
    info "Training from scratch — lr=$LR  epochs=$EPOCHS"
    PID=$(launch_bg "$LOG" $BASE_CMD \
      trainer.max_epochs="$EPOCHS" \
      model.optimizer.lr="$LR")
    success "Training started (PID $PID)"
    info "Best ckpt → logs/train/runs/<timestamp>/checkpoints/best.ckpt"
    follow_log "$LOG" "$PID"
    ;;

  # ── resume ──────────────────────────────────────────────────────────────────
  resume)
    [[ -n "$POSITIONAL" ]] || die "Provide a checkpoint path: ./train.sh resume <ckpt>"
    [[ -f "$POSITIONAL" ]] || die "Checkpoint not found: $POSITIONAL"
    LOG="$REPO_DIR/logs/train_forest_resume.log"
    info "Resuming from: $POSITIONAL"
    PID=$(launch_bg "$LOG" $BASE_CMD \
      ckpt_path="$POSITIONAL" \
      trainer.max_epochs="$EPOCHS")
    success "Resume started (PID $PID)"
    follow_log "$LOG" "$PID"
    ;;

  # ── eval ────────────────────────────────────────────────────────────────────
  eval)
    [[ -n "$POSITIONAL" ]] || die "Provide a checkpoint path: ./train.sh eval <ckpt>"
    [[ -f "$POSITIONAL" ]] || die "Checkpoint not found: $POSITIONAL"
    info "Evaluating: $POSITIONAL"
    python "$REPO_DIR/src/eval.py" $EXPERIMENT \
      ckpt_path="$POSITIONAL" \
      trainer=gpu
    ;;

  # ── inference ───────────────────────────────────────────────────────────────
  infer)
    [[ -n "$POSITIONAL" ]] || die "Provide a checkpoint path: ./train.sh infer <ckpt>"
    [[ -f "$POSITIONAL"  ]] || die "Checkpoint not found: $POSITIONAL"
    info "Running inference — ckpt=$POSITIONAL  split=$SPLIT  data-dir=${DATA_DIR:-data/forest}"
    INFER_ARGS=(--ckpt "$POSITIONAL" --split "$SPLIT")
    [[ -n "$DATA_DIR" ]] && INFER_ARGS+=(--data-dir "$DATA_DIR")
    [[ -n "$OUT_DIR"  ]] && INFER_ARGS+=(--out "$OUT_DIR")
    python "$REPO_DIR/scripts/inference.py" "${INFER_ARGS[@]}"
    ;;

  # ── mini sanity check ────────────────────────────────────────────────────────
  mini)
    info "Mini sanity check — 2 plots per stage, 5 epochs (interactive) ..."
    python "$REPO_DIR/src/train.py" $EXPERIMENT \
      "++datamodule.mini=True" \
      trainer.max_epochs=5
    ;;

  # ── help / default ───────────────────────────────────────────────────────────
  help|--help|-h)
    usage
    ;;

  *)
    die "Unknown mode '$MODE'. Run './train.sh help' for usage."
    ;;
esac
