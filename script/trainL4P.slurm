#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#
### Specify Memory allocate to this job ###
#SBATCH --mem=48G
#
### Specify number of core (CPU) to allocate to per task ###
#SBATCH --cpus-per-task=1
#
### Specify number of node to compute ###
#SBATCH --nodes=1
#
### Optional: Specify node to execute the job ###
#SBATCH --nodelist=TC2N07
#
### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ###
#SBATCH --time=06:00:00
#
#SBATCH --job-name=llm4poi_train
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONDA_MODULE="${CONDA_MODULE:-anaconda/25.5.1}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8.0}"
DATASETS=(${DATASETS:-nyc tky ca})
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${DATASET:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#DATASETS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is outside DATASETS=(${DATASETS[*]})"
    exit 1
  fi
  DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
fi
DATASET="${DATASET:-nyc}"     # nyc | tky | ca
ENV_NAME="${ENV_NAME:-llm4poi}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$PROJECT_ROOT/models/llama-2-7b-longlora-32k-ft}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/datasets/$DATASET/preprocessed/train_qa_pairs_kqt.json}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"
if ! [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]]; then
  GPUS_PER_NODE=1
fi
if (( GPUS_PER_NODE > 1 )); then
  echo "ERROR: this script is constrained to at most 1 GPU, got GPUS_PER_NODE=$GPUS_PER_NODE"
  exit 1
fi

load_hpc_modules() {
  if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/Modules/init/bash ]]; then
      # shellcheck disable=SC1091
      source /usr/share/Modules/init/bash
    fi
  fi
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: environment modules are not available in this Slurm shell."
    exit 1
  fi
  module load "$CONDA_MODULE"
  module load "$CUDA_MODULE"
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found after loading module $CONDA_MODULE"
    exit 1
  fi
}

if [[ "$DATASET" != "nyc" && "$DATASET" != "tky" && "$DATASET" != "ca" ]]; then
  echo "ERROR: DATASET must be one of nyc|tky|ca, got: $DATASET"
  exit 1
fi

checkpoint_step() {
  basename "$1" | sed 's/^checkpoint-//'
}

checkpoint_is_complete() {
  local ckpt="$1"
  local tag
  local tag_dir

  [[ -d "$ckpt" ]] || return 1
  [[ -s "$ckpt/trainer_state.json" ]] || return 1
  [[ -s "$ckpt/latest" ]] || return 1

  tag="$(head -n 1 "$ckpt/latest" | tr -d '[:space:]')"
  [[ -n "$tag" ]] || return 1
  tag_dir="$ckpt/$tag"
  [[ -d "$tag_dir" ]] || return 1

  find "$tag_dir" -maxdepth 1 -type f -name 'mp_rank_*_model_states.pt' | grep -q . || return 1
  find "$tag_dir" -maxdepth 1 -type f -name '*optim_states.pt' | grep -q . || return 1
  return 0
}

find_latest_complete_checkpoint() {
  local run_dir="$1"
  local ckpt

  while IFS= read -r ckpt; do
    if checkpoint_is_complete "$ckpt"; then
      echo "$ckpt"
      return 0
    fi
    echo "[trainL4P] Skipping incomplete checkpoint: $ckpt" >&2
  done < <(
    find "$run_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f %p\n' 2>/dev/null \
      | awk '{name=$1; sub(/^checkpoint-/, "", name); print name, $2}' \
      | sort -nr \
      | cut -d' ' -f2-
  )
  return 1
}

mkdir -p "$PROJECT_ROOT/outputs"
if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ "$RESUME_FROM_CHECKPOINT" == "auto" ]]; then
    LATEST_RUN_WITH_CKPT="$(
      while IFS= read -r run_dir; do
        if find_latest_complete_checkpoint "$run_dir" >/dev/null; then
          echo "$run_dir"
          break
        fi
      done < <(find "$PROJECT_ROOT/outputs" -maxdepth 1 -type d -name "llm4poi_${DATASET}*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-)
    )"
    if [[ -n "$LATEST_RUN_WITH_CKPT" ]]; then
      OUTPUT_DIR="$LATEST_RUN_WITH_CKPT"
    else
      OUTPUT_DIR="$PROJECT_ROOT/outputs/llm4poi_${DATASET}"
    fi
  else
    OUTPUT_DIR="$PROJECT_ROOT/outputs/llm4poi_${DATASET}"
  fi
fi

RESUME_ARG=()
if [[ "$RESUME_FROM_CHECKPOINT" == "auto" ]]; then
  RESUME_FROM_CHECKPOINT="$(find_latest_complete_checkpoint "$OUTPUT_DIR" || true)"
  if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    LATEST_STEP="$(checkpoint_step "$RESUME_FROM_CHECKPOINT")"
    echo "[trainL4P] Latest complete checkpoint step: $LATEST_STEP"
  else
    RESUME_FROM_CHECKPOINT=""
    echo "[trainL4P] No complete checkpoint found under OUTPUT_DIR; starting from scratch."
  fi
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" && "$RESUME_FROM_CHECKPOINT" != "none" && "$RESUME_FROM_CHECKPOINT" != "false" && "$RESUME_FROM_CHECKPOINT" != "0" ]]; then
  if [[ ! -d "$RESUME_FROM_CHECKPOINT" ]]; then
    echo "ERROR: RESUME_FROM_CHECKPOINT does not exist: $RESUME_FROM_CHECKPOINT"
    exit 1
  fi
  if ! checkpoint_is_complete "$RESUME_FROM_CHECKPOINT"; then
    echo "ERROR: RESUME_FROM_CHECKPOINT is not a complete DeepSpeed checkpoint: $RESUME_FROM_CHECKPOINT"
    echo "Choose an earlier checkpoint-* directory or set RESUME_FROM_CHECKPOINT=auto."
    exit 1
  fi
  RESUME_ARG=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "ERROR: train data not found: $TRAIN_DATA"
  echo "Run script/env_prepare.slurm first."
  exit 1
fi
if [[ ! -f "$BASE_MODEL_DIR/config.json" ]]; then
  echo "ERROR: base model not found: $BASE_MODEL_DIR"
  echo "Run script/env_prepare.slurm first or set BASE_MODEL_DIR."
  exit 1
fi

load_hpc_modules
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "[trainL4P] DATASET=$DATASET"
echo "[trainL4P] TRAIN_DATA=$TRAIN_DATA"
echo "[trainL4P] BASE_MODEL_DIR=$BASE_MODEL_DIR"
echo "[trainL4P] OUTPUT_DIR=$OUTPUT_DIR"
echo "[trainL4P] GPUS_PER_NODE=$GPUS_PER_NODE"
echo "[trainL4P] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-<none>}"
echo "[trainL4P] SAVE_STEPS=$SAVE_STEPS"
echo "[trainL4P] CONDA_MODULE=$CONDA_MODULE"
echo "[trainL4P] CUDA_MODULE=$CUDA_MODULE"

# Hyperparameters follow paper + repo defaults:
# lr=2e-5, warmup_steps=20, epochs=3, sequence_length=32768, batch_size=1 per GPU.
torchrun --nproc_per_node="$GPUS_PER_NODE" supervised-fine-tune-qlora.py \
  --model_name_or_path "$BASE_MODEL_DIR" \
  --model_type llama \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir True \
  "${RESUME_ARG[@]}" \
  --model_max_length 32768 \
  --use_flash_attn True \
  --data_path "$TRAIN_DATA" \
  --low_rank_training True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_steps 20 \
  --lr_scheduler_type constant_with_warmup \
  --logging_steps "$LOGGING_STEPS" \
  --deepspeed ds_configs/stage2.json \
  --tf32 True

echo "[trainL4P] Done. Fine-tuned adapter saved to: $OUTPUT_DIR"
