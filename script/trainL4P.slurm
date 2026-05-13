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
### Remove 1st # at next line for the option to take effect ###
#SBATCH --nodelist=TC2N07
#
### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ###
#SBATCH --time=06:00:00
#
#SBATCH --job-name=llm4poi_train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
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
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/llm4poi_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE##*:}"
if ! [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]]; then
  GPUS_PER_NODE=1
fi
if (( GPUS_PER_NODE > 1 )); then
  echo "ERROR: this script is constrained to at most 1 GPU, got GPUS_PER_NODE=$GPUS_PER_NODE"
  exit 1
fi

if [[ "$DATASET" != "nyc" && "$DATASET" != "tky" && "$DATASET" != "ca" ]]; then
  echo "ERROR: DATASET must be one of nyc|tky|ca, got: $DATASET"
  exit 1
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

# Hyperparameters follow paper + repo defaults:
# lr=2e-5, warmup_steps=20, epochs=3, sequence_length=32768, batch_size=1 per GPU.
torchrun --nproc_per_node="$GPUS_PER_NODE" supervised-fine-tune-qlora.py \
  --model_name_or_path "$BASE_MODEL_DIR" \
  --model_type llama \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
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
  --save_steps 1000 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_steps 20 \
  --lr_scheduler_type constant_with_warmup \
  --logging_steps 1 \
  --deepspeed ds_configs/stage2.json \
  --tf32 True

echo "[trainL4P] Done. Fine-tuned adapter saved to: $OUTPUT_DIR"
