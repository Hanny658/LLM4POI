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
#SBATCH --job-name=llm4poi_eval
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
DATASET="${DATASET:-nyc}"    # nyc | tky | ca
ENV_NAME="${ENV_NAME:-llm4poi}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$PROJECT_ROOT/models/llama-2-7b-longlora-32k-ft}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/datasets/$DATASET/preprocessed}"
TEST_FILE="${TEST_FILE:-test_qa_pairs_kqt.txt}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/outputs/eval_${DATASET}_$(date +%Y%m%d_%H%M%S)}"

if [[ "$DATASET" != "nyc" && "$DATASET" != "tky" && "$DATASET" != "ca" ]]; then
  echo "ERROR: DATASET must be one of nyc|tky|ca, got: $DATASET"
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

if [[ ! -f "$BASE_MODEL_DIR/config.json" ]]; then
  echo "ERROR: base model not found: $BASE_MODEL_DIR"
  exit 1
fi
if [[ -z "$ADAPTER_DIR" ]]; then
  shopt -s nullglob
  ADAPTER_CANDIDATES=(
    "$PROJECT_ROOT"/outputs/llm4poi_"${DATASET}"
    "$PROJECT_ROOT"/outputs/llm4poi_"${DATASET}"_*
  )
  shopt -u nullglob
  if (( ${#ADAPTER_CANDIDATES[@]} > 0 )); then
    ADAPTER_DIR="$(ls -dt "${ADAPTER_CANDIDATES[@]}" 2>/dev/null | head -n 1 || true)"
  fi
fi
if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "ERROR: adapter dir not found: $ADAPTER_DIR"
  echo "Run script/trainL4P.slurm first or set ADAPTER_DIR."
  exit 1
fi
if [[ ! -f "$DATA_PATH/$TEST_FILE" ]]; then
  echo "ERROR: test file not found: $DATA_PATH/$TEST_FILE"
  echo "Run script/env_prepare.slurm first or set DATA_PATH/TEST_FILE."
  exit 1
fi

load_hpc_modules
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$RESULT_DIR"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[evaluation] DATASET=$DATASET"
echo "[evaluation] BASE_MODEL_DIR=$BASE_MODEL_DIR"
echo "[evaluation] ADAPTER_DIR=$ADAPTER_DIR"
echo "[evaluation] DATA_PATH=$DATA_PATH"
echo "[evaluation] TEST_FILE=$TEST_FILE"
echo "[evaluation] RESULT_DIR=$RESULT_DIR"
echo "[evaluation] CONDA_MODULE=$CONDA_MODULE"
echo "[evaluation] CUDA_MODULE=$CUDA_MODULE"

python eval_next_poi.py \
  --model_path "$BASE_MODEL_DIR" \
  --dataset_name "$DATASET" \
  --output_dir "$ADAPTER_DIR" \
  --data_path "$DATA_PATH" \
  --test_file "$TEST_FILE" \
  2>&1 | tee "$RESULT_DIR/eval.log"

grep "ACC@1" "$RESULT_DIR/eval.log" | tail -n 1 > "$RESULT_DIR/metrics.txt" || true
echo "[evaluation] Done. Metrics saved to: $RESULT_DIR/metrics.txt"
