#!/bin/bash
#SBATCH --job-name=llm4poi_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
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

if [[ ! -f "$BASE_MODEL_DIR/config.json" ]]; then
  echo "ERROR: base model not found: $BASE_MODEL_DIR"
  exit 1
fi
if [[ -z "$ADAPTER_DIR" ]]; then
  ADAPTER_DIR="$(ls -dt "$PROJECT_ROOT"/outputs/llm4poi_"${DATASET}"_* 2>/dev/null | head -n 1 || true)"
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

python eval_next_poi.py \
  --model_path "$BASE_MODEL_DIR" \
  --dataset_name "$DATASET" \
  --output_dir "$ADAPTER_DIR" \
  --data_path "$DATA_PATH" \
  --test_file "$TEST_FILE" \
  2>&1 | tee "$RESULT_DIR/eval.log"

grep "ACC@1" "$RESULT_DIR/eval.log" | tail -n 1 > "$RESULT_DIR/metrics.txt" || true
echo "[evaluation] Done. Metrics saved to: $RESULT_DIR/metrics.txt"
