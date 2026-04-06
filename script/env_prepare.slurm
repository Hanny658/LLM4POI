#!/bin/bash
#SBATCH --job-name=llm4poi_env_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DATASET="${DATASET:-nyc}"                     # one of: nyc, tky, ca
ENV_NAME="${ENV_NAME:-llm4poi}"
BASE_MODEL_ID="${BASE_MODEL_ID:-Yukang/Llama-2-7b-longlora-32k-ft}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$PROJECT_ROOT/models/llama-2-7b-longlora-32k-ft}"
DATASET_ZIP_URL="${DATASET_ZIP_URL:-https://www.dropbox.com/scl/fi/teo5pn8t296joue5c8pim/datasets.zip?rlkey=xvcgtdd9vlycep3nw3k17lfae&dl=1}"
DATASET_ZIP="${DATASET_ZIP:-$PROJECT_ROOT/datasets.zip}"
SIM_OUTPUT_DIR="${SIM_OUTPUT_DIR:-$PROJECT_ROOT/outputs/traj_sim_${DATASET}}"
RAW_DIR="$PROJECT_ROOT/datasets/$DATASET/raw"
PREPROCESSED_DIR="$PROJECT_ROOT/datasets/$DATASET/preprocessed"

if [[ "$DATASET" != "nyc" && "$DATASET" != "tky" && "$DATASET" != "ca" ]]; then
  echo "ERROR: DATASET must be one of nyc|tky|ca, got: $DATASET"
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: command not found: $1"
    exit 1
  fi
}

echo "[env_prepare] PROJECT_ROOT=$PROJECT_ROOT"
echo "[env_prepare] DATASET=$DATASET"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/models" "$RAW_DIR"

require_cmd python
require_cmd conda
require_cmd unzip
require_cmd curl

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk 'NR>2 {print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[env_prepare] Creating conda env: $ENV_NAME"
  TMP_ENV_YML="$(mktemp)"
  grep -v '^prefix:' "$PROJECT_ROOT/environment.yml" > "$TMP_ENV_YML"
  conda env create -n "$ENV_NAME" -f "$TMP_ENV_YML"
  rm -f "$TMP_ENV_YML"
fi
conda activate "$ENV_NAME"

if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
  pip install "huggingface_hub[cli]"
fi
if [[ "$DATASET" == "ca" ]]; then
  if ! python -c "import shapely" >/dev/null 2>&1; then
    pip install shapely
  fi
fi

if [[ ! -f "$BASE_MODEL_DIR/config.json" ]]; then
  echo "[env_prepare] Downloading base model: $BASE_MODEL_ID"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
  fi
  huggingface-cli download "$BASE_MODEL_ID" \
    --local-dir "$BASE_MODEL_DIR" \
    --local-dir-use-symlinks False
fi

# Optional dataset bootstrap from the official ZIP in README.
if [[ ! -f "$DATASET_ZIP" ]]; then
  echo "[env_prepare] Downloading datasets.zip from Dropbox"
  curl -L "$DATASET_ZIP_URL" -o "$DATASET_ZIP"
fi
if [[ -f "$DATASET_ZIP" ]]; then
  unzip -o "$DATASET_ZIP" -d "$PROJECT_ROOT" >/dev/null
fi
if [[ -f "$PROJECT_ROOT/datasets/$DATASET/raw.zip" ]]; then
  unzip -o "$PROJECT_ROOT/datasets/$DATASET/raw.zip" -d "$PROJECT_ROOT/datasets/$DATASET" >/dev/null
fi

# Normalize raw file layout to datasets/<dataset>/raw/* expected by preprocessing.
mkdir -p "$RAW_DIR"
if [[ "$DATASET" == "nyc" ]]; then
  for f in NYC_train.csv NYC_val.csv NYC_test.csv; do
    if [[ ! -f "$RAW_DIR/$f" && -f "$PROJECT_ROOT/datasets/nyc/$f" ]]; then
      cp "$PROJECT_ROOT/datasets/nyc/$f" "$RAW_DIR/$f"
    fi
  done
fi
if [[ "$DATASET" == "tky" ]]; then
  if [[ ! -f "$RAW_DIR/dataset_TSMC2014_TKY.txt" && -f "$PROJECT_ROOT/datasets/tky/dataset_TSMC2014_TKY.txt" ]]; then
    cp "$PROJECT_ROOT/datasets/tky/dataset_TSMC2014_TKY.txt" "$RAW_DIR/dataset_TSMC2014_TKY.txt"
  fi
fi
if [[ "$DATASET" == "ca" ]]; then
  for f in loc-gowalla_totalCheckins.txt gowalla_spots_subset1.csv us_state_polygon_json.json dataset_gowalla_ca_ne.csv; do
    if [[ ! -f "$RAW_DIR/$f" && -f "$PROJECT_ROOT/datasets/ca/$f" ]]; then
      cp "$PROJECT_ROOT/datasets/ca/$f" "$RAW_DIR/$f"
    fi
  done
  if [[ ! -f "$RAW_DIR/dataset_gowalla_ca_ne.csv" ]]; then
    mkdir -p "$PROJECT_ROOT/preprocessing/data/ca/raw"
    cp "$RAW_DIR/loc-gowalla_totalCheckins.txt" "$PROJECT_ROOT/preprocessing/data/ca/raw/"
    cp "$RAW_DIR/gowalla_spots_subset1.csv" "$PROJECT_ROOT/preprocessing/data/ca/raw/"
    cp "$RAW_DIR/us_state_polygon_json.json" "$PROJECT_ROOT/preprocessing/data/ca/raw/"
    (
      cd "$PROJECT_ROOT/preprocessing"
      python generate_ca_raw.py
    )
    cp "$PROJECT_ROOT/preprocessing/data/ca/raw/dataset_gowalla_ca_ne.csv" "$RAW_DIR/"
  fi
fi

echo "[env_prepare] Running LLM4POI preprocessing pipeline"
(
  cd "$PROJECT_ROOT/preprocessing"
  python run.py -f "conf/best_conf/${DATASET}.yml"
  python traj_qk.py -dataset_name "$DATASET"
)

python "$PROJECT_ROOT/traj_sim.py" \
  --dataset_name "$DATASET" \
  --base_model "$BASE_MODEL_DIR" \
  --data_path "$PREPROCESSED_DIR" \
  --output_dir "$SIM_OUTPUT_DIR"

(
  cd "$PROJECT_ROOT/preprocessing"
  python to_nextpoi_kqt.py -dataset_name "$DATASET"
)

# Compatibility path for scripts/readme using datasets/processed/<dataset>/...
mkdir -p "$PROJECT_ROOT/datasets/processed"
ln -sfn "../$DATASET/preprocessed" "$PROJECT_ROOT/datasets/processed/$DATASET"

test -f "$PREPROCESSED_DIR/train_qa_pairs_kqt.json"
test -f "$PREPROCESSED_DIR/test_qa_pairs_kqt.txt"
echo "[env_prepare] Done. QA files generated under: $PREPROCESSED_DIR"
