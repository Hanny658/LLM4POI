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
#SBATCH --job-name=llm4poi_env_prep
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail
TMP_ENV_YML=""
cleanup() {
  if [[ -n "$TMP_ENV_YML" ]]; then
    rm -f "$TMP_ENV_YML"
  fi
}
trap cleanup EXIT

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONDA_MODULE="${CONDA_MODULE:-anaconda/25.5.1}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8.0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.3.2}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
RECREATE_ENV="${RECREATE_ENV:-0}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
FORCE_TRAJ_QK="${FORCE_TRAJ_QK:-0}"
FORCE_TRAJ_SIM="${FORCE_TRAJ_SIM:-0}"
FORCE_QA="${FORCE_QA:-0}"
DATASETS=(${DATASETS:-nyc tky ca})
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${DATASET:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#DATASETS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is outside DATASETS=(${DATASETS[*]})"
    exit 1
  fi
  DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
fi
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

files_exist() {
  local path
  for path in "$@"; do
    [[ -s "$path" ]] || return 1
  done
  return 0
}

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

echo "[env_prepare] PROJECT_ROOT=$PROJECT_ROOT"
echo "[env_prepare] DATASET=$DATASET"
echo "[env_prepare] CONDA_MODULE=$CONDA_MODULE"
echo "[env_prepare] CUDA_MODULE=$CUDA_MODULE"
echo "[env_prepare] FLASH_ATTN_VERSION=$FLASH_ATTN_VERSION"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/models" "$RAW_DIR"

load_hpc_modules
require_cmd python
require_cmd conda
require_cmd unzip
require_cmd curl

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "$RECREATE_ENV" == "1" ]] && conda env list | awk 'NR>2 {print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[env_prepare] Removing existing conda env because RECREATE_ENV=1: $ENV_NAME"
  conda env remove -n "$ENV_NAME" -y
fi

if ! conda env list | awk 'NR>2 {print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[env_prepare] Creating conda env: $ENV_NAME"
  TMP_ENV_YML="${TMPDIR:-/tmp}/llm4poi_env_${SLURM_JOB_ID:-$$}.yml"
  # Install flash-attn after the env exists. Its setup.py imports packaging
  # during metadata generation, so putting it inside the pip section can fail
  # before packaging is installed.
  grep -v '^prefix:' "$PROJECT_ROOT/environment.yml" \
    | grep -v -E '^[[:space:]]*-[[:space:]]*flash-attn(==.*)?[[:space:]]*$' \
    | sed -E 's/^([[:space:]]*-[[:space:]]*)transformers==4\.34\.0\.dev0[[:space:]]*$/\1transformers==4.34.0/' \
    > "$TMP_ENV_YML"
  conda env create -n "$ENV_NAME" -f "$TMP_ENV_YML"
  rm -f "$TMP_ENV_YML"
  TMP_ENV_YML=""
fi
conda activate "$ENV_NAME"

if ! python -c "import torch, transformers, peft, pandas" >/dev/null 2>&1; then
  echo "ERROR: conda env '$ENV_NAME' exists but appears incomplete."
  echo "The previous failed flash-attn install may have interrupted pip dependency installation."
  echo "Rerun with RECREATE_ENV=1, or remove it manually with: conda env remove -n $ENV_NAME"
  exit 1
fi

python -m pip install --upgrade packaging wheel ninja cmake lit
if ! python -c "import tensorboard" >/dev/null 2>&1; then
  python -m pip install tensorboard
fi
if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
  if ! python -c "import flash_attn" >/dev/null 2>&1; then
    echo "[env_prepare] Installing flash-attn==$FLASH_ATTN_VERSION with MAX_JOBS=$MAX_JOBS"
    python -m pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation
  fi
else
  echo "[env_prepare] Skipping flash-attn install because INSTALL_FLASH_ATTN=$INSTALL_FLASH_ATTN"
fi

if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
  python -m pip install "huggingface_hub[cli]"
fi
if [[ "$DATASET" == "ca" ]]; then
  if ! python -c "import shapely" >/dev/null 2>&1; then
    python -m pip install shapely
  fi
fi

if [[ ! -f "$BASE_MODEL_DIR/config.json" ]]; then
  echo "[env_prepare] Downloading base model: $BASE_MODEL_ID"
  python - "$BASE_MODEL_ID" "$BASE_MODEL_DIR" <<'PY'
import inspect
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
token = os.environ.get("HF_TOKEN")

kwargs = {
    "repo_id": repo_id,
    "local_dir": local_dir,
    "local_dir_use_symlinks": False,
}
params = inspect.signature(snapshot_download).parameters
if token:
    if "token" in params:
        kwargs["token"] = token
    elif "use_auth_token" in params:
        kwargs["use_auth_token"] = token

snapshot_download(**kwargs)
PY
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

echo "[env_prepare] Running/resuming LLM4POI preprocessing pipeline"
if [[ "$FORCE_PREPROCESS" == "1" ]] || ! files_exist \
  "$PREPROCESSED_DIR/train_sample.csv" \
  "$PREPROCESSED_DIR/test_sample_with_traj.csv"; then
  echo "[env_prepare] Stage preprocess: running"
  (
    cd "$PROJECT_ROOT/preprocessing"
    python run.py -f "best_conf/${DATASET}.yml"
  )
else
  echo "[env_prepare] Stage preprocess: skipped, outputs already exist"
fi

if [[ "$FORCE_TRAJ_QK" == "1" ]] || ! files_exist \
  "$PREPROCESSED_DIR/train_kq_pairs.json" \
  "$PREPROCESSED_DIR/test_kq_pairs.json"; then
  echo "[env_prepare] Stage traj_qk: running"
  (
    cd "$PROJECT_ROOT/preprocessing"
    python traj_qk.py -dataset_name "$DATASET"
  )
else
  echo "[env_prepare] Stage traj_qk: skipped, outputs already exist"
fi

if [[ "$FORCE_TRAJ_SIM" == "1" ]] || ! files_exist \
  "$PREPROCESSED_DIR/train_key_top200.json" \
  "$PREPROCESSED_DIR/test_key_top200.json"; then
  echo "[env_prepare] Stage traj_sim: running"
  python "$PROJECT_ROOT/traj_sim.py" \
    --dataset_name "$DATASET" \
    --base_model "$BASE_MODEL_DIR" \
    --data_path "$PREPROCESSED_DIR" \
    --output_dir "$SIM_OUTPUT_DIR"
else
  echo "[env_prepare] Stage traj_sim: skipped, outputs already exist"
fi

python - "$PREPROCESSED_DIR" <<'PY'
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])

def load_json(name):
    with (data_dir / name).open() as f:
        return json.load(f)

for split in ("train", "test"):
    kq = load_json(f"{split}_kq_pairs.json")
    top = load_json(f"{split}_key_top200.json")
    total = len(kq)
    covered = len(top)
    ratio = covered / total if total else 0.0
    print(f"[env_prepare] {split} key-query similarity coverage: {covered}/{total} ({ratio:.2%})")
    if total and covered == 0:
        raise SystemExit(
            f"ERROR: no {split} key-query similarities were generated. "
            "Check earlier CUDA OOM errors in traj_sim.py."
        )
    if total and ratio < 0.8:
        print(
            f"WARNING: low {split} key-query similarity coverage. "
            "Prompts without top200 similar trajectories will fall back to same-user history."
        )
PY

if [[ "$FORCE_QA" == "1" ]] || ! files_exist \
  "$PREPROCESSED_DIR/train_qa_pairs_kqt.json" \
  "$PREPROCESSED_DIR/test_qa_pairs_kqt.txt"; then
  echo "[env_prepare] Stage to_nextpoi_kqt: running"
  (
    cd "$PROJECT_ROOT/preprocessing"
    python to_nextpoi_kqt.py -dataset_name "$DATASET"
  )
else
  echo "[env_prepare] Stage to_nextpoi_kqt: skipped, outputs already exist"
fi

# Compatibility path for scripts/readme using datasets/processed/<dataset>/...
mkdir -p "$PROJECT_ROOT/datasets/processed"
PROCESSED_COMPAT_PATH="$PROJECT_ROOT/datasets/processed/$DATASET"
if [[ -L "$PROCESSED_COMPAT_PATH" ]]; then
  rm -f "$PROCESSED_COMPAT_PATH"
fi
if [[ ! -e "$PROCESSED_COMPAT_PATH" ]]; then
  ln -s "../$DATASET/preprocessed" "$PROCESSED_COMPAT_PATH"
elif [[ ! -d "$PROCESSED_COMPAT_PATH" ]]; then
  echo "ERROR: compatibility path exists but is not a directory: $PROCESSED_COMPAT_PATH"
  exit 1
fi

test -f "$PREPROCESSED_DIR/train_qa_pairs_kqt.json"
test -f "$PREPROCESSED_DIR/test_qa_pairs_kqt.txt"
echo "[env_prepare] Done. QA files generated under: $PREPROCESSED_DIR"
