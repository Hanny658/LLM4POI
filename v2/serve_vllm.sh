set -euo pipefail

BASE_MODEL_DIR="${MODEL_DIR:-{your base model/full finetuning checkpoint}}"
ADAPTER_DIR="${ADAPTER_DIR:-{lor adaptor if exist}"
SERVE_NAME="${SERVE_NAME:-sft}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8100}"
DTYPE="${DTYPE:-bfloat16}"          # set to bfloat16 only if your GPU+stack supports it
ENABLE_LORA="${ENABLE_LORA:-0}"    # set to 1 to serve a LoRA adapter
# Batching/throughput knobs (server-side caps). Tune alongside client concurrency.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"                  # e.g. 64 / 128
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"  # e.g. 32768 / 65536

ARGS=(
  --model "${BASE_MODEL_DIR}"
  --served-model-name "${SERVE_NAME}"
  --dtype "${DTYPE}"
  --tensor-parallel-size 4
  --max-model-len 24000
  --host "${HOST}"
  --port "${PORT}"
)

if [[ -n "${MAX_NUM_SEQS}" ]]; then
  ARGS+=( --max-num-seqs "${MAX_NUM_SEQS}" )
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  ARGS+=( --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" )
fi

if [[ "${ENABLE_LORA}" == "1" ]]; then
  ARGS+=( --enable-lora --lora-modules "${SERVE_NAME}=${ADAPTER_DIR}" )
fi

python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
