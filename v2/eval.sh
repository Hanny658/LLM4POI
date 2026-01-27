#!/usr/bin/env bash
set -euo pipefail

CONCURRENCY="${CONCURRENCY:-{adjust the number according to you gpu memory}"

python eval.py \
  --base-url http://127.0.0.1:8100/v1 \
  --api-key dummy \
  --model sft-lora \
  --dataset {Your dataset path} \
  --output {your output path} \
  --system_prompt "You are a helpful assistant." \
  --max-new-tokens 36 \
  --concurrency "${CONCURRENCY}" \
  --temperature 0.2

 
