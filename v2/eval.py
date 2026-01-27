#!/usr/bin/env python
""" Evaluate the SFT model served via vLLM on the held-out JSONL dataset.

The script assumes the vLLM server exposes an OpenAI-compatible endpoint.
It sends each conversation in the dataset, captures the model response, and compares
the stripped string directly against the ground-truth assistant message in the
JSONL file. The `--max-new-tokens` flag controls how many tokens the model may
generate for each answer.


The script prints aggregate accuracy and writes per-example details to the
specified output file (or stdout if omitted).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Union, Dict, Any
import re
from tqdm import tqdm

try:
    from openai import AsyncOpenAI, BadRequestError
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit(
        "openai package is required. Install via `pip install openai`."
    ) from exc


def load_messages(jsonl_path: Path) -> List[dict]:
    """Load dataset supporting JSONL or Parquet (with 'messages' column)."""

    if jsonl_path.suffix.lower() == ".parquet":
        import pandas as pd

        df = pd.read_parquet(jsonl_path)
        conversations: List[dict] = []
        if "messages" in df.columns:
            for entry in df["messages"]:
                if isinstance(entry, str):
                    conversations.append(json.loads(entry))
                else:
                    conversations.append(entry)
        else:
            question_col = "question" if "question" in df.columns else df.columns[0]
            answer_col = "answer" if "answer" in df.columns else df.columns[1]
            for _, row in df.iterrows():
                conversations.append(
                    {
                        "messages": [
                            {"role": "user", "content": str(row[question_col])},
                            {"role": "assistant", "content": str(row[answer_col])},
                        ]
                    }
                )
        return conversations

    conversations: List[dict] = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conversations.append(json.loads(line))
    return conversations


def _message_to_text(message: Any) -> str:
    """
    Convert an OpenAI-compatible message content to plain text.
    Supports both legacy string content and newer structured lists
    (e.g., Llama 3.1 responses).
    """
    if message is None:
        return ""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    pieces.append(part["text"])
                elif "content" in part:
                    pieces.append(str(part["content"]))
            else:
                pieces.append(str(part))
        return "".join(pieces)
    return str(content)


def _build_fallback_prompt(system_prompt: str, conversation: List[dict]) -> str:
    """Construct a plain-text prompt for completion-style models."""
    lines: List[str] = []
    if system_prompt:
        lines.append(f"[System]\n{system_prompt}\n")
    for msg in conversation:
        role = msg.get("role")
        content = _message_to_text(msg.get("content"))
        if not content:
            continue
        if role == "assistant":
            lines.append(f"[Assistant]\n{content}\n")
        else:
            lines.append(f"[User]\n{content}\n")
    lines.append("[Assistant]\n")
    return "\n".join(lines)


async def _evaluate_one(
    client: AsyncOpenAI,
    record: dict,
    args: argparse.Namespace,
) -> dict:
    messages = record["messages"]

    # The last assistant message in the dataset is treated as ground truth.
    gt_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    if not gt_messages:
        raise ValueError("No assistant answer found in dataset entry.")
    gold_text = _message_to_text(gt_messages[-1].get("content")).strip()
    if args.eos_token and gold_text.endswith(args.eos_token):
        gold_text = gold_text[: -len(args.eos_token)].rstrip()

    # Prepare request messages by removing the ground-truth assistant replies.
    prompt_messages = [msg for msg in messages if msg.get("role") != "assistant"]
    if prompt_messages and prompt_messages[0].get("role") == "system":
        prompt_messages = prompt_messages[1:]
    prompt_messages = [{"role": "system", "content": args.system_prompt}] + prompt_messages

    try:
        response = await client.chat.completions.create(
            model=args.model,
            messages=prompt_messages,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        pred_text = _message_to_text(response.choices[0].message).strip()
    except BadRequestError:
        fallback_prompt = _build_fallback_prompt(
            args.system_prompt,
            prompt_messages[1:],  # exclude injected system for user/assistant log
        )
        completion = await client.completions.create(
            model=args.model,
            prompt=fallback_prompt,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        pred_text = (completion.choices[0].text or "").strip()

    if args.eos_token and pred_text.endswith(args.eos_token):
        pred_text = pred_text[: -len(args.eos_token)].rstrip()


    pred_text = pred_text.split("<")[0]
    gold_text = gold_text.split("<")[0]

    is_correct = pred_text == gold_text
    return {
        "prompt": prompt_messages,
        "gold_response": gold_text,
        "prediction": pred_text,
        "correct": is_correct,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to evaluation dataset (JSONL or Parquet)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="dummy", help="API key for the OpenAI-compatible server")
    parser.add_argument("--model", default="qwen2.5-nyc-sft", help="Model name exposed by the vLLM server")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL file to store predictions")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional limit on number of examples")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="How many requests to run concurrently (enables vLLM continuous batching).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--eos-token",
        default="<|end_of_text|>",
        help="Trailing token to strip from gold/predicted text before comparison (empty string to disable)",
    )
    parser.add_argument(
        "--system_prompt",
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        help="System message prepended to every conversation",
    )

    args = parser.parse_args(argv)

    data = load_messages(args.dataset)
    if args.max_examples is not None:
        data = data[: args.max_examples]

    async def _run() -> List[dict]:
        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
        sem = asyncio.Semaphore(max(1, int(args.concurrency)))
        out: List[Optional[dict]] = [None] * len(data)

        async def _guarded(i: int, rec: dict) -> tuple[int, dict]:
            async with sem:
                return (i, await _evaluate_one(client, rec, args))

        tasks = [asyncio.create_task(_guarded(i, rec)) for i, rec in enumerate(data)]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating", unit="example"):
            i, r = await fut
            out[i] = r
        return [r for r in out if r is not None]

    results = asyncio.run(_run())
    correct = sum(int(r["correct"]) for r in results)

    total = len(results)
    accuracy = correct / total if total else 0.0
    print(f"Examples evaluated: {total}")
    print(f"Top-1 accuracy: {accuracy:.4f} ({correct}/{total})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fout:
            for item in results:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Detailed predictions written to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

