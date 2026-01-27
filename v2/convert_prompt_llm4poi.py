#!/usr/bin/env python3
"""
This script converts a check-in CSV (train/test) into a GSM8K-like parquet where:
  - question: a prompt containing user's historical check-ins + current trajectory context
  - answer: the ground-truth next POI id (PoiId) as a string

It is designed to be reproducible and portable:
  - no hardcoded absolute paths
  - configurable via CLI flags
  - minimal dependencies: pandas, pyarrow, tqdm

Expected input CSV columns (train/test):
  - UserId
  - pseudo_session_trajectory_id
  - UTCTimeOffset (parseable datetime string)
  - PoiId
  - PoiCategoryName
  - Latitude, Longitude (optional; not used in this prompt format)

Example:
  python convert_prompt_llm4poi_repro.py \
    --dataset NYC \
    --train_csv datasets/NYC/train.csv \
    --test_csv datasets/NYC/test.csv \
    --out_dir datasets/NYC \
    --history_limit 50
"""

from __future__ import annotations

import argparse
import ast
import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def normalize_category(value: Any) -> str:
    """
    Extract a human-readable category from multiple possible formats.
    Handles plain strings and Python-literal-like strings such as:
      "[{'url': '/categories/79', 'name': 'Stadium'}]"
    """
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except Exception:
        pass

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return s
        # Try to parse list/dict literals.
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            # Undo common CSV escaping: ""Trader Joe's"" -> "Trader Joe's"
            s2 = s.replace('""', '"')
            try:
                parsed = ast.literal_eval(s2)
            except Exception:
                return s
            return normalize_category(parsed)
        return s

    if isinstance(value, dict):
        for key in ("name", "Name", "category", "Category", "title", "Title"):
            v = value.get(key)
            if v:
                return normalize_category(v)
        # fallback: stringify values
        return ", ".join([normalize_category(v) for v in value.values() if v])

    if isinstance(value, (list, tuple, set)):
        parts = [normalize_category(v) for v in value]
        parts = [p for p in parts if p]
        # de-dup while preserving order
        parts = list(dict.fromkeys(parts))
        return ", ".join(parts) if parts else ""

    return str(value)


def prepare_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("UserId", "pseudo_session_trajectory_id", "UTCTimeOffset", "PoiId", "PoiCategoryName"):
        if col not in df.columns:
            raise ValueError(f"Missing required column {col!r} in {path}")
    df["UTCTimeOffset"] = pd.to_datetime(df["UTCTimeOffset"])
    df.sort_values(["UserId", "UTCTimeOffset", "pseudo_session_trajectory_id"], inplace=True)
    df["category_name"] = df["PoiCategoryName"].apply(normalize_category)
    return df


@dataclass
class HistoryEntry:
    datetime: Any
    time_str: str
    poiid: str
    category: str


def build_history(train_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    history_map[user_id] = {"entries": [HistoryEntry...], "datetimes": [datetime...]}
    Used to fetch user history up to a cutoff time via bisect.
    """
    history: Dict[int, Dict[str, Any]] = {}
    for user_id, group in train_df.groupby("UserId"):
        group = group.sort_values("UTCTimeOffset")
        entries: List[Dict[str, Any]] = []
        datetimes: List[Any] = []
        for _, row in group.iterrows():
            dt = row["UTCTimeOffset"]
            entries.append(
                {
                    "datetime": dt,
                    "time_str": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "poiid": str(row["PoiId"]),
                    "category": row["category_name"],
                }
            )
            datetimes.append(dt)
        history[int(user_id)] = {"entries": entries, "datetimes": datetimes}
    return history


def get_history_entries(history_map: Dict[int, Dict[str, Any]], user_id: int, cutoff_time, limit: int) -> List[Dict[str, Any]]:
    user_history = history_map.get(int(user_id))
    if not user_history:
        return []
    idx = bisect_left(user_history["datetimes"], cutoff_time)
    entries = user_history["entries"][:idx]
    return entries[-limit:]


def format_entries(entries: List[Dict[str, Any]], header: str) -> str:
    if not entries:
        return header + "\nNone.\n"
    lines = [header]
    for e in entries:
        lines.append(f"{e['time_str']} | {e['poiid']} | {e['category']}")
    return "\n".join(lines) + "\n"


def build_samples(
    df: pd.DataFrame,
    history_map: Dict[int, Dict[str, Any]],
    dataset_name: str,
    dataset_split: str,
    history_limit: int,
) -> List[Dict[str, Any]]:
    user_template = (
        "You will be given history and current trajectory data of a user from {dataset}.\n"
        "<history>\n"
        "{history_section_header}\n"
        "{history_section}"
        "</history>\n"
        "<current>\n"
        "The following is the current trajectory of user {user_id}:\n"
        "The trajectories consist of check-in records and each check-in record is represented as a tuple "
        "as (time, poi_id, poi category):\n"
        "{current_section}"
        "</current>\n"
        "Given the data, at {target_time}, which POI will user {user_id} visit?\n"
    )

    samples: List[Dict[str, Any]] = []

    for trajectory_id, group in tqdm(
        df.groupby("pseudo_session_trajectory_id"),
        desc=f"Building {dataset_name} {dataset_split} samples",
        unit="traj",
    ):
        group = group.sort_values("UTCTimeOffset")
        if len(group) < 2:
            continue

        user_id = int(group.iloc[0]["UserId"])
        target_row = group.iloc[-1]
        current_rows = group.iloc[:-1]
        if current_rows.empty:
            continue

        start_time = current_rows.iloc[0]["UTCTimeOffset"]
        history_entries = get_history_entries(history_map, user_id, start_time, limit=history_limit)
        history_header = (
            "Same-User Historical Trajectories (from the same user's past trajectories):\n"
            "Each entry is formatted as (time, poi_id, poi category)."
        )
        history_text = format_entries(history_entries, header="Entries:")

        current_entries: List[Dict[str, Any]] = []
        for _, row in current_rows.iterrows():
            current_entries.append(
                {
                    "time_str": row["UTCTimeOffset"].strftime("%Y-%m-%d %H:%M:%S"),
                    "poiid": str(row["PoiId"]),
                    "category": row["category_name"],
                }
            )
        current_text = format_entries(current_entries, "the most recent entries (time, poi_id, poi category):")

        user_prompt = user_template.format(
            dataset=dataset_name,
            history_section_header=history_header,
            history_section=history_text,
            current_section=current_text,
            user_id=user_id,
            target_time=target_row["UTCTimeOffset"].strftime("%Y-%m-%d %H:%M:%S"),
        )

        target_poiid = str(target_row["PoiId"])
        samples.append(
            {
                "user_prompt": user_prompt,
                "assistant_prompt": target_poiid,
                "ground_truth": target_poiid,
                "user_id": user_id,
                "trajectory_id": int(trajectory_id) if str(trajectory_id).isdigit() else trajectory_id,
                "target_time": target_row["UTCTimeOffset"].strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_split": dataset_split,
            }
        )

    return samples


def write_parquet(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="NYC", help="Dataset name used in prompt text (default: NYC).")
    ap.add_argument("--train_csv", type=Path, required=True)
    ap.add_argument("--test_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--history_limit", type=int, default=50, help="Max number of history entries per prompt (default: 50).")
    ap.add_argument("--write_jsonl", action="store_true", help="Also write raw samples as jsonl for debugging.")
    args = ap.parse_args()

    train_df = prepare_dataframe(args.train_csv)
    test_df = prepare_dataframe(args.test_csv)

    history_map = build_history(train_df)

    train_samples = build_samples(
        train_df, history_map, dataset_name=args.dataset, dataset_split="train", history_limit=args.history_limit
    )
    test_samples = build_samples(
        test_df, history_map, dataset_name=args.dataset, dataset_split="test", history_limit=args.history_limit
    )

    def to_gsm(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"question": s["user_prompt"], "answer": s["assistant_prompt"]} for s in samples]

    gsm_train = to_gsm(train_samples)
    gsm_test = to_gsm(test_samples)

    out_train = args.out_dir / f"{args.dataset.lower()}_gsm8k_train_llm4poi.parquet"
    out_test = args.out_dir / f"{args.dataset.lower()}_gsm8k_test_llm4poi.parquet"
    write_parquet(out_train, gsm_train)
    write_parquet(out_test, gsm_test)
    print(f"[info] wrote: {out_train}")
    print(f"[info] wrote: {out_test}")

    if args.write_jsonl:
        write_jsonl(args.out_dir / f"{args.dataset.lower()}_llm4poi_train_samples.jsonl", train_samples)
        write_jsonl(args.out_dir / f"{args.dataset.lower()}_llm4poi_test_samples.jsonl", test_samples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


