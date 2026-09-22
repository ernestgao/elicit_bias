"""Trajectory-labeled opener preferences, not full-trajectory DPO."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .common import interviewer_messages, read_jsonl, trajectories, write_jsonl
from .metrics import valid_score


def validate_pairs(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Empty DPO data")
    for row in rows:
        messages = json.loads(row["prompt"])
        if not isinstance(messages, list) or [m["role"] for m in messages] != ["system", "user"]:
            raise ValueError("Expected a JSON-encoded two-message opener prompt")
        if not all(isinstance(m.get("content"), str) and m["content"] for m in messages):
            raise ValueError("Empty DPO prompt")
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("chosen", "rejected")):
            raise ValueError("Missing opener response")
        if row["chosen"] == row["rejected"]:
            raise ValueError("Identical preference responses")
        if "chosen_score" in row:
            gap = valid_score(row["chosen_score"]) - valid_score(row["rejected_score"])
            if gap <= 0 or abs(gap - row["score_margin"]) > 1e-8:
                raise ValueError("Inconsistent preference margin")
    return rows


def reward(record):
    """MiMo MAX from the original recorded preference-training scores."""
    # Source cuts can use message indices; preserve their recorded semantics.
    rows = record.get("judge_scores", [])
    if not rows:
        raise ValueError("MiMo reward requires source judge_scores")
    return max(valid_score(row.get("score")) for row in rows)


def build(records, target, margin=1):
    if margin < 1:
        raise ValueError("Preference margin must be at least 1")
    cells = defaultdict(list)
    for identity, record in records:
        cells[(record["g"], record["b"], record["strategy"])].append((identity, record))
    output = []
    for cell, candidates in sorted(cells.items()):
        if len(candidates) != 2:
            raise ValueError("Preference construction requires two candidates per cell")
        scored = sorted([(reward(r), key, r) for key, r in candidates], key=lambda x: (x[0], x[1]))
        low, high = scored[0], scored[-1]
        if high[0] - low[0] < margin:
            continue
        rejected, chosen = low[2], high[2]
        if chosen["turns"][0]["content"] == rejected["turns"][0]["content"]:
            continue
        output.append({"prompt": json.dumps(interviewer_messages([], chosen["g_label"], chosen["b_label"], cell[2]), ensure_ascii=False),
                       "chosen": chosen["turns"][0]["content"], "rejected": rejected["turns"][0]["content"],
                       "g": cell[0], "b": cell[1], "strategy": cell[2], "target": target,
                       "chosen_score": high[0], "rejected_score": low[0], "score_margin": high[0] - low[0],
                       "reward_definition": "mimo_max"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True, help="Public target labels, one per directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reward", choices=["recorded_single"], default="recorded_single",
                        help="MiMo MAX from the supplied judge_scores")
    parser.add_argument("--min-margin", type=float, default=1)
    args = parser.parse_args()
    if len(args.input_dirs) != len(args.targets) or args.min_margin < 1:
        parser.error("Provide one target label per directory and a margin of at least 1")
    rows = []
    for path, target in zip(args.input_dirs, args.targets):
        records = trajectories(path)
        rows.extend(build(records, target, args.min_margin))
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} trajectory-labeled opener preferences")


if __name__ == "__main__":
    main()
