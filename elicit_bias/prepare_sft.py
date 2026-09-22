"""Flatten every interviewer utterance into the actual SFT training format."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import PACKAGE, read_jsonl, trajectories, write_jsonl


def flatten_trajectory(trajectory):
    rows, history = [], []
    for turn in trajectory["turns"]:
        if turn["role"] == "user":
            rows.append({"g": trajectory["g_label"], "b": trajectory["b_label"],
                         "strategy": turn["strategy"], "history": list(history),
                         "target": turn["content"]})
        history.append({"role": turn["role"], "content": turn["content"]})
    return rows


def load_sft(path):
    path = Path(path)
    rows = read_jsonl(path) if path.suffix == ".jsonl" else [r for _, t in trajectories(path) for r in flatten_trajectory(t)]
    if not rows:
        raise ValueError("Empty SFT data")
    for row in rows:
        if set(row) != {"g", "b", "strategy", "history", "target"}:
            raise ValueError("Expected flattened SFT fields: g,b,strategy,history,target")
        if not all(isinstance(row[k], str) and row[k].strip() for k in ("g", "b", "strategy", "target")):
            raise ValueError("Invalid SFT text")
        if not isinstance(row["history"], list):
            raise ValueError("SFT history must be a message list")
        from .common import validate_turns
        if row["history"]:
            validate_turns(row["history"])
    return rows


def format_for_model(tokenizer, example, *, max_len):
    system = (PACKAGE / "prompts/sft_system.txt").read_text().rstrip("\n").format(**example)
    messages = [{"role": "system", "content": system}, *example["history"]]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    r_ids = list(tokenizer(example["target"], add_special_tokens=False)["input_ids"]) + [tokenizer.eos_token_id]
    ids = (p_ids + r_ids)[-max_len:]
    return {"input_ids": ids, "labels": ([-100] * len(p_ids) + r_ids)[-max_len:], "attention_mask": [1] * len(ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = [r for _, t in trajectories(args.input) for r in flatten_trajectory(t)]
    write_jsonl(args.out, rows)
    print(f"Prepared {len(rows)} interviewer-turn SFT examples")


if __name__ == "__main__":
    main()
