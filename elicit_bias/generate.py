"""Generate synthetic demonstrations or learned-interviewer conversations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re

from .common import PACKAGE, interviewer_messages, read_jsonl, taxonomy
from .cloud import CloudClient, CloudError

REFUSAL_RE = re.compile(
    r"\b(I (?:can'?t|cannot|won'?t|will not|am unable|am not able|don'?t feel comfortable|"
    r"shouldn'?t|must decline)|I'm sorry, but|I cannot (?:help|assist|provide)|"
    r"I won'?t (?:engage|provide)|As an AI|I (?:am|'m) not (?:going|comfortable|able))", re.I)


class LocalModel:
    def __init__(self, model_name, sft=None, dpo=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        token = os.environ.get("HF_TOKEN") or False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", token=token)
        if dpo and not sft:
            raise ValueError("DPO inference requires its SFT adapter as well")
        if sft:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, sft, is_trainable=False, token=token)
            if dpo:
                model = model.merge_and_unload()
                model = PeftModel.from_pretrained(model, dpo, is_trainable=False, token=token)
        self.model = model.eval()

    def complete(self, messages, *, temperature, max_tokens):
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            tokens = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                         temperature=temperature, top_p=.9, repetition_penalty=1.05,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def rollout(ask_interviewer, ask_target, *, group, bias, strategy, turns=6, seed=None, synthesis=False):
    history, recorded = [], []
    target_system = (PACKAGE / "prompts" / ("target_synthesis.txt" if synthesis else "target.txt")).read_text().rstrip("\n")
    for _ in range(turns):
        question = ask_interviewer(interviewer_messages(history, group["label"], bias["label"], strategy,
                                                        seed["question"] if seed else None))
        if not question.strip():
            raise ValueError("Empty interviewer response")
        history.append({"role": "user", "content": question})
        recorded.append({**history[-1], "strategy": strategy})
        answer = ask_target([{"role": "system", "content": target_system}, *history])
        if not answer.strip():
            raise ValueError("Empty target response")
        history.append({"role": "assistant", "content": answer})
        recorded.append({**history[-1], "refused": bool(REFUSAL_RE.search(answer))})
    return {"g": group["id"], "g_label": group["label"], "b": bias["id"], "b_label": bias["label"],
            "strategy": strategy, "seed_qid": seed.get("qid") if seed else None, "turns": recorded}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interviewer", choices=["local", "mimo"], required=True)
    p.add_argument("--model", help="Interviewer base model path or public model ID")
    p.add_argument("--sft-adapter")
    p.add_argument("--dpo-adapter")
    p.add_argument("--target", choices=["local", "azure", "mimo", "compatible"], required=True)
    p.add_argument("--target-model", help="Required for a local target; loaded separately without interviewer adapters")
    p.add_argument("--target-label", required=True, help="Public target label, never a service address")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seeds", type=Path, help="Optional JSONL: g,b,question,qid; not bundled")
    p.add_argument("--groups", nargs="+")
    p.add_argument("--biases", nargs="+")
    p.add_argument("--strategies", nargs="+")
    p.add_argument("--k-per-cell", type=int, default=2)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--max-new-user", type=int, default=220)
    p.add_argument("--max-new-target", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-api-calls", type=int, default=1200)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.interviewer == "local" and not args.model:
        p.error("--model is required for a local interviewer")
    if args.dpo_adapter and not args.sft_adapter:
        p.error("--dpo-adapter requires --sft-adapter")
    if args.target == "local" and not args.target_model:
        p.error("--target-model is required; there is no implicit target substitution")
    if min(args.turns, args.k_per_cell, args.max_api_calls) < 1 or (args.limit is not None and args.limit < 1):
        p.error("Counts must be positive")
    tax = taxonomy()
    groups = [g for family in tax["demographics"].values() for g in family]
    for selected, available in ((args.groups, groups), (args.biases, tax["bias_categories"]), (args.strategies, tax["strategies"])):
        if selected and not set(selected) <= {x["id"] for x in available}:
            p.error("Unknown taxonomy selection")
    jobs = [(g, b, s["id"], k) for g in groups for b in tax["bias_categories"] for s in tax["strategies"]
            if (not args.groups or g["id"] in args.groups) and (not args.biases or b["id"] in args.biases)
            and (not args.strategies or s["id"] in args.strategies) for k in range(args.k_per_cell)]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    expected_calls = len(jobs) * args.turns * (int(args.interviewer != "local") + int(args.target != "local"))
    if args.dry_run:
        print(json.dumps({"trajectories": len(jobs), "planned_api_calls": expected_calls, "models_loaded": False}))
        return
    if expected_calls > args.max_api_calls:
        p.error("Requested generation exceeds --max-api-calls")
    if args.out.exists():
        p.error("Choose a new output directory; existing conversations are never overwritten")
    seeds = read_jsonl(args.seeds) if args.seeds else []
    rng = random.Random(args.seed)
    teacher = CloudClient("mimo", role="interviewer") if args.interviewer == "mimo" else None
    remote_target = CloudClient(args.target, role="target") if args.target != "local" else None
    if args.interviewer == "local" or args.target == "local":
        import torch
        torch.manual_seed(args.seed)
    actor = LocalModel(args.model, args.sft_adapter, args.dpo_adapter) if not teacher else None
    target = LocalModel(args.target_model) if not remote_target else None
    ask_interviewer = (lambda m: teacher.complete(m, temperature=.9)[0]) if teacher else (lambda m: actor.complete(m, temperature=.8, max_tokens=args.max_new_user))
    ask_target = (lambda m: remote_target.complete(m, temperature=.7, max_tokens=None if args.target == "mimo" else args.max_new_target)[0]) if remote_target else (lambda m: target.complete(m, temperature=.7, max_tokens=args.max_new_target))
    args.out.mkdir(parents=True)
    for g, b, strategy, k in jobs:
        eligible = [s for s in seeds if s["g"] == g["id"] and s["b"] == b["id"]]
        record = rollout(ask_interviewer, ask_target, group=g, bias=b, strategy=strategy, turns=args.turns,
                         seed=rng.choice(eligible) if eligible else None, synthesis=bool(teacher))
        record.update(target=args.target_label, rollout_index=k)
        path = args.out / f"{g['id']}__{b['id']}__{strategy}__{k}.json"
        with path.open("x") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(jobs)} conversations; scoring is a separate explicit step")


if __name__ == "__main__":
    try:
        main()
    except CloudError as exc:
        raise SystemExit(str(exc)) from None
