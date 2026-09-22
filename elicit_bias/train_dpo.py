"""Opener-only DPO (Rafailov et al., 2023).

Merge the frozen SFT adapter into the base, then train a new LoRA adapter.
Disabling that adapter gives the SFT reference. The loss uses response-token
log-probability sums, not length-normalized scores or full-trajectory tokens.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


class DPODataset:
    def __init__(self, path: Path, tokenizer, max_prompt_len: int, max_total_len: int):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_total_len = max_total_len
        self.rows: list[DPOExample] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            msgs = json.loads(r["prompt"])
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            self.rows.append(DPOExample(
                prompt=prompt_text, chosen=r["chosen"], rejected=r["rejected"]))

    def __len__(self): return len(self.rows)

    def _encode_pair(self, prompt: str, response: str) -> dict[str, torch.Tensor]:
        import torch
        # tokenize prompt + response, mask labels on prompt portion
        p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        r_ids = r_ids + [self.tokenizer.eos_token_id]

        # truncate prompt from the LEFT to keep generation prompt suffix intact
        if len(p_ids) > self.max_prompt_len:
            p_ids = p_ids[-self.max_prompt_len:]
        # then truncate response from the RIGHT if needed
        budget = self.max_total_len - len(p_ids)
        if budget < 4:
            r_ids = r_ids[:4]
        elif len(r_ids) > budget:
            r_ids = r_ids[:budget]

        input_ids = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
            "prompt_len": len(p_ids),
        }

    def __getitem__(self, i):
        ex = self.rows[i]
        return {
            "chosen":   self._encode_pair(ex.prompt, ex.chosen),
            "rejected": self._encode_pair(ex.prompt, ex.rejected),
        }


def collate(batch, pad_id: int):
    """Pad a list of {chosen, rejected} dicts to a single tensor batch."""
    import torch
    import torch.nn.functional as F
    def stack(side: str):
        ids = [b[side]["input_ids"] for b in batch]
        lbl = [b[side]["labels"]    for b in batch]
        max_len = max(t.size(0) for t in ids)
        pads = lambda t, fill: F.pad(t, (0, max_len - t.size(0)), value=fill)
        return {
            "input_ids":      torch.stack([pads(t, pad_id) for t in ids]),
            "attention_mask": torch.stack([pads(torch.ones_like(t), 0) for t in ids]),
            "labels":         torch.stack([pads(t, -100) for t in lbl]),
        }
    return {"chosen": stack("chosen"), "rejected": stack("rejected")}


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------
def _logp_response(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum log p(token | prefix) over the *response* (label != -100) tokens.
    Returns a (B,) tensor."""
    import torch.nn.functional as F
    out = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = (labels != -100)
    safe_lbl = labels.clone()
    safe_lbl[~mask] = 0
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, safe_lbl.unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * mask
    return tok_logp.sum(dim=-1)


def dpo_loss(policy_logps_chosen, policy_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta: float):
    import torch.nn.functional as F
    pi_logratios = policy_logps_chosen - policy_logps_rejected
    ref_logratios = ref_logps_chosen - ref_logps_rejected
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (policy_logps_chosen - ref_logps_chosen).detach()
    rejected_reward = beta * (policy_logps_rejected - ref_logps_rejected).detach()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, chosen_reward.mean(), rejected_reward.mean(), accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--sft-adapter", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bsz", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--max-total-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    from .build_preferences import validate_pairs
    rows = validate_pairs(args.data)
    if args.dry_run:
        print(f"Validated {len(rows)} opener DPO pairs; no model loaded")
        return
    if args.out.exists():
        p.error("Output already exists; choose a new adapter directory")
    if args.max_total_len < args.max_prompt_len + 4:
        p.error("--max-total-len must exceed --max-prompt-len by at least four")
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)


    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"

    token = os.environ.get("HF_TOKEN") or False
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # --- model -----------------------------------------------------------
    # Strategy: merge the SFT LoRA into the base weights (so SFT becomes
    # permanent), then add a fresh trainable DPO LoRA. With this layout:
    #   policy             = sft_baked_base + dpo_adapter (active)
    #   reference (no-grad)= sft_baked_base                 (disable_adapter())
    print("[dpo] loading base ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", token=token)

    print("[dpo] mounting SFT adapter and merging into base ...")
    sft_loaded = PeftModel.from_pretrained(
        base, args.sft_adapter, is_trainable=False, token=token)
    merged = sft_loaded.merge_and_unload()    # bake SFT into base weights
    merged.gradient_checkpointing_enable()
    merged.enable_input_require_grads()

    dpo_lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    policy = get_peft_model(merged, dpo_lora)
    policy.print_trainable_parameters()

    # --- data ------------------------------------------------------------
    ds = DPODataset(args.data, tokenizer,
                    max_prompt_len=args.max_prompt_len,
                    max_total_len=args.max_total_len)
    print(f"[dpo] dataset rows: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.bsz, shuffle=True,
                        collate_fn=lambda b: collate(b, pad_id))

    # --- optim -----------------------------------------------------------
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr,
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    if len(loader) < args.grad_accum:
        p.error("Too few microbatches; reduce --grad-accum for a format-sample run")
    total_macro = math.ceil(len(loader) / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=int(args.warmup_ratio * total_macro),
        num_training_steps=total_macro)

    # --- train -----------------------------------------------------------
    log_fh = log_path.open("w")
    macro = 0
    accumulated_loss = 0.0
    accumulated_acc = 0.0
    accumulated_n = 0

    t0 = time.time()
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            for side in ("chosen", "rejected"):
                for k, v in batch[side].items():
                    batch[side][k] = v.to(policy.device)

            # policy log-probs (DPO adapter active by default)
            policy.train()
            pi_chosen = _logp_response(policy, batch["chosen"])
            pi_rejected = _logp_response(policy, batch["rejected"])

            # reference log-probs: DPO adapter disabled → equals the
            # SFT-baked base = π_SFT.
            with torch.no_grad(), policy.disable_adapter():
                policy.eval()
                ref_chosen = _logp_response(policy, batch["chosen"])
                ref_rejected = _logp_response(policy, batch["rejected"])
            policy.train()

            loss, r_chosen, r_rejected, acc = dpo_loss(
                pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=args.beta)
            # Average the final partial accumulation group over its actual size.
            group_start = (i // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(loader) - group_start)
            (loss / group_size).backward()

            accumulated_loss += loss.item()
            accumulated_acc += acc.item()
            accumulated_n += 1

            if (i + 1) % args.grad_accum == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                macro += 1

                msg = {
                    "step": macro, "epoch": epoch + (i + 1) / len(loader),
                    "loss": accumulated_loss / accumulated_n,
                    "accuracy": accumulated_acc / accumulated_n,
                    "r_chosen": r_chosen.item(),
                    "r_rejected": r_rejected.item(),
                    "lr": sched.get_last_lr()[0],
                    "wall": time.time() - t0,
                }
                log_fh.write(json.dumps(msg) + "\n")
                log_fh.flush()
                if macro % 5 == 0:
                    print(f"[dpo] step {macro:4d}  loss {msg['loss']:.4f}  "
                          f"acc {msg['accuracy']:.3f}  "
                          f"r_chosen {msg['r_chosen']:+.3f}  "
                          f"r_rej {msg['r_rejected']:+.3f}  "
                          f"lr {msg['lr']:.2e}")
                accumulated_loss = 0.0
                accumulated_acc = 0.0
                accumulated_n = 0

                if macro % args.save_every == 0:
                    ckpt = args.out / f"checkpoint-{macro}"
                    policy.save_pretrained(ckpt.as_posix())

    # final save (only the trainable DPO adapter; SFT is baked into base
    # so it's not stored here — to recreate the policy you need both
    # the SFT adapter AND this adapter, applied in that order).
    policy.save_pretrained(args.out.as_posix())
    tokenizer.save_pretrained(args.out.as_posix())
    log_fh.close()
    print(f"[dpo] done. saved to {args.out}")


if __name__ == "__main__":
    main()
