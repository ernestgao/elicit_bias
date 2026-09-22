"""LoRA all-turn interviewer SFT; masks history and trains only the next utterance."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from .prepare_sft import load_sft, format_for_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bsz", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--dry-run", action="store_true", help="Validate data without loading models")
    args = p.parse_args()
    rows = load_sft(args.data)
    if args.dry_run:
        print(f"Validated {len(rows)} SFT examples; no model loaded")
        return
    if Path(args.out).exists():
        p.error("Output already exists; choose a new adapter directory")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments,
                              DataCollatorForSeq2Seq)

    token = os.environ.get("HF_TOKEN") or False
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = Dataset.from_list(rows).map(
        lambda ex: format_for_model(tokenizer, ex, max_len=args.max_len),
        remove_columns=["g", "b", "strategy", "history", "target"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", token=token)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True, logging_steps=20, save_steps=500,
        save_total_limit=2,
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=4,
    )

    Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    ).train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()
