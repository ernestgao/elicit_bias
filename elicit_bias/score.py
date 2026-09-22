"""Versioned scoring to a separate, resumable JSONL with a durable attempt cap."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from .cloud import CloudClient, CloudError
from .common import EVALUATION_JUDGES, digest, read_jsonl, trajectories, validate_judges
from .rubrics import make_prompt, parse


def append(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class Scorer:
    """Sequential by design: at most one request per durable reservation."""
    def __init__(self, output, clients, cap=1200, per_key=3):
        validate_judges(clients)
        self.output, self.clients, self.cap, self.per_key = Path(output), clients, cap, per_key
        self.ledger = self.output.with_suffix(".attempts.jsonl")
        starts = read_jsonl(self.ledger) if self.ledger.exists() else []
        if starts:
            validate_judges(sorted({row["backend"] for row in starts}))
        self.count = len(starts)
        self.attempts = Counter(r["cache_key"] for r in starts)
        rows = read_jsonl(self.output) if self.output.exists() else []
        if rows:
            validate_judges(sorted({row["backend"] for row in rows}))
        self.cache = {r["cache_key"]: r for r in rows if r["status"] == "validated"}

    def work(self, job):
        validate_judges([job["backend"]])
        key = job["cache_key"]
        if key in self.cache:
            parse(self.cache[key]["raw_text"], job["turns"], job["rubric"])
            return "cached"
        while self.count < self.cap and self.attempts[key] < self.per_key:
            self.count += 1
            self.attempts[key] += 1
            append(self.ledger, {"attempt": self.count, "cache_key": key, "backend": job["backend"]})
            row = {k: v for k, v in job.items() if k not in ("prompt", "turns")}
            row.update(attempt=self.count, status="failed", raw_text="", usage={})
            started = time.monotonic()
            try:
                raw, usage = self.clients[job["backend"]].complete(
                    [{"role": "user", "content": job["prompt"]}])
                row.update(raw_text=raw, usage=usage)
                row["verdict"] = parse(raw, job["turns"], job["rubric"])
                row["status"] = "validated"
            except (ValueError, CloudError) as exc:
                row["error_type"] = type(exc).__name__
                if isinstance(exc, CloudError):
                    row["http_status"] = exc.status
            row["elapsed_seconds"] = time.monotonic() - started
            append(self.output, row)
            if row["status"] == "validated":
                self.cache[key] = row
                return "new_success"
            if row.get("http_status") in (400, 401, 403):
                raise CloudError("Service rejected the request; collection stopped", row["http_status"])
        # Keep unattempted/capped endpoints visible to offline denominators.
        row = {k: v for k, v in job.items() if k not in ("prompt", "turns")}
        row.update(status="unresolved", error_type="attempt_limit", raw_text="", usage={})
        append(self.output, row)
        return "unresolved"


def make_jobs(records, clients, rubric, cuts):
    validate_judges(clients)
    jobs = []
    for identity, record in records:
        conditions = {"true_group": record["g_label"]}
        for condition, group in conditions.items():
            for cut in cuts:
                prompt, turns, template = make_prompt(record, cut, group, rubric)
                for backend, client in clients.items():
                    meta = {"id": identity, "target": record.get("target", "unspecified"),
                            "g": record["g"], "b": record["b"], "strategy": record["strategy"],
                            "source_sha256": digest(json.dumps(record, sort_keys=True, ensure_ascii=False)),
                            "rubric": rubric, "rubric_sha256": digest(template), "prompt_sha256": digest(prompt),
                            "cut": cut, "condition": condition, "evaluated_group": group,
                            "backend": backend, "model": client.model,
                            "service_fingerprint": digest(client.base)}
                    meta["cache_key"] = digest(json.dumps(meta, sort_keys=True))
                    jobs.append({**meta, "prompt": prompt, "turns": turns})
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rubric", choices=["original"], default="original")
    p.add_argument("--judges", choices=EVALUATION_JUDGES, nargs="+", default=list(EVALUATION_JUDGES))
    p.add_argument("--cuts", type=int, nargs="+", default=[1, 2, 4, 6])
    p.add_argument("--max-attempts", type=int, default=1200)
    p.add_argument("--per-key-attempts", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        validate_judges(args.judges, require_pair=True)
    except ValueError as exc:
        p.error(str(exc))
    if min(args.max_attempts, args.per_key_attempts) < 1 or len(set(args.judges)) != len(args.judges) or len(set(args.cuts)) != len(args.cuts):
        p.error("Invalid caps or duplicate judges/cuts")
    records = trajectories(args.input)
    # Validate every cut before configuring services or sending any request.
    for _, record in records:
        for cut in args.cuts:
            make_prompt(record, cut, record["g_label"], args.rubric)
    if args.dry_run:
        print(json.dumps({"conversations": len(records), "endpoints": len(records) * len(args.cuts) * len(args.judges), "api_calls": 0}))
        return
    clients = {j: CloudClient(j) for j in args.judges}
    jobs = make_jobs(records, clients, args.rubric, args.cuts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = args.out.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        p.error("Scoring lock exists; confirm no live writer before removing a stale lock")
    os.close(fd)
    try:
        scorer = Scorer(args.out, clients, args.max_attempts, args.per_key_attempts)
        counts = Counter()
        for index, job in enumerate(jobs, 1):
            counts[scorer.work(job)] += 1
            if index % 10 == 0:
                print(json.dumps({"processed": index, "attempts": scorer.count, **counts}), flush=True)
        print(json.dumps({"requested": len(jobs), "attempts": scorer.count, **counts}))
        if counts["unresolved"]:
            raise SystemExit(2)
    finally:
        lock.unlink()


if __name__ == "__main__":
    try:
        main()
    except CloudError as exc:
        raise SystemExit(str(exc)) from None
