"""Offline fixed-cut, distribution, paired-control and budget comparisons."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics as stats

from .common import EVALUATION_JUDGES, read_jsonl, trajectories, validate_judges
from .metrics import METRICS, distribution, metric, summary, valid_score


def load_scores(path, rubric):
    """Accept this package's JSONL or original cached trajectory JSON files."""
    path = Path(path)
    if path.is_dir() or path.suffix == ".json":
        if rubric != "original":
            raise ValueError("Trajectory judge_scores_v2 inputs use the original rubric only")
        rows = []
        for identity, record in trajectories(path):
            backends = record.get("judge_scores_v2", {})
            if backends:
                validate_judges(backends)
            for backend, entries in backends.items():
                for entry in entries:
                    cut = entry.get("exchange_cut")
                    if type(cut) is not int or entry.get("turn_cut") != 2 * cut:
                        raise ValueError("Scored trajectories need unambiguous exchange_cut and turn_cut")
                    rows.append({"id": identity, "target": record.get("target", "unspecified"),
                                 "g": record["g"], "b": record["b"], "strategy": record["strategy"],
                                 "backend": backend, "condition": "true_group", "cut": cut,
                                 "rubric": rubric, "rubric_sha256": "trajectory-original-rubric",
                                 "status": "validated" if entry.get("score") is not None else "failed",
                                 "verdict": entry})
    else:
        rows = read_jsonl(path)
    if not rows or any(row["rubric"] != rubric for row in rows):
        raise ValueError("Empty data or mixed rubric versions; analyze versions separately")
    validate_judges(sorted({row["backend"] for row in rows}))
    if len({r["rubric_sha256"] for r in rows}) != 1:
        raise ValueError("Mixed prompt versions are not an analysis condition")
    indexed = {}
    for row in rows:
        key = (row["target"], row["id"], row["backend"], row["condition"], row["cut"])
        if row["status"] == "validated":
            valid_score(row["verdict"]["score"])
            if key in indexed and indexed[key]["status"] == "validated":
                raise ValueError("Duplicate successful endpoint; do not concatenate repeated scoring versions")
            indexed[key] = row
        elif key not in indexed:
            indexed[key] = row
    return list(indexed.values())


def packed(rows, condition="true_group"):
    samples = {}
    for row in rows:
        if row["condition"] != condition:
            continue
        key = (row["target"], row["id"])
        sample = samples.setdefault(key, {"g": row["g"], "b": row["b"], "strategy": row["strategy"], "judge_scores_v2": {}})
        if (sample["g"], sample["b"], sample["strategy"]) != (row["g"], row["b"], row["strategy"]):
            raise ValueError("Identity/taxonomy mismatch")
        if row["status"] == "validated":
            sample["judge_scores_v2"].setdefault(row["backend"], []).append(
                {"exchange_cut": row["cut"], "turn_cut": 2 * row["cut"], **row["verdict"]})
    return samples


def values(samples, judges, name):
    output = {}
    for key, record in samples.items():
        try:
            output[key] = metric(record, judges, name)
        except ValueError:
            pass  # Missing endpoints are reported by the caller, never zero-filled.
    return output


def judge_sets(judges):
    validate_judges(judges)
    return [(j, [j]) for j in judges] + ([("ensemble", judges)] if len(judges) > 1 else [])


def subsets(samples, by_bias=False):
    targets = sorted({key[0] for key in samples})
    for target in targets + (["pooled"] if len(targets) > 1 else []):
        ids = [key for key in samples if target == "pooled" or key[0] == target]
        for bias in ["overall"] + (sorted({samples[key]["b"] for key in ids}) if by_bias else []):
            yield target, bias, [key for key in ids if bias == "overall" or samples[key]["b"] == bias]


def cluster(samples, keys):
    return [(samples[key]["g"], samples[key]["b"]) for key in keys]


def summarize_scores(rows, judges, metrics, seed, by_bias=False):
    samples, output = packed(rows), []
    for name in metrics:
        for label, js in judge_sets(judges):
            scores = values(samples, js, name)
            for target, bias, requested in subsets(samples, by_bias):
                keys = [key for key in requested if key in scores]
                vals = [scores[key] for key in keys]
                output.append({"target": target, "bias": bias, "judge": label, "metric": name,
                               "expected_n": len(requested), "missing_n": len(requested) - len(keys),
                               **summary(vals, cluster(samples, keys), seed), "distribution": distribution(vals)})
    return output


def paired(left, right, judges, metrics, seed, by_bias=False):
    a, b, output = packed(left), packed(right), []
    requested_samples = {**b, **a}
    for key in a.keys() & b.keys():
        if (a[key]["g"], a[key]["b"], a[key]["strategy"]) != (b[key]["g"], b[key]["b"], b[key]["strategy"]):
            raise ValueError("Paired taxonomy mismatch")
    for name in metrics:
        for label, js in judge_sets(judges):
            av, bv = values(a, js, name), values(b, js, name)
            for target, bias, requested in subsets(requested_samples, by_bias):
                keys = [key for key in requested if key in av and key in bv]
                diffs = [av[key] - bv[key] for key in keys]
                output.append({"target": target, "bias": bias, "judge": label, "metric": name,
                               "difference": "left_minus_right", "expected_n": len(requested),
                               "missing_pair_n": len(requested) - len(keys),
                               "left_mean": stats.mean(av[k] for k in keys) if keys else None,
                               "right_mean": stats.mean(bv[k] for k in keys) if keys else None,
                               **summary(diffs, cluster(requested_samples, keys), seed)})
    return output


def trajectory_changes(rows, judges, seed):
    """First-to-later score changes and numerical severity bands."""
    samples, output = packed(rows), []
    bands = ("zero", "minor", "moderate", "severe")
    def band(value):
        return "zero" if value == 0 else "minor" if value < 4 else "moderate" if value < 7 else "severe"
    for label, js in judge_sets(judges):
        one = values(samples, js, "cut1")
        for name in ("cut6", "realizable_max"):
            later = values(samples, js, name)
            for target, _, requested in subsets(samples):
                keys = [k for k in requested if k in one and k in later]
                changes = [later[k] - one[k] for k in keys]
                cells = [{"start_band": a, "end_band": b,
                          "n": sum(band(one[k]) == a and band(later[k]) == b for k in keys)}
                         for a in bands for b in bands]
                output.append({"target": target, "judge": label, "comparison": name + "_minus_cut1",
                               "expected_n": len(requested), "missing_pair_n": len(requested) - len(keys),
                               **summary(changes, cluster(samples, keys), seed), "severity_transitions": cells})
    return output


def bestof(multi, candidates, judges, seed, multi_index=0):
    mt, singles = packed(multi), packed(candidates)
    out = []
    def ordinal(key):
        match = re.search(r"__(\d+)$", key[1])
        if not match:
            raise ValueError("Best-of inputs need IDs ending in __candidate_index")
        return int(match.group(1))
    for label, js in judge_sets(judges):
        cv = values(singles, js, "cut1")
        bank = defaultdict(dict)
        for key, value in cv.items():
            r = singles[key]
            cell = (key[0], r["g"], r["b"], r["strategy"])
            if ordinal(key) in bank[cell]:
                raise ValueError("Duplicate one-turn candidate")
            bank[cell][ordinal(key)] = value
        for name in ("realizable_max", "cut6"):
            mv = values(mt, js, name)
            for n in (3, 6):
                for target in sorted({k[0] for k in mt}):
                    diffs, clusters, expected = [], [], 0
                    for key, record in mt.items():
                        if key[0] != target or ordinal(key) != multi_index:
                            continue
                        expected += 1
                        cell = (target, record["g"], record["b"], record["strategy"])
                        # First three of the frozen six candidates, not the best three chosen later.
                        if key not in mv or not set(range(6)) <= bank[cell].keys():
                            continue
                        diffs.append(mv[key] - max(bank[cell][i] for i in range(n)))
                        clusters.append((record["g"], record["b"]))
                    out.append({"target": target, "judge": label, "metric": name, "best_of": n,
                                "expected_n": expected, "missing_pair_n": expected - len(diffs),
                                "difference": "multi_minus_best_of_n", **summary(diffs, clusters, seed)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["summary", "trajectory", "paired", "bestof"])
    p.add_argument("--input", type=Path, required=True, help="Scoring JSONL or cached trajectory directory")
    p.add_argument("--other", type=Path, help="Right-hand paired condition, or six-candidate one-turn scores")
    p.add_argument("--rubric", choices=["original"], default="original")
    p.add_argument("--judges", choices=EVALUATION_JUDGES, nargs="+", default=list(EVALUATION_JUDGES))
    p.add_argument("--metrics", choices=METRICS, nargs="+", default=["cut1", "cut6", "realizable_max"])
    p.add_argument("--by-bias", action="store_true")
    p.add_argument("--multi-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        validate_judges(args.judges, require_pair=True)
    except ValueError as exc:
        p.error(str(exc))
    rows = load_scores(args.input, args.rubric)
    if args.kind == "summary":
        results = summarize_scores(rows, args.judges, args.metrics, args.seed, args.by_bias)
    elif args.kind == "trajectory":
        results = trajectory_changes(rows, args.judges, args.seed)
    else:
        if not args.other:
            p.error("--other is required")
        other = load_scores(args.other, args.rubric)
        if {r["rubric_sha256"] for r in rows} != {r["rubric_sha256"] for r in other}:
            p.error("Both paired conditions must use the same rubric version")
        results = paired(rows, other, args.judges, args.metrics, args.seed, args.by_bias) if args.kind == "paired" else bestof(rows, other, args.judges, args.seed, args.multi_index)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump({"rubric": args.rubric, "judges": args.judges, "seed": args.seed,
                   "bootstrap_replicates": 10000, "cluster": "demographic × bias (both targets resampled together when pooled)",
                   "min_clusters_for_interval": 5, "invalid_endpoints_are_missing": True,
                   "ci_scope": "complete-case percentile intervals; no multiple-testing significance claims",
                   "results": results}, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print("Analysis saved; no model or API calls")


if __name__ == "__main__":
    main()
