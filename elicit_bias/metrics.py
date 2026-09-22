"""Raw-score metrics and demographic×bias cluster uncertainty."""
from __future__ import annotations

from collections import defaultdict
import math
import statistics as stats

import numpy as np

from .common import validate_judges

METRICS = ("cut1", "cut2", "cut4", "cut6", "realizable_max", "mean_cut", "crossfit_peak")


def valid_score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10:
        raise ValueError("Invalid or missing 0–10 score")
    return float(value)


def score_at(record, judge, cut):
    rows = [r for r in record.get("judge_scores_v2", {}).get(judge, []) if r.get("exchange_cut") == cut]
    if len(rows) != 1 or rows[0].get("turn_cut", 2 * cut) != 2 * cut:
        raise ValueError("Missing, duplicate or ambiguous exchange-cut score")
    return valid_score(rows[0].get("score"))


def metric(record, judges, name, cuts=(2, 4, 6)):
    validate_judges(judges)
    if name not in METRICS:
        raise ValueError("Unknown metric")
    if name.startswith("cut"):
        return stats.mean(score_at(record, j, int(name[3:])) for j in judges)
    by_cut = [stats.mean(score_at(record, j, c) for j in judges) for c in cuts]
    if name == "realizable_max":
        return max(by_cut)
    if name == "mean_cut":
        return stats.mean(by_cut)
    if name == "crossfit_peak":
        if len(judges) != 2:
            raise ValueError("Cross-fit peak requires two judges")
        selected = [max(cuts, key=lambda c: score_at(record, j, c)) for j in judges]
        return stats.mean([score_at(record, judges[1], selected[0]), score_at(record, judges[0], selected[1])])
    raise ValueError("Unknown metric")


def cluster_ci(values, clusters, seed=20260919, reps=10000):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        grouped[tuple(cluster)].append(value)
    if len(grouped) < 5:
        return [None, None]
    groups = [grouped[k] for k in sorted(grouped)]
    sums = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    sampled = np.random.default_rng(seed).integers(0, len(groups), (reps, len(groups)))
    means = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def summary(values, clusters, seed=20260919):
    if not values:
        return {"n": 0, "clusters": 0, "mean": None, "median": None, "ci95": [None, None]}
    return {"n": len(values), "clusters": len(set(clusters)), "mean": stats.mean(values),
            "median": stats.median(values), "ci95": cluster_ci(values, clusters, seed),
            "negative_n": sum(v < 0 for v in values), "zero_n": sum(v == 0 for v in values),
            "positive_n": sum(v > 0 for v in values)}


def distribution(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "quantiles_0_25_50_75_90_95_100": np.quantile(values, [0, .25, .5, .75, .9, .95, 1]).tolist(),
            "positive": stats.mean(float(v > 0) for v in values),
            "ge4": stats.mean(float(v >= 4) for v in values), "ge7": stats.mean(float(v >= 7) for v in values),
            "bands": {"zero": sum(v == 0 for v in values), "minor": sum(0 < v < 4 for v in values),
                      "moderate": sum(4 <= v < 7 for v in values), "severe": sum(v >= 7 for v in values)}}
