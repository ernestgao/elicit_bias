"""Two-judge selection and literal group neutralization (not symmetric identity swaps)."""
import argparse
import json
import re
from pathlib import Path

from .common import EVALUATION_JUDGES, trajectories, validate_judges
from .metrics import metric

SWAPS_BY_G: dict[str, list[tuple[str, str]]] = {
    "female":      [(r"\bwomen\b", "people"),
                    (r"\ba woman\b", "a person"),
                    (r"\bwoman\b", "person"),
                    (r"\bfemale\b", "person"),
                    (r"\bgirls?\b", "children"),
                    (r"\bshe\b", "they"), (r"\bher\b", "their"),
                    (r"\bhers\b", "theirs"), (r"\bherself\b", "themselves")],
    "nonbinary":   [(r"\bnon-?binary\b", "person"),
                    (r"\bgenderqueer\b", "person"),
                    (r"\bthey/them\b", "they")],
    "black":       [(r"\bblack people\b", "people"),
                    (r"\bblack person\b", "person"),
                    (r"\bblack\b", "")],
    "asian":       [(r"\basian americans?\b", "people"),
                    (r"\basian people\b", "people"),
                    (r"\basians?\b", "people")],
    "hispanic":    [(r"\bhispanic people\b", "people"),
                    (r"\bhispanics?\b", "people"),
                    (r"\blatino[ax]?\b", "people")],
    "muslim":      [(r"\bmuslims?\b", "people"),
                    (r"\bislam(?:ic)?\b", "")],
    "jewish":      [(r"\bjewish people\b", "people"),
                    (r"\bjews?\b", "people"),
                    (r"\bjudaism\b", "religion")],
    "elderly":     [(r"\belderly people\b", "people"),
                    (r"\belderly\b", ""),
                    (r"\bsenior citizens?\b", "people"),
                    (r"\bolder adults?\b", "adults")],
    "physical":    [(r"\bpeople with physical disabilit(?:y|ies)\b", "people"),
                    (r"\bphysically disabled people\b", "people"),
                    (r"\bphysically disabled\b", ""),
                    (r"\bwheelchair users?\b", "people")],
    "mental":      [(r"\bpeople with mental disabilit(?:y|ies)\b", "people"),
                    (r"\bmentally ill people\b", "people"),
                    (r"\bmentally disabled\b", ""),
                    (r"\bmentally ill\b", "")],
    "lgbtq":       [(r"\blgbtq\+?\b", "people"),
                    (r"\bgay people\b", "people"),
                    (r"\bgay\b", ""),
                    (r"\blesbians?\b", "people"),
                    (r"\bqueer\b", "")],
    "immigrant":   [(r"\bimmigrants?\b", "people"),
                    (r"\brefugees?\b", "people"),
                    (r"\bforeign-born\b", ""),
                    (r"\bnon-citizens?\b", "people")],
}


def _swap_text(text: str, g: str) -> str:
    rules = SWAPS_BY_G.get(g, [])
    out = text
    for pat, repl in rules:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    # Tidy double spaces left by deletions.
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _swap_trajectory(d: dict) -> dict:
    g = d.get("g", "")
    new_turns = []
    for t in d["turns"]:
        new_turns.append({**t, "content": _swap_text(t["content"], g)})
    out = {**d}
    out["turns"] = new_turns
    out["counterfactual"] = True
    out["counterfactual_source_g"] = g
    # Strip pre-existing scores; rejudge will fill them in fresh.
    out["judge_scores"] = []
    out["judge_scores_v2"] = {}
    return out



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--judges", choices=EVALUATION_JUDGES, nargs="+", default=list(EVALUATION_JUDGES))
    p.add_argument("--ranking", choices=["cut6", "realizable_max"], default="realizable_max")
    args = p.parse_args()
    try:
        validate_judges(args.judges, require_pair=True)
    except ValueError as exc:
        p.error(str(exc))
    if args.out.exists() or args.top_n < 1:
        p.error("Use a new output directory and positive --top-n")
    ranked, excluded = [], 0
    for directory_index, directory in enumerate(args.input_dirs):
        for identity, record in trajectories(directory):
            backends = record.get("judge_scores_v2", {})
            if backends:
                try:
                    validate_judges(backends)
                except ValueError as exc:
                    p.error(str(exc))
            try:
                value = metric(record, args.judges, args.ranking)
            except ValueError:
                excluded += 1
                continue
            ranked.append((value, f"{directory_index}__{identity}", record))
    selected = sorted(ranked, key=lambda row: -row[0])[:args.top_n]
    if not selected:
        p.error("No complete scored trajectories")
    for name in ("original", "neutralized"):
        (args.out / name).mkdir(parents=True)
    for _, identity, record in selected:
        for name, payload in (("original", record), ("neutralized", _swap_trajectory(record))):
            with (args.out / name / f"{identity}.json").open("x") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Selected {len(selected)} pairs; {excluded} incomplete records excluded; no scoring invoked")


if __name__ == "__main__":
    main()
