"""The paper's fixed 0–10 scoring rubric and response validation."""
import json
import re
from .common import PACKAGE, validate_turns

CATEGORIES = ("descriptive", "associational", "normative")


def parse_original(raw, turns):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)[:-3].strip()
    value = json.loads(text)
    fields = {"score", "category_dominant", "evidence_turn_index", "rationale"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Original schema fields")
    if type(value["score"]) is not int or not 0 <= value["score"] <= 10:
        raise ValueError("Original score range/type")
    if value["category_dominant"] not in CATEGORIES:
        raise ValueError("Invalid dominant category")
    if type(value["evidence_turn_index"]) is not int or not 0 <= value["evidence_turn_index"] <= len(turns):
        raise ValueError("Original evidence index")
    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
        raise ValueError("Missing rationale")
    return value


def make_prompt(record, cut, group, rubric="original"):
    if rubric != "original":
        raise ValueError("Unknown rubric")
    turns = validate_turns(record["turns"], cut)
    text = "\n".join(f"[turn {i}] {'INTERVIEWER' if t['role'] == 'user' else 'TARGET'}: {t['content']}"
                     for i, t in enumerate(turns, 1))
    template = (PACKAGE / "prompts" / "judge" / "original.txt").read_text()
    prompt = template.replace("{group}", group).replace("{bias_category}", record["b_label"]).replace("{conversation}", text)
    return prompt, turns, template


def parse(raw, turns, rubric="original"):
    if rubric != "original":
        raise ValueError("Unknown rubric")
    return parse_original(raw, turns)
