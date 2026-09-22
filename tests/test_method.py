"""Offline checks using invented text; temporary inputs are never research data."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

from elicit_bias import analyze, build_preferences, metrics, neutralize, rubrics, score
from elicit_bias.cloud import CloudClient, CloudError
from elicit_bias.common import EVALUATION_JUDGES, validate_judges
from elicit_bias.generate import rollout
from elicit_bias.prepare_sft import flatten_trajectory, format_for_model, load_sft
from elicit_bias.score import Scorer, make_jobs

def fixture():
    return {"g": "female", "g_label": "women", "b": "decision_bias", "b_label": "decision bias",
            "strategy": "S5_contradiction", "target": "test-target",
            "turns": [{"role": "user" if i % 2 == 0 else "assistant", "content": "Question." if i % 2 == 0 else "Evaluate each candidate fairly.",
                       **({"strategy": "S5_contradiction"} if i % 2 == 0 else {})} for i in range(12)]}


def training_fixtures(directory):
    """Write invented format-check inputs into the caller's temporary directory."""
    directory = Path(directory)
    sft, dpo = directory / "sft.jsonl", directory / "dpo.jsonl"
    sft.write_text("".join(json.dumps(row) + "\n" for row in flatten_trajectory(fixture())))
    pair = {"prompt": json.dumps([{"role": "system", "content": "Ask a test question."},
                                   {"role": "user", "content": "Begin the test."}]),
            "chosen": "Test question A.", "rejected": "Test question B."}
    dpo.write_text(json.dumps(pair) + "\n")
    return sft, dpo


class FakeClient:
    model = "test-model"
    base = "https://example.invalid/v1"
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail
    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.fail:
            return "invalid JSON", {}
        return json.dumps({"score": 0, "category_dominant": "descriptive",
                           "evidence_turn_index": 0, "rationale": "No bias."}), {}


class Tokenizer:
    eos_token_id = 999
    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT"
    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3] if text == "PROMPT" else [4, 5]}


class MethodTests(unittest.TestCase):
    def test_invented_training_formats_and_dry_runs(self):
        from elicit_bias import train_sft, train_dpo
        with tempfile.TemporaryDirectory() as tmp:
            sft_path, dpo_path = training_fixtures(tmp)
            sft = load_sft(sft_path)
            dpo = build_preferences.validate_pairs(dpo_path)
            self.assertEqual((len(sft), len(dpo)), (6, 1))
            self.assertTrue(any(row["history"] for row in sft))
            for module, data, extra in ((train_sft, sft_path, []),
                                        (train_dpo, dpo_path, ["--sft-adapter", "UNUSED"])):
                args = ["test", "--model", "UNUSED", "--data", str(data),
                        "--out", str(Path(tmp) / "unused"), "--dry-run", *extra]
                with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                    module.main()
            self.assertFalse((Path(tmp) / "unused").exists())

    def test_sft_all_turns_and_response_mask(self):
        rows = flatten_trajectory(fixture())
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(rows[-1]["history"]), 10)
        encoded = format_for_model(Tokenizer(), rows[0], max_len=5)
        self.assertEqual(encoded["labels"], [-100, -100, 4, 5, 999])
        self.assertEqual(encoded["input_ids"], [2, 3, 4, 5, 999])

    def test_realizable_max_is_not_mean_of_maxima(self):
        record = fixture()
        record["judge_scores_v2"] = {j: [{"exchange_cut": cut, "turn_cut": 2 * cut, "score": value} for cut, value in zip([2, 4, 6], scores)]
                                      for j, scores in [("azure", [10, 0, 2]), ("gemini", [0, 10, 4])]}
        self.assertEqual(metrics.metric(record, ["azure", "gemini"], "realizable_max"), 5)
        self.assertLess(metrics.metric(record, ["azure", "gemini"], "realizable_max"), 10)
        self.assertEqual(metrics.metric(record, ["azure", "gemini"], "cut6"), 3)
        record["judge_scores_v2"]["gemini"].pop()
        with self.assertRaises(ValueError):
            metrics.metric(record, ["azure", "gemini"], "cut6")

    def test_preferences_are_opener_only_and_source_cuts_stay_recorded(self):
        low, high = fixture(), fixture()
        high["turns"][0]["content"] = "Higher reward opener."
        low["judge_scores"] = [{"turn_cut": 2, "score": 1}, {"turn_cut": 6, "score": 2}]
        high["judge_scores"] = [{"turn_cut": 2, "score": 6}, {"turn_cut": 6, "score": 3}]
        pairs = build_preferences.build([("a", low), ("b", high)], "test", 1)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"], high["turns"][0]["content"])
        self.assertEqual(pairs[0]["chosen_score"], 6)
        self.assertEqual(len(json.loads(pairs[0]["prompt"])), 2)


    def test_rubric_and_exchange_cut_validation(self):
        record = fixture()
        prompt, turns, _ = rubrics.make_prompt(record, 1, "women")
        self.assertEqual(len(turns), 2)
        self.assertIn("[turn 2] TARGET:", prompt)
        self.assertNotIn("[turn 3]", prompt)
        with self.assertRaises(ValueError):
            rubrics.make_prompt(record, 7, "women")
        raw, _ = FakeClient().complete([])
        verdict = json.loads(raw)
        self.assertEqual(rubrics.parse_original(raw, turns)["score"], 0)
        for changes in ({"score": True}, {"score": 11}, {"category_dominant": "unknown"},
                        {"evidence_turn_index": 3}, {"rationale": ""}):
            with self.assertRaises(ValueError):
                rubrics.parse_original(json.dumps({**verdict, **changes}), turns)

    def test_preferences_require_complete_pairs_and_recorded_rewards(self):
        low, high = fixture(), fixture()
        high["turns"][0]["content"] = "Higher reward opener."
        low["judge_scores"] = [{"turn_cut": 2, "score": 1}]
        high["judge_scores"] = [{"turn_cut": 2, "score": 2}]
        pairs = build_preferences.build([("a", low), ("b", high)], "test")
        self.assertEqual(pairs[0]["reward_definition"], "mimo_max")
        self.assertEqual(pairs[0]["score_margin"], 1)
        for records in ([('a', low)], [('a', low), ('b', high), ('c', fixture())],
                        [('a', low), ('b', fixture())]):
            with self.assertRaises(ValueError):
                build_preferences.build(records, "test")
        high["judge_scores"][0]["score"] = 1.5
        self.assertEqual(build_preferences.build([("a", low), ("b", high)], "test"), [])

    def test_trajectory_roles_and_rendering(self):
        captured = []
        def ask(messages):
            captured.append(messages)
            return "An interviewer question."
        r = rollout(ask, lambda m: "A target answer.", group={"id": "female", "label": "women"},
                    bias={"id": "decision_bias", "label": "decision bias"}, strategy="S5_contradiction", turns=2)
        self.assertEqual([t["role"] for t in r["turns"]], ["user", "assistant"] * 2)
        self.assertIn("YOU: An interviewer question.", captured[1][1]["content"])
        self.assertIn("TARGET: A target answer.", captured[1][1]["content"])

    def test_budget_failures_and_cache_resume(self):
        with tempfile.TemporaryDirectory(prefix="elicit-bias-test-") as tmp:
            client = FakeClient()
            jobs = make_jobs([("a", fixture())], {"azure": client}, "original", [1, 6])
            path = Path(tmp) / "scores.jsonl"
            runner = Scorer(path, {"azure": client}, cap=1)
            self.assertEqual(runner.work(jobs[0]), "new_success")
            self.assertEqual(runner.work(jobs[1]), "unresolved")
            resumed = Scorer(path, {"azure": client}, cap=1)
            self.assertEqual(resumed.work(jobs[0]), "cached")
            self.assertEqual(client.calls, 1)
            failure = FakeClient(fail=True)
            runner = Scorer(Path(tmp) / "bad.jsonl", {"azure": failure}, cap=10, per_key=3)
            self.assertEqual(runner.work(jobs[0]), "unresolved")
            self.assertEqual(failure.calls, 3)
            self.assertEqual(runner.cache, {})

    def test_missing_cloud_config_has_no_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            for backend, role in (("azure", "judge"), ("gemini", "judge"), ("mimo", "interviewer"), ("mimo", "target"), ("compatible", "target")):
                with self.assertRaisesRegex(CloudError, "Missing environment variable"):
                    CloudClient(backend, role=role)

    def test_evaluation_backends_are_explicit_and_unique(self):
        self.assertEqual(EVALUATION_JUDGES, ("azure", "gemini"))
        validate_judges(EVALUATION_JUDGES, require_pair=True)
        for selected in ([], ["azure"], ["azure", "azure"], ["unsupported"]):
            with self.assertRaises(ValueError):
                validate_judges(selected, require_pair=True)
        with self.assertRaisesRegex(CloudError, "only Azure and Gemini"):
            CloudClient("unsupported")
        with self.assertRaises(ValueError):
            make_jobs([("a", fixture())], {"unsupported": FakeClient()}, "original", [1])

    def test_evaluation_clis_reject_invalid_backends_before_io(self):
        commands = [
            (score.main, ["--input", "UNUSED", "--out", "UNUSED", "--rubric", "original"]),
            (analyze.main, ["summary", "--input", "UNUSED", "--out", "UNUSED", "--rubric", "original"]),
            (neutralize.main, ["--input-dirs", "UNUSED", "--out", "UNUSED"]),
        ]
        for function, args in commands:
            for selected in (["azure"], ["azure", "azure"], ["unsupported"]):
                with self.subTest(command=function.__module__, judges=selected):
                    with patch("sys.argv", ["test", *args, "--judges", *selected]), contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            function()
                        self.assertEqual(error.exception.code, 2)

    def test_scored_inputs_reject_unsupported_backends(self):
        record = fixture()
        record["judge_scores_v2"] = {"unsupported": [{"exchange_cut": 6, "turn_cut": 12, "score": 2}]}
        with tempfile.TemporaryDirectory(prefix="elicit-bias-test-") as tmp:
            path = Path(tmp) / "input.json"
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                analyze.load_scores(path, "original")
            with patch("sys.argv", ["test", "--input-dirs", tmp, "--out", str(Path(tmp) / "output")]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    neutralize.main()
                self.assertEqual(error.exception.code, 2)
            self.assertFalse((Path(tmp) / "output").exists())

    def test_neutralization_selects_shared_cut_with_both_judges(self):
        with tempfile.TemporaryDirectory(prefix="elicit-bias-test-") as tmp:
            inputs, output = Path(tmp) / "inputs", Path(tmp) / "output"
            inputs.mkdir()
            for identity, score_pairs in (("different_peaks", ([10, 0, 2], [0, 10, 4])), ("shared_peak", ([6, 6, 6], [6, 6, 6]))):
                record = fixture()
                record["judge_scores_v2"] = {
                    j: [{"exchange_cut": cut, "turn_cut": 2 * cut, "score": value} for cut, value in zip((2, 4, 6), values)]
                    for j, values in zip(EVALUATION_JUDGES, score_pairs)
                }
                (inputs / f"{identity}.json").write_text(json.dumps(record))
            with patch("sys.argv", ["test", "--input-dirs", str(inputs), "--out", str(output), "--top-n", "1"]), contextlib.redirect_stdout(io.StringIO()):
                neutralize.main()
            selected = list((output / "original").glob("*.json"))
            self.assertEqual([p.stem for p in selected], ["0__shared_peak"])
            self.assertEqual(set(json.loads(selected[0].read_text())["judge_scores_v2"]), set(EVALUATION_JUDGES))
            self.assertEqual(json.loads(next((output / "neutralized").glob("*.json")).read_text())["judge_scores_v2"], {})

    def test_transport_does_not_echo_credentials_or_follow_redirects(self):
        config = {"AZURE_OPENAI_ENDPOINT": "https://example.invalid/v1", "AZURE_OPENAI_API_KEY": "TEST_ONLY_PLACEHOLDER", "AZURE_OPENAI_DEPLOYMENT": "test-model"}
        with patch.dict(os.environ, config, clear=True), patch("requests.Session") as session:
            connection = session.return_value.__enter__.return_value
            connection.post.side_effect = RuntimeError("TEST_ONLY_PLACEHOLDER")
            client = CloudClient("azure")
            with self.assertRaises(CloudError) as error:
                client.complete([])
            self.assertNotIn("TEST_ONLY_PLACEHOLDER", str(error.exception))
            self.assertFalse(connection.trust_env)
            self.assertFalse(connection.post.call_args.kwargs["allow_redirects"])

    def test_release_scan_rejects_credentials_and_generated_payloads(self):
        from elicit_bias.verify import inspect, MAX_FILE_BYTES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("A small source release.\n")
            self.assertTrue(inspect(root)["passed"])
            value = "sk-" + "A" * 32
            (root / ".env").write_text("API_KEY=" + value)
            payload = root / "cache.jsonl"
            with payload.open("wb") as handle:
                handle.truncate(MAX_FILE_BYTES + 1)
            report = inspect(root)
            issues = {r["issue"] for r in report["findings"]}
            self.assertTrue({"excluded-artifact", "token-pattern", "file-size-limit"} <= issues)
            self.assertNotIn(value, json.dumps(report))

    def test_release_scan_rejects_small_data_and_unreviewed_files(self):
        from elicit_bias.verify import inspect
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = ["samples/sft.jsonl", "samples/dpo.jsonl", "elicit_bias/data.txt",
                     "elicit_bias/experiment.py", "elicit_bias/prompts/transcript.txt",
                     "weights/adapter_model.safetensors", "weights/adapter_config.json"]
            for rel in paths:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
            report = inspect(root)
            excluded = {r["file"] for r in report["findings"] if r["issue"] == "excluded-artifact"}
            self.assertEqual(excluded, set(paths))


    def test_offline_analysis_cli_and_scoring_dry_run(self):
        from elicit_bias import score, prepare_sft
        with tempfile.TemporaryDirectory(prefix="elicit-bias-test-") as tmp:
            directory = Path(tmp) / "inputs"
            directory.mkdir()
            record = fixture()
            record["judge_scores_v2"] = {j: [{"exchange_cut": c, "turn_cut": 2 * c, "score": 2}
                                              for c in (1, 2, 4, 6)] for j in ("azure", "gemini")}
            (directory / "female__decision_bias__S5_contradiction__0.json").write_text(json.dumps(record))
            def invoke(function, args):
                with patch("sys.argv", ["test", *args]), contextlib.redirect_stdout(io.StringIO()):
                    function()
            output = Path(tmp) / "summary.json"
            invoke(analyze.main, ["summary", "--input", str(directory), "--rubric", "original", "--out", str(output)])
            self.assertTrue(json.loads(output.read_text())["results"])
            output = Path(tmp) / "prepared.jsonl"
            invoke(prepare_sft.main, ["--input", str(directory), "--out", str(output)])
            self.assertEqual(len(load_sft(output)), 6)
            with patch.object(CloudClient, "__init__", side_effect=AssertionError("Dry run configured a service")):
                invoke(score.main, ["--input", str(directory), "--out", str(Path(tmp) / "unused.jsonl"),
                                    "--rubric", "original", "--judges", "azure", "gemini", "--dry-run"])

    def test_bestof_three_uses_fixed_first_three_of_six(self):
        def rows(identity, scores):
            base = fixture()
            return [{"target": base["target"], "id": identity, "g": base["g"], "b": base["b"],
                     "strategy": base["strategy"], "backend": "azure", "cut": cut, "condition": "true_group",
                     "status": "validated", "verdict": {"score": val}} for cut, val in scores.items()]
        multi = rows("cell__0", {1: 0, 2: 5, 4: 5, 6: 5})
        candidates = [row for i in range(6) for row in rows(f"cell__{i}", {1: 1 if i < 3 else 9})]
        result = analyze.bestof(multi, candidates, ["azure"], 0)
        self.assertEqual([r["mean"] for r in result if r["metric"] == "realizable_max"], [4, -4])

    def test_dpo_loss_and_masking_if_training_extra_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional training dependencies absent")
        from elicit_bias.train_dpo import dpo_loss, DPODataset, collate, _logp_response
        z = torch.zeros(2)
        loss, *_ = dpo_loss(z, z, z, z, .1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        with tempfile.TemporaryDirectory() as tmp:
            _, path = training_fixtures(tmp)
            ds = DPODataset(path, Tokenizer(), 10, 20)
        self.assertEqual(ds[0]["chosen"]["labels"].tolist(), [-100, -100, -100, 4, 5, 999])
        batch = collate([ds[0]], pad_id=0)["chosen"]
        class Model:
            def __call__(self, input_ids, attention_mask):
                from types import SimpleNamespace
                return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 1000)))
        self.assertAlmostEqual(_logp_response(Model(), batch).item(), -3 * math.log(1000), places=4)


if __name__ == "__main__":
    unittest.main()
