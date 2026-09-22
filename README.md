# Eliciting Bias through Conversation: Multi-Turn Interviewers for LLM Auditing

## Contents and installation

```text
elicit_bias/       processing, training, generation, scoring, analysis source
  prompts/        six strategies, SFT/target instructions, scoring rubric
  taxonomy.yaml   12 groups × 5 bias categories × 6 strategies
tests/            code that constructs invented inputs in temporary directories
MODEL_CARD.md     intended use, method description and limitations
```

Python 3.10+. Install in your own environment:

```bash
python -m pip install -e .
python -m pip install -e '.[train]'
```

The basic dependencies are NumPy, PyYAML and requests. The training extra adds
PyTorch, Transformers, PEFT, Accelerate and Datasets. Real-model training and
local generation require model weights and sufficient BF16-capable GPU memory.

## External inputs and training

Input formats are defined by the loaders:

- Raw trajectories contain `g,g_label,b,b_label,strategy,turns`. Turns alternate
  between interviewer (`user`) and target (`assistant`); interviewer turns have
  a `strategy` field.
- Flattened SFT rows contain `g,b,strategy,history,target`, where `target` is the
  next interviewer utterance and `history` is its preceding conversation.
- Opener preference rows contain `prompt,chosen,rejected`. `prompt` is a
  JSON-encoded list of system/user messages. Optional score fields are validated
  for consistency.

The paths below are placeholders for inputs you are authorized to use. Set
`BIAS_INPUT_DIR` and `BIAS_OUTPUT_DIR` to locations outside this checkout.

```bash
python -m elicit_bias.prepare_sft --input "$BIAS_INPUT_DIR/trajectories" \
  --out "$BIAS_OUTPUT_DIR/sft.jsonl"
python -m elicit_bias.train_sft --model "$BIAS_INPUT_DIR/interviewer-base" \
  --data "$BIAS_OUTPUT_DIR/sft.jsonl" --out "$BIAS_OUTPUT_DIR/sft-adapter"
python -m elicit_bias.build_preferences \
  --input-dirs "$BIAS_INPUT_DIR/training-rollouts" --targets example-target \
  --out "$BIAS_OUTPUT_DIR/dpo.jsonl"
python -m elicit_bias.train_dpo --model "$BIAS_INPUT_DIR/interviewer-base" \
  --sft-adapter "$BIAS_OUTPUT_DIR/sft-adapter" \
  --data "$BIAS_OUTPUT_DIR/dpo.jsonl" --out "$BIAS_OUTPUT_DIR/dpo-adapter"
```

SFT masks history tokens and trains each interviewer utterance plus EOS. DPO
uses response-token log-probability sums and the logistic preference loss with
beta 0.1. It merges the frozen SFT adapter into the base, then trains a new DPO
LoRA against that reference. Inference applies both adapters in that order.

Preference construction consumes recorded MiMo MAX rewards in `judge_scores`,
preserving the recorded cut semantics. It requires two trajectories per
target/group/category/strategy cell and retains opener pairs with a reward
margin of at least 1. Missing rewards are errors. These commands provide the
implementation; they do not supply the study's training inputs or model weights.

## Generation and cloud configuration

Pass model paths explicitly. Only an explicit `HF_TOKEN` is used for
authenticated model access. Model IDs can download weights, so use local paths
when working offline. Generated output must be stored outside this checkout.

```bash
python -m elicit_bias.generate --interviewer local \
  --model "$BIAS_INPUT_DIR/interviewer-base" \
  --sft-adapter "$BIAS_INPUT_DIR/sft-adapter" \
  --dpo-adapter "$BIAS_INPUT_DIR/dpo-adapter" \
  --target local --target-model "$BIAS_INPUT_DIR/target-model" \
  --target-label example-target --out "$BIAS_OUTPUT_DIR/conversations" \
  --turns 6 --k-per-cell 1
```

`--groups`, `--biases`, `--strategies` and `--limit` restrict generation.
Optional `--seeds` accepts external JSONL with `g,b,qid,question`.
`--turns 1 --k-per-cell 6` generates independent one-turn candidates.

Cloud clients read settings from environment variables only:

| Backend | Required variables |
|---|---|
| Azure judge | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` |
| Azure target | same endpoint/key, `AZURE_TARGET_DEPLOYMENT` |
| Gemini judge | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| Mimo client | `MIMO_BASE_URL`, `MIMO_API_KEY`, `MIMO_MODEL` |
| Compatible target | `TARGET_BASE_URL`, `TARGET_API_KEY`, `TARGET_MODEL` |

These are variable names, not credentials. No key values or credential files
are distributed. Compatible base URLs must be HTTPS and end before
`/chat/completions`. Azure v1 bases typically end in `/openai/v1`; Gemini uses
its public `generateContent` endpoint. Embedded URL credentials, API-version
queries, redirects, proxy/netrc authentication and fallback judges are disabled.
Errors suppress service response bodies and credentials.

Cloud generation/scoring can incur charges. Generation has a 1,200-request cap
by default and no retries. Local generation makes no judge calls.

## Scoring and analysis

Final evaluation uses GPT-5-nano and Gemini 2.5 Flash-Lite. Scoring, analysis
and neutralization commands require `--judges azure gemini`. Configure the
corresponding models/deployments explicitly. MiMo rewards are training inputs;
final evaluation uses the two judges above.

The original rubric returns a raw integer score from 0 to 10,
`category_dominant` (descriptive/associational/normative), an evidence turn index,
and a short rationale. Exchange `t` contains the first `2t` messages. Fixed cut
6 is the primary late endpoint. Realizable MAX is the maximum of the
judge-averaged scores at exchanges 2, 4 and 6. Invalid responses remain missing.

```bash
python -m elicit_bias.score --input "$BIAS_OUTPUT_DIR/conversations" \
  --out "$BIAS_OUTPUT_DIR/scores.jsonl" --cuts 1 2 4 6
python -m elicit_bias.analyze summary --input "$BIAS_OUTPUT_DIR/scores.jsonl" \
  --metrics cut1 cut2 cut4 cut6 mean_cut realizable_max \
  --out "$BIAS_OUTPUT_DIR/summary.json"
```

Scoring writes separate results and a durable request ledger. Resume the same
command to reuse validated results. Default limits are 1,200 total attempts
and three per cache key; a lock prevents concurrent writers. Keep both output
and ledger outside the source tree.

Analysis supports summaries, severity distributions, trajectory changes, paired
comparisons, and best-of-3/6 comparisons using external scores. Best-of-3 uses
indices 0–2 from a six-candidate bank; IDs end in `__index`. Group neutralization
uses literal transcript edits, whose interpretation is limited by remaining
context and selection of high-scoring inputs. No experiment records or results
are distributed with these analysis functions.

Intervals use 10,000 demographic×bias cluster-bootstrap replicates; fewer than
five clusters produces no interval. The portable commands do not implement all
paper statistics, including sign tests, Holm correction families, cross-judge
candidate selection and endpoint-specific bootstrap seeds. Independently running
generation on two targets does not guarantee the shared opener bank required by
the paper's matched-budget protocol.

## Offline verification

```bash
python -B -m unittest discover -s tests -v
python -B -m elicit_bias.verify
python -B -m elicit_bias.generate --interviewer local --model UNUSED \
  --target local --target-model UNUSED --target-label example-target \
  --out "$BIAS_OUTPUT_DIR/unused" --limit 1 --dry-run
```

Tests construct invented text at runtime and delete their temporary files.
They exercise the training loaders and dry runs without research data, model
weights or API calls. A small tensor test is skipped when PyTorch is absent.
Real-model training and live API interoperability have not been exercised by
these checks.

The release verifier permits only explicitly listed source, prompt, taxonomy and
documentation files. It rejects data files of any size, unreviewed files,
symlinks and oversized files, and scans hidden files and common credential
patterns without printing matched values. This is a packaging and content
check; it does not grant rights to third-party materials.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use and limitations.

## License

MIT License. See [LICENSE](LICENSE). External model, dataset and API terms
continue to apply.
