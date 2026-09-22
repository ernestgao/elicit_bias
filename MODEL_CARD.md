# Conversational bias interviewer

This code supports research auditing of bias in model-generated conversations.
Use it to evaluate systems you are authorized to test. The strategy prompts can
elicit offensive or discriminatory responses; using those responses to target
people or to deploy a user-facing discriminatory system is outside the intended
use. The research prompts do not endorse the content they may elicit.

## Training and model setup

The interviewer backbone is Qwen3-4B-Instruct with LoRA adapters. MiMo-v2.5-pro
supplies 720 synthetic six-exchange demonstrations across 360 demographic,
bias-category and strategy cells. All 4,320 interviewer turns supply SFT
examples. Original opener DPO uses MiMo MAX trajectory rewards with a minimum
preference margin of 1: 181 Qwen3 pairs for the single-target variant, or 544
pairs pooled across Qwen3-4B, Mistral-7B and Phi-3-mini. Apply the SFT adapter
first, then the DPO adapter. The target model has no interviewer adapter.

Final evaluation uses GPT-5-nano and Gemini 2.5 Flash-Lite, averaging their raw
0–10 scores at each exchange cut. Cut 6 is the primary late endpoint; realizable
MAX selects the largest averaged score at cuts 2, 4 and 6. Yi-1.5-6B and
GPT-5-mini are held-out targets. Exact prompts and the operational group labels
are included in the package.

## Contents and limitations

This distribution contains source code only. Research training/evaluation data,
including excerpts, model-generated responses, complete trajectories, score
caches, and all pretrained adapters are excluded for licensing reasons. The
SFT and DPO_multi adapters are not distributed. Tests construct invented inputs
in temporary directories; they do not load research examples. Training and live API interoperability have not been verified
by the offline checks. The code does not reproduce every supplementary analysis;
see the README for the implemented coverage and the required external inputs.

Scores depend on the prompts, model versions and evaluator judgments. Synthetic
training dialogue does not represent natural user interaction. The first-turn
baseline and later cuts use different context and query budgets. Results cover
a limited target set, and Qwen shares the interviewer's backbone. Group
neutralization uses literal edits and can leave comparison identities intact.
These measurements do not establish that an audited model is fair or safe.

The source code is distributed under the [MIT License](LICENSE).
External model, dataset and API terms continue to apply.
