# Eliciting Bias through Conversation: Multi-Turn Interviewers for LLM Auditing

Code for training and evaluating multi-turn bias-elicitation interviewers.

## Repository

- `elicit_bias/`: data processing, training, generation, scoring and analysis.
- `elicit_bias/prompts/`: interviewer strategies, target instructions and judge rubric.
- `elicit_bias/taxonomy.yaml`: demographic groups, bias categories and strategies.
- `tests/`: offline tests with invented inputs.

## Usage

See the [usage guide](docs/usage.md) for installation, input formats, training,
generation and evaluation commands. Data and model weights must be supplied
separately.

See the [model card](MODEL_CARD.md) for intended use and limitations.

## License

[MIT License](LICENSE).
