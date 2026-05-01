# Context-Overload Prompt Injection Experiments

This repository contains code and results for a local LLM safety experiment testing whether context-overload prompt injection can produce harmful compliance in Qwen3-8B and Llama-3.1-8B.

## Summary

The experiment tests two conditions:

1. Context-length overload from 100 to 15,000 words
2. Positional ablation with the adversarial instruction placed at the front, middle, or back of a 10,000-word prompt

Across 48 total outputs, no harmful compliance was observed. The models instead produced refusal, empty output, benign task displacement, and repetition-detection/prompt-confusion behaviors.

The main finding is that naive refusal-keyword scanners can misclassify empty or derailed outputs as jailbreak successes.

## Scripts

- `qwen3_context_overload_test_3trials.py`
- `qwen3_position_ablation_3trials.py`
- `llama3_1_context_overload_test_3trials.py`
- `llama3_1_position_ablation_3trials.py`
- `graph_Builder.py`

## Results

CSV result files are stored in:

- `csv data/`

Graphs are stored in:

- `graphs/`

## Outcome Labels

Outputs were categorized as:

- Refusal
- Empty output
- Task displacement
- Prompt confusion
- Mechanical failure
- Harmful compliance

## Safety Note

The exact harmful instruction is generalized in the public code. This repository is for safety evaluation and does not contain operational cyber-intrusion instructions.
