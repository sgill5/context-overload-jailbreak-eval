# Context-Overload Prompt Injection Experiments

This repository contains the code and result files for a local LLM safety experiment testing whether context-overload prompt injection can produce harmful compliance in Qwen3-8B and Llama-3.1-8B.

The project evaluates two conditions:

1. Context-length overload from 100 to 15,000 words
2. Positional ablation with the adversarial instruction placed at the front, middle, or back of a 10,000-word prompt

Across 48 total outputs, no harmful compliance was observed. Instead, the models produced refusal, empty output, benign task displacement, and repetition-detection/prompt-confusion behaviors. The main finding is that naive refusal-keyword scanners can misclassify empty or derailed outputs as jailbreak successes.
