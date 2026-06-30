"""Shared code for the clinical multi-adapter LoRA system.

`config`  — base model + adapter registry + LoRA hyperparameters.
`prompts` — the prompt-formatting contract shared by training, eval, and inference.

Keeping these in one place is the whole point: a LoRA adapter is only valid for the
exact prompt format it was trained on. Train and serve must read from the same module.
"""
