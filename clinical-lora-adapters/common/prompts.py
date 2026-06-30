"""Prompt-formatting contract — shared by training, eval, and inference.

A LoRA adapter learns to respond to *one specific* prompt format. If training wraps the
note one way and inference wraps it another, the adapter silently underperforms. So both
call into this module; nothing here is duplicated elsewhere.

We build a chat-style message list and render it with the tokenizer's own chat template
when it has one (Mistral-Instruct, Llama-2-chat both do). For base models without a chat
template we fall back to a plain `### Instruction / ### Input / ### Response` layout.
"""

from __future__ import annotations

import json
from typing import Any

from .config import TASK_EXTRACTION, TASK_SUMMARIZATION, AdapterSpec

# System instructions per task. Kept terse — the adapter internalizes the behavior;
# the prompt just names the job so the same base model can host both adapters.
_SYSTEM = {
    TASK_SUMMARIZATION: (
        "You are a clinical documentation assistant specializing in cardiology. "
        "Summarize the note into a concise structured summary with sections: "
        "Presentation, Key Findings, Assessment, Plan. Preserve all numeric values "
        "(vitals, labs, ejection fraction) exactly. Do not invent information."
    ),
    TASK_EXTRACTION: (
        "You are a clinical information-extraction engine for radiology reports. "
        "Return ONLY a JSON object with keys: modality, body_part, findings (list of "
        "strings), impression (string), measurements (list of strings), "
        "critical_flag (boolean). Do not add commentary or markdown fences."
    ),
}

_INSTRUCTION = {
    TASK_SUMMARIZATION: "Summarize the following cardiology note.",
    TASK_EXTRACTION: "Extract structured findings from the following radiology report.",
}


def render_target(spec: AdapterSpec, target: Any) -> str:
    """Serialize a ground-truth target to the string the model should produce."""
    if spec.task_type == TASK_EXTRACTION:
        # Canonical, compact JSON with stable key order — so training target and
        # parsed model output compare cleanly during eval.
        return json.dumps(target, ensure_ascii=False, sort_keys=True)
    return str(target).strip()


def build_messages(spec: AdapterSpec, input_text: str) -> list[dict[str, str]]:
    """Chat messages for the *prompt only* (no assistant turn)."""
    return [
        {"role": "system", "content": _SYSTEM[spec.task_type]},
        {"role": "user", "content": f"{_INSTRUCTION[spec.task_type]}\n\n{input_text.strip()}"},
    ]


def _fallback_prompt(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    sys = next((m["content"] for m in messages if m["role"] == "system"), "")
    usr = next((m["content"] for m in messages if m["role"] == "user"), "")
    text = f"### System\n{sys}\n\n### Instruction\n{usr}\n\n### Response\n"
    if not add_generation_prompt:
        # caller will append the target then the eos token
        return text
    return text


def format_inference_prompt(tokenizer, spec: AdapterSpec, input_text: str) -> str:
    """Prompt string ending right where the model should start generating."""
    messages = build_messages(spec, input_text)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return _fallback_prompt(messages, add_generation_prompt=True)


def format_training_text(tokenizer, spec: AdapterSpec, input_text: str, target: Any) -> str:
    """Full prompt + target + eos — the single string a causal LM trains on."""
    target_str = render_target(spec, target)
    if getattr(tokenizer, "chat_template", None):
        messages = build_messages(spec, input_text) + [
            {"role": "assistant", "content": target_str}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    eos = tokenizer.eos_token or ""
    prompt = _fallback_prompt(build_messages(spec, input_text), add_generation_prompt=True)
    return f"{prompt}{target_str}{eos}"
