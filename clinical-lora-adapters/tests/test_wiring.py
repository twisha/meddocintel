"""Lightweight wiring tests — no torch / GPU / model download required.

Validates the parts that break silently: the adapter registry, the dataset schema, and
the prompt-format contract that train-time and serve-time must share. Run: pytest -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import prompts
from common.config import ADAPTERS, TASK_EXTRACTION, get_adapter


class _FakeTokenizer:
    """Stand-in with no chat_template, exercising the fallback prompt path."""

    chat_template = None
    eos_token = "</s>"


def test_registry_and_data_align():
    assert set(ADAPTERS) == {"cardiology-summary", "radiology-extract"}
    for spec in ADAPTERS.values():
        records = json.loads(spec.data_path.read_text())
        assert len(records) >= 4, f"{spec.name} needs more seed data"
        for rec in records:
            assert spec.input_key in rec and spec.target_key in rec


def test_extraction_targets_have_schema():
    spec = get_adapter("radiology-extract")
    required = {"modality", "body_part", "findings", "impression", "measurements", "critical_flag"}
    for rec in json.loads(spec.data_path.read_text()):
        assert required <= set(rec["extraction"]), rec["id"]


def test_prompt_contract_prompt_is_prefix_of_training_text():
    tok = _FakeTokenizer()
    spec = get_adapter("cardiology-summary")
    rec = json.loads(spec.data_path.read_text())[0]
    prompt = prompts.format_inference_prompt(tok, spec, rec["note"])
    full = prompts.format_training_text(tok, spec, rec["note"], rec["summary"])
    # The training text must extend the inference prompt — otherwise the prompt-mask in
    # train.py masks the wrong tokens and the adapter learns nothing useful.
    assert full.startswith(prompt)
    assert full.endswith(tok.eos_token)


def test_extraction_target_is_canonical_json():
    spec = get_adapter("radiology-extract")
    rec = json.loads(spec.data_path.read_text())[0]
    rendered = prompts.render_target(spec, rec["extraction"])
    assert spec.task_type == TASK_EXTRACTION
    assert json.loads(rendered) == rec["extraction"]  # round-trips
