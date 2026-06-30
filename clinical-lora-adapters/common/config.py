"""Central configuration: base model, adapter registry, LoRA hyperparameters.

Design decisions:
- One base model, many adapters. The base is loaded once; adapters are ~10-50 MB
  each and swap in/out at runtime. This is what cuts memory ~90% vs. one full model
  per specialty.
- The adapter registry is the single source of truth. Training writes adapters keyed
  by `name`; inference loads them by the same key. No string drift between the two.
- Everything env-overridable so the same code runs on a CPU laptop (tiny model, for a
  smoke test) and an A100 (Mistral-7B, real training).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
# Default to Mistral-7B-Instruct: open weights, strong instruction following,
# Apache-2.0 (no gated license friction like Llama-2). Swap to Llama-2 by setting
# BASE_MODEL=meta-llama/Llama-2-7b-chat-hf. For a CPU smoke test, point at a tiny
# model, e.g. BASE_MODEL=sshleifer/tiny-gpt2 (won't be clinically useful, just wiring).
BASE_MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# 4-bit QLoRA quantization for the base weights. Cuts a 7B model from ~14 GB (fp16)
# to ~4.5 GB so it trains on a single 16 GB GPU. Requires bitsandbytes + CUDA;
# auto-disabled off-CUDA (see training.train.load_base_model).
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"

ADAPTERS_DIR = Path(os.environ.get("ADAPTERS_DIR", REPO_ROOT / "adapters"))
DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Tasks & adapter registry
# ---------------------------------------------------------------------------
# Two task "shapes" with different eval metrics:
#   - summarization → free text, scored with ROUGE
#   - extraction    → structured JSON, scored with field-level F1
TASK_SUMMARIZATION = "summarization"
TASK_EXTRACTION = "extraction"


@dataclass(frozen=True)
class AdapterSpec:
    name: str                       # registry key, also the on-disk folder name
    specialty: str                  # human label, e.g. "cardiology"
    task_type: str                  # TASK_SUMMARIZATION | TASK_EXTRACTION
    data_file: str                  # JSON file under data/
    input_key: str                  # field in each record holding the model input
    target_key: str                 # field holding the ground-truth target
    description: str

    @property
    def data_path(self) -> Path:
        return DATA_DIR / self.data_file

    @property
    def output_dir(self) -> Path:
        return ADAPTERS_DIR / self.name


ADAPTERS: dict[str, AdapterSpec] = {
    "cardiology-summary": AdapterSpec(
        name="cardiology-summary",
        specialty="cardiology",
        task_type=TASK_SUMMARIZATION,
        data_file="cardiology_notes.json",
        input_key="note",
        target_key="summary",
        description="Condense a cardiology consult/progress note into a structured clinical summary.",
    ),
    "radiology-extract": AdapterSpec(
        name="radiology-extract",
        specialty="radiology",
        task_type=TASK_EXTRACTION,
        data_file="radiology_notes.json",
        input_key="report",
        target_key="extraction",
        description="Extract structured findings (modality, body part, impression, measurements, critical flag) from a radiology report.",
    ),
}


def get_adapter(name: str) -> AdapterSpec:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown adapter '{name}'. Known: {list(ADAPTERS)}")
    return ADAPTERS[name]


# ---------------------------------------------------------------------------
# LoRA hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class LoraHParams:
    r: int = 16                     # rank — capacity of the low-rank update
    lora_alpha: int = 32            # scaling (alpha/r = 2.0 effective)
    lora_dropout: float = 0.05
    bias: str = "none"
    # Attention + MLP projections. Names match Mistral/Llama. For other
    # architectures peft also accepts "all-linear".
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainHParams:
    epochs: float = 3.0
    learning_rate: float = 2e-4
    per_device_batch_size: int = 4
    grad_accum_steps: int = 4       # effective batch = 16
    max_seq_len: int = 1024
    warmup_ratio: float = 0.03
    logging_steps: int = 5
    seed: int = 42


DEFAULT_LORA = LoraHParams()
DEFAULT_TRAIN = TrainHParams()
