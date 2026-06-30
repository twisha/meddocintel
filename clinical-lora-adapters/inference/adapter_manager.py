"""AdapterManager — one base model in memory, many adapters swapped at O(1).

This is the production payoff of the whole project. PEFT lets multiple named adapters
live on top of a single base model; switching specialties is `set_adapter(name)`, a
pointer flip, not a model reload. So serving cardiology + radiology costs:

    one 7B base (~4.5 GB in 4-bit)  +  ~30 MB per adapter

instead of one full 7B per specialty. That is the ~90% memory reduction in the narrative.

Concurrency note: set_adapter() mutates shared state on the single model, so generation is
serialized behind a lock. For higher throughput you'd batch by adapter or run a replica per
hot adapter — called out in the README scaling section.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("adapter_manager")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import prompts  # noqa: E402
from common.config import ADAPTERS, BASE_MODEL, TASK_EXTRACTION, get_adapter  # noqa: E402


@dataclass
class GenResult:
    adapter: str
    output_text: str
    parsed: dict | None          # populated for extraction tasks
    latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass
class AdapterStats:
    requests: int = 0
    total_latency_ms: float = 0.0
    total_output_tokens: int = 0

    def record(self, latency_ms: float, out_tokens: int) -> None:
        self.requests += 1
        self.total_latency_ms += latency_ms
        self.total_output_tokens += out_tokens

    def as_dict(self) -> dict:
        n = self.requests or 1
        return {
            "requests": self.requests,
            "mean_latency_ms": round(self.total_latency_ms / n, 1),
            "total_output_tokens": self.total_output_tokens,
        }


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class AdapterManager:
    def __init__(self, base_model: str = BASE_MODEL, load_in_4bit: bool | None = None):
        self.base_model_name = base_model
        self._lock = threading.Lock()
        self._loaded: set[str] = set()
        self._active: str | None = None
        self.stats: dict[str, AdapterStats] = {}
        self._peft_model = None
        self._tokenizer = None
        self._load_in_4bit = load_in_4bit
        self._load_base()

    # -- setup -------------------------------------------------------------
    def _load_base(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        use_4bit = self._load_in_4bit
        if use_4bit is None:
            use_4bit = torch.cuda.is_available()

        logger.info("Loading base model %s (4bit=%s)", self.base_model_name, use_4bit)
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs = {}
        if use_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self._base = AutoModelForCausalLM.from_pretrained(self.base_model_name, **kwargs)
        if not use_4bit and torch.cuda.is_available():
            self._base = self._base.cuda()
        self._device = next(self._base.parameters()).device

    # -- adapter lifecycle -------------------------------------------------
    def load_adapter(self, name: str) -> None:
        """Resident-load an adapter onto the shared base. Idempotent."""
        spec = get_adapter(name)
        if not (spec.output_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Adapter '{name}' not trained. Run: python training/train.py --adapter {name}"
            )
        with self._lock:
            if name in self._loaded:
                return
            from peft import PeftModel

            if self._peft_model is None:
                # First adapter wraps the base; subsequent ones attach to the same wrapper.
                self._peft_model = PeftModel.from_pretrained(
                    self._base, str(spec.output_dir), adapter_name=name
                )
            else:
                self._peft_model.load_adapter(str(spec.output_dir), adapter_name=name)
            self._loaded.add(name)
            self.stats.setdefault(name, AdapterStats())
            logger.info("Loaded adapter '%s' (resident: %s)", name, sorted(self._loaded))

    def unload_adapter(self, name: str) -> None:
        with self._lock:
            if name not in self._loaded or self._peft_model is None:
                return
            try:
                self._peft_model.delete_adapter(name)
            except Exception as e:  # pragma: no cover - peft version differences
                logger.warning("delete_adapter('%s') failed: %s", name, e)
            self._loaded.discard(name)
            if self._active == name:
                self._active = None

    # -- generation --------------------------------------------------------
    def generate(self, name: str, input_text: str, max_new_tokens: int = 320) -> GenResult:
        import torch

        if name not in self._loaded:
            self.load_adapter(name)  # lazy: pull it in on first request
        spec = get_adapter(name)

        with self._lock:  # set_adapter mutates shared state → serialize
            if self._active != name:
                self._peft_model.set_adapter(name)
                self._active = name

            prompt = prompts.format_inference_prompt(self._tokenizer, spec, input_text)
            inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._device)
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = self._peft_model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            latency_ms = (time.perf_counter() - t0) * 1000

            new_ids = gen[0][inputs["input_ids"].shape[1]:]
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            self.stats[name].record(latency_ms, len(new_ids))

        parsed = _parse_json(text) if spec.task_type == TASK_EXTRACTION else None
        return GenResult(
            adapter=name, output_text=text, parsed=parsed, latency_ms=round(latency_ms, 1),
            input_tokens=int(inputs["input_ids"].shape[1]), output_tokens=int(len(new_ids)),
        )

    # -- introspection -----------------------------------------------------
    def registry(self) -> list[dict]:
        out = []
        for name, spec in ADAPTERS.items():
            out.append({
                "name": name,
                "specialty": spec.specialty,
                "task_type": spec.task_type,
                "description": spec.description,
                "trained": (spec.output_dir / "adapter_config.json").exists(),
                "loaded": name in self._loaded,
                "active": name == self._active,
                "stats": self.stats.get(name, AdapterStats()).as_dict(),
            })
        return out

    def memory_summary(self) -> dict:
        import torch

        mem = {}
        if torch.cuda.is_available():
            mem["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            mem["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 1)
        mem["base_model"] = self.base_model_name
        mem["resident_adapters"] = sorted(self._loaded)
        mem["active_adapter"] = self._active
        return mem
