"""Benchmark: base model vs. LoRA-adapted model on the same held-out clinical task.

We load the base ONCE, attach the adapter, and toggle it with PeftModel.disable_adapter().
That guarantees the "base" and "adapted" numbers come from the identical weights + prompt —
the only variable is the adapter. Anything else would not be a fair comparison.

Metrics per task:
  summarization → ROUGE-1 / ROUGE-2 / ROUGE-L
  extraction    → slot micro-F1 (modality, body_part, findings, measurements),
                  critical_flag accuracy (the clinically important one), impression ROUGE-L
Both report mean latency, tokens/sec, and an estimated $ / 1k inferences.

Usage:
    python training/eval.py --adapter cardiology-summary
    python training/eval.py --adapter radiology-extract --n 6 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import prompts  # noqa: E402
from common.config import (  # noqa: E402
    ADAPTERS,
    BASE_MODEL,
    REPO_ROOT,
    TASK_EXTRACTION,
    get_adapter,
)

# Rough on-demand cloud GPU rate (e.g. AWS g5.xlarge / A10G). Override with GPU_RATE_USD_HR.
GPU_RATE_USD_HR = 1.006


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def rouge(pred: str, ref: str) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(ref, pred)
    return {k: round(v.fmeasure, 4) for k, v in s.items()}


def _atoms(extraction: dict) -> set[str]:
    """Flatten an extraction dict into normalized field=value atoms for set-based F1."""
    atoms: set[str] = set()
    for field in ("modality", "body_part"):
        if extraction.get(field):
            atoms.add(f"{field}={_norm(extraction[field])}")
    for field in ("findings", "measurements"):
        for item in extraction.get(field, []) or []:
            atoms.add(f"{field}={_norm(item)}")
    return atoms


def extraction_metrics(preds: list[dict | None], golds: list[dict]) -> dict:
    tp = fp = fn = 0
    crit_correct = crit_total = 0
    impression_rouge = []
    parse_failures = 0
    for pred, gold in zip(preds, golds):
        if pred is None:
            parse_failures += 1
            fn += len(_atoms(gold))  # missed every slot
            crit_total += 1
            continue
        g, p = _atoms(gold), _atoms(pred)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        crit_total += 1
        if bool(pred.get("critical_flag")) == bool(gold.get("critical_flag")):
            crit_correct += 1
        impression_rouge.append(rouge(str(pred.get("impression", "")), str(gold.get("impression", "")))["rougeL"])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "slot_precision": round(precision, 4),
        "slot_recall": round(recall, 4),
        "slot_f1": round(f1, 4),
        "critical_flag_accuracy": round(crit_correct / crit_total, 4) if crit_total else 0.0,
        "impression_rougeL": round(sum(impression_rouge) / len(impression_rouge), 4) if impression_rouge else 0.0,
        "json_parse_failures": parse_failures,
    }


def summarization_metrics(preds: list[str], golds: list[str]) -> dict:
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, gold in zip(preds, golds):
        for k, v in rouge(pred, gold).items():
            agg[k] += v
    n = len(preds) or 1
    return {k: round(v / n, 4) for k, v in agg.items()}


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_all(model, tokenizer, spec, records, max_new_tokens: int) -> tuple[list[str], dict]:
    device = next(model.parameters()).device
    outputs, total_new_tokens, total_time = [], 0, 0.0
    for rec in records:
        prompt = prompts.format_inference_prompt(tokenizer, spec, rec[spec.input_key])
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        total_time += time.perf_counter() - t0
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        total_new_tokens += len(new_tokens)
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    perf = {
        "mean_latency_s": round(total_time / len(records), 3),
        "tokens_per_s": round(total_new_tokens / total_time, 1) if total_time else 0.0,
        "est_usd_per_1k_inferences": round((total_time / len(records)) * GPU_RATE_USD_HR / 3600 * 1000, 4),
    }
    return outputs, perf


def score(spec, outputs, records):
    golds = [rec[spec.target_key] for rec in records]
    if spec.task_type == TASK_EXTRACTION:
        preds = [_parse_json(o) for o in outputs]
        return extraction_metrics(preds, golds)
    return summarization_metrics(outputs, golds)


def evaluate(adapter_name: str, n: int | None, max_new_tokens: int) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_adapter(adapter_name)
    if not (spec.output_dir / "adapter_config.json").exists():
        sys.exit(f"No trained adapter at {spec.output_dir}. Run: python training/train.py --adapter {adapter_name}")

    records = json.loads(spec.data_path.read_text())
    # Held-out slice: last min(n, 25%) records. Tiny here; scale data with generate_data.py.
    holdout = max(1, len(records) // 4)
    records = records[-holdout:] if n is None else records[-n:]

    tokenizer = AutoTokenizer.from_pretrained(str(spec.output_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    if torch.cuda.is_available():
        base = base.cuda()
    model = PeftModel.from_pretrained(base, str(spec.output_dir))
    model.eval()

    print(f"== Eval '{spec.name}' on {len(records)} held-out examples ==")
    adapted_out, adapted_perf = generate_all(model, tokenizer, spec, records, max_new_tokens)
    with model.disable_adapter():  # same weights, adapter off → true base behavior
        base_out, base_perf = generate_all(model, tokenizer, spec, records, max_new_tokens)

    result = {
        "adapter": spec.name,
        "task_type": spec.task_type,
        "base_model": BASE_MODEL,
        "n_eval": len(records),
        "base": {"metrics": score(spec, base_out, records), "perf": base_perf},
        "adapted": {"metrics": score(spec, adapted_out, records), "perf": adapted_perf},
    }

    out_dir = REPO_ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{spec.name}.json").write_text(json.dumps(result, indent=2))
    _print_report(result)
    return result


def _print_report(r: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  {r['adapter']}  ({r['task_type']})   n={r['n_eval']}")
    print("=" * 60)
    keys = sorted(set(r["base"]["metrics"]) | set(r["adapted"]["metrics"]))
    print(f"  {'metric':<26}{'base':>12}{'adapted':>12}")
    for k in keys:
        b, a = r["base"]["metrics"].get(k, "-"), r["adapted"]["metrics"].get(k, "-")
        print(f"  {k:<26}{str(b):>12}{str(a):>12}")
    print("  " + "-" * 50)
    for k in ("mean_latency_s", "tokens_per_s", "est_usd_per_1k_inferences"):
        print(f"  {k:<26}{str(r['base']['perf'][k]):>12}{str(r['adapted']['perf'][k]):>12}")
    print("=" * 60 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, choices=list(ADAPTERS))
    ap.add_argument("--n", type=int, default=None, help="eval on last N records (default: 25%% holdout)")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()
    evaluate(args.adapter, args.n, args.max_new_tokens)


if __name__ == "__main__":
    main()
