"""Synthetic clinical training data — Claude as the ground-truth generator.

MIMIC-III requires PhysioNet credentialing and a signed DUA, so it is not a drop-in
public download. Rather than block on access, we generate synthetic notes with Claude:
it writes a realistic clinical note AND the gold target (summary or extraction) in one
shot, so the pair is self-consistent. The committed seed files act as few-shot anchors so
the generated distribution matches the format the adapters are trained to produce.

Synthetic data is intentional here, not a fallback: it carries no PHI, is freely
shareable in a portfolio repo, and lets us control the label schema exactly.

Usage:
    export ANTHROPIC_API_KEY=...
    python data/generate_data.py --adapter cardiology-summary --n 40
    python data/generate_data.py --adapter radiology-extract  --n 40 --append
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import TASK_EXTRACTION, get_adapter  # noqa: E402

# Match MedDocIntel's model slot: Sonnet is the cost/quality sweet spot for generation.
GEN_MODEL = "claude-sonnet-4-6"

_TASK_BRIEF = {
    "cardiology-summary": (
        "Write a realistic, de-identified cardiology note (consult, progress, or clinic "
        "note) and its structured summary. Vary the scenario: ACS, heart failure, "
        "arrhythmia, valvular disease, cardiomyopathy, pericarditis, preop eval, etc. "
        "Use realistic vitals, labs, ECG/echo findings, and medications. No real patient data."
    ),
    "radiology-extract": (
        "Write a realistic, de-identified radiology report and its structured extraction. "
        "Vary modality (CT, MRI, X-ray, ultrasound, mammogram) and body part. Include some "
        "critical findings (PE, stroke, malignancy, free air) and some normal/benign studies. "
        "No real patient data."
    ),
}


def _build_prompt(spec, examples: list[dict], batch: int) -> str:
    keys = f'"{spec.input_key}" and "{spec.target_key}"'
    shots = json.dumps(examples[:2], indent=2, ensure_ascii=False)
    target_shape = (
        'an object with keys: modality, body_part, findings (list), impression (string), '
        'measurements (list), critical_flag (boolean)'
        if spec.task_type == TASK_EXTRACTION
        else 'a string with sections Presentation, Key Findings, Assessment, Plan'
    )
    return (
        f"{_TASK_BRIEF[spec.name]}\n\n"
        f"Each item is a JSON object with keys {keys}, where '{spec.target_key}' is "
        f"{target_shape}.\n\n"
        f"Here are two reference examples (match this style and schema exactly):\n{shots}\n\n"
        f"Generate {batch} NEW, diverse examples. Return ONLY a JSON array of {batch} "
        f"objects, no prose, no markdown fences."
    )


def _parse_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)  # strip fences if present
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array in model output:\n{text[:400]}")
    return json.loads(text[start : end + 1])


def generate(adapter_name: str, n: int, append: bool, batch: int = 8) -> Path:
    import anthropic  # local import so --help works without the dep installed

    spec = get_adapter(adapter_name)
    existing = json.loads(spec.data_path.read_text()) if spec.data_path.exists() else []
    seed = existing[:]  # few-shot anchors come from the committed seeds

    client = anthropic.Anthropic()
    out: list[dict] = existing[:] if append else []
    next_idx = len(out)

    while len(out) - (len(existing) if append else 0) < n:
        want = min(batch, n - (len(out) - (len(existing) if append else 0)))
        msg = client.messages.create(
            model=GEN_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": _build_prompt(spec, seed, want)}],
        )
        items = _parse_array(msg.content[0].text)
        for item in items:
            if spec.input_key not in item or spec.target_key not in item:
                continue  # skip malformed
            next_idx += 1
            item["id"] = f"{spec.specialty[:4]}-gen-{next_idx:04d}"
            out.append(item)
        print(f"  generated {len(out) - (len(existing) if append else 0)}/{n} ...", flush=True)

    spec.data_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {len(out)} records to {spec.data_path}")
    return spec.data_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, choices=["cardiology-summary", "radiology-extract"])
    ap.add_argument("--n", type=int, default=40, help="number of NEW examples to generate")
    ap.add_argument("--append", action="store_true", help="append to existing file instead of overwrite")
    args = ap.parse_args()
    generate(args.adapter, args.n, args.append)


if __name__ == "__main__":
    main()
