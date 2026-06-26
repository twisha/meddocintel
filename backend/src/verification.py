"""Verification layer: Opus-as-judge + rule-based gates.

Two-stage process:
  Stage 1 — Rule gates (deterministic, free):
    - Schema completeness (required fields present)
    - Vital signs range checks
    - Date logic (visit date not in the future)

  Stage 2 — Opus judge (expensive, sampled):
    - Only runs when extraction confidence < 0.85 OR a rule gate fires
    - Scores each field independently against the original OCR text
    - Returns per-field scores + overall recommendation

Verdict:
  ACCEPT  — overall_score > 0.85 and no rule flags
  FLAG    — 0.70 < overall_score <= 0.85 or minor rule flags → manual review
  REJECT  — overall_score <= 0.70 or critical rule flags
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import anthropic

logger = logging.getLogger(__name__)

VERIFICATION_MODEL = "claude-opus-4-8"
ACCEPT_THRESHOLD = 0.85
REJECT_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Rule gates
# ---------------------------------------------------------------------------

VITAL_RANGES = {
    "blood_pressure_systolic": (60, 250),
    "blood_pressure_diastolic": (30, 150),
    "heart_rate": (30, 220),
    "respiratory_rate": (6, 60),
    "temperature": (90.0, 108.0),   # Fahrenheit
    "weight": (5.0, 700.0),         # lbs
    "bmi": (10.0, 80.0),
    "oxygen_saturation": (50.0, 100.0),
}


def run_rule_gates(extraction: dict) -> list[str]:
    """Return list of rule violation strings (empty = all clear)."""
    flags: list[str] = []

    # Required fields present
    patient = extraction.get("patient", {})
    for req in ["name", "dob", "mrn"]:
        field_val = patient.get(req, {})
        if not field_val or field_val.get("value") is None:
            flags.append(f"missing_required_field:patient.{req}")

    visit = extraction.get("visit", {})
    for req in ["visit_date", "provider_name", "chief_complaint"]:
        field_val = visit.get(req, {})
        if not field_val or field_val.get("value") is None:
            flags.append(f"missing_required_field:visit.{req}")

    # Visit date not in the future
    visit_date_str = visit.get("visit_date", {}).get("value")
    if visit_date_str:
        try:
            for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
                try:
                    vd = datetime.strptime(visit_date_str, fmt).replace(tzinfo=timezone.utc)
                    if vd > datetime.now(timezone.utc):
                        flags.append("invalid:visit_date_in_future")
                    break
                except ValueError:
                    continue
        except Exception:
            pass

    # Vital sign range checks
    vitals = extraction.get("vitals") or {}
    for vital_name, (lo, hi) in VITAL_RANGES.items():
        val = vitals.get(vital_name, {})
        if val and val.get("value") is not None:
            v = val["value"]
            if not (lo <= v <= hi):
                flags.append(f"out_of_range:{vital_name}={v} (expected {lo}-{hi})")

    return flags


# ---------------------------------------------------------------------------
# Opus judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a clinical extraction quality auditor. You will receive:
1. The original OCR text of a clinical document
2. The structured extraction produced by an AI system

Your job is to score each extracted field for accuracy against the source text.

For each top-level section return a score 0.0-1.0:
- 1.0: All fields match the source text exactly
- 0.8-0.99: Minor formatting differences, no semantic errors
- 0.6-0.79: Some fields missing or slightly inaccurate
- 0.4-0.59: Significant inaccuracies
- 0.0-0.39: Major errors or hallucinations

Return ONLY valid JSON with this structure:
{
  "field_scores": {
    "patient": 0.0,
    "visit": 0.0,
    "vitals": 0.0,
    "medications": 0.0,
    "diagnoses": 0.0,
    "assessment_plan": 0.0
  },
  "overall_score": 0.0,
  "issues": ["list of specific issues found, or empty array"],
  "recommendation": "ACCEPT | FLAG | REJECT"
}"""


def run_opus_judge(
    ocr_text: str,
    extraction: dict,
    client: anthropic.Anthropic,
) -> dict:
    """Call Claude Opus to score an extraction against the source OCR text."""
    user_message = (
        f"SOURCE OCR TEXT:\n```\n{ocr_text[:6000]}\n```\n\n"
        f"EXTRACTION:\n```json\n{json.dumps(extraction, indent=2)[:4000]}\n```"
    )
    response = client.messages.create(
        model=VERIFICATION_MODEL,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Opus judge returned invalid JSON: %s", raw[:300])
        return {
            "field_scores": {},
            "overall_score": 0.5,
            "issues": ["judge_parse_error"],
            "recommendation": "FLAG",
        }


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    verdict: str                          # ACCEPT | FLAG | REJECT
    overall_score: float
    field_scores: dict[str, float]
    rule_flags: list[str]
    ran_opus: bool
    model: str = VERIFICATION_MODEL


def verify(
    ocr_text: str,
    extraction: dict,
    extraction_confidence: float,
    client: anthropic.Anthropic | None = None,
) -> VerificationResult:
    """Run the full two-stage verification pipeline."""
    rule_flags = run_rule_gates(extraction)
    has_rule_flags = len(rule_flags) > 0

    # Decide whether to run Opus:
    # - Always run if confidence is below ACCEPT threshold
    # - Always run if rule gates fired
    # - Skip (cost saving) when extraction is already high-confidence and clean
    run_opus = extraction_confidence < ACCEPT_THRESHOLD or has_rule_flags

    if run_opus:
        if client is None:
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        judge_result = run_opus_judge(ocr_text, extraction, client)
        overall_score = judge_result.get("overall_score", 0.5)
        field_scores = judge_result.get("field_scores", {})
        logger.info(
            "Opus judge: score=%.2f recommendation=%s issues=%s",
            overall_score,
            judge_result.get("recommendation"),
            judge_result.get("issues"),
        )
    else:
        # High-confidence, no rule flags → accept without Opus (cost saving)
        overall_score = extraction_confidence
        field_scores = {}
        logger.info("Skipping Opus judge — high confidence + no rule flags")

    # Determine final verdict
    if rule_flags and any("missing_required_field" in f for f in rule_flags):
        verdict = "REJECT"
    elif overall_score > ACCEPT_THRESHOLD and not has_rule_flags:
        verdict = "ACCEPT"
    elif overall_score > REJECT_THRESHOLD:
        verdict = "FLAG"
    else:
        verdict = "REJECT"

    return VerificationResult(
        verdict=verdict,
        overall_score=overall_score,
        field_scores=field_scores,
        rule_flags=rule_flags,
        ran_opus=run_opus,
    )
