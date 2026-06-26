"""Extraction layer: Claude Sonnet parses OCR text into structured clinical data.

Design decisions:
- Spec-first: prompt is built around the Pydantic schema, not vice versa.
- Source spans: every field includes [start, end] character indices in the OCR text.
- Confidence per field: downstream verification uses these to route low-confidence extractions.
- Sonnet (not Opus): balanced cost/quality for the extraction slot. Opus is reserved for verification.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import anthropic

from .schemas import (
    AssessmentPlan,
    ClinicalProgressNoteExtraction,
    DiagnosisInfo,
    ExtractedField,
    ExtractedFloat,
    ExtractedInt,
    ExtractedStr,
    MedicationInfo,
    PatientInfo,
    TokenUsage,
    VisitInfo,
    VitalsInfo,
)

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a clinical document extraction specialist. Your task is to extract structured data from clinical progress notes with high precision.

## Output Format

You MUST return valid JSON matching this exact schema. Every field must include:
- "value": the extracted value (or null if not found)
- "confidence": a float 0.0–1.0 representing your certainty
- "source_span": [start_char, end_char] pointing to the exact characters in the input text that support this extraction (or null if inferred/not found)

## Confidence Guidelines

- 0.95–1.00: Explicitly stated, unambiguous
- 0.80–0.94: Clearly implied or standard abbreviation
- 0.65–0.79: Inferred from context, some ambiguity
- 0.50–0.64: Best guess with significant uncertainty
- Below 0.50: Use null value instead

## Schema

```json
{
  "patient": {
    "name": {"value": "string|null", "confidence": 0.0, "source_span": [0, 0]},
    "dob": {"value": "string|null", "confidence": 0.0, "source_span": [0, 0]},
    "mrn": {"value": "string|null", "confidence": 0.0, "source_span": [0, 0]},
    "gender": {"value": "string|null", "confidence": 0.0, "source_span": [0, 0]},
    "insurance_id": {"value": "string|null", "confidence": 0.0, "source_span": null}
  },
  "visit": {
    "visit_date": {"value": "string|null", "confidence": 0.0, "source_span": [0, 0]},
    "provider_name": {"value": "string|null", "confidence": 0.0, "source_span": [0, 0]},
    "facility_name": {"value": "string|null", "confidence": 0.0, "source_span": [0, 0]},
    "chief_complaint": {"value": "string|null", "confidence": 0.0, "source_span": [0, 0]},
    "visit_type": {"value": "string|null", "confidence": 0.0, "source_span": null}
  },
  "vitals": {
    "blood_pressure_systolic": {"value": null, "confidence": 0.0, "source_span": null},
    "blood_pressure_diastolic": {"value": null, "confidence": 0.0, "source_span": null},
    "heart_rate": {"value": null, "confidence": 0.0, "source_span": null},
    "respiratory_rate": {"value": null, "confidence": 0.0, "source_span": null},
    "temperature": {"value": null, "confidence": 0.0, "source_span": null},
    "weight": {"value": null, "confidence": 0.0, "source_span": null},
    "height": {"value": null, "confidence": 0.0, "source_span": null},
    "bmi": {"value": null, "confidence": 0.0, "source_span": null},
    "oxygen_saturation": {"value": null, "confidence": 0.0, "source_span": null}
  },
  "medications": [
    {
      "name": {"value": "string", "confidence": 0.0, "source_span": [0, 0]},
      "dose": {"value": "string", "confidence": 0.0, "source_span": [0, 0]},
      "frequency": {"value": "string", "confidence": 0.0, "source_span": [0, 0]},
      "route": {"value": "string|null", "confidence": 0.0, "source_span": null},
      "indication": {"value": "string|null", "confidence": 0.0, "source_span": null}
    }
  ],
  "diagnoses": [
    {
      "description": {"value": "string", "confidence": 0.0, "source_span": [0, 0]},
      "icd10_code": {"value": "string|null", "confidence": 0.0, "source_span": null},
      "status": {"value": "string|null", "confidence": 0.0, "source_span": null}
    }
  ],
  "assessment_plan": {
    "assessment": {"value": "string", "confidence": 0.0, "source_span": [0, 0]},
    "plan": {"value": "string", "confidence": 0.0, "source_span": [0, 0]}
  }
}
```

## Rules

1. source_span indices are character positions in the INPUT text (0-indexed).
2. If a section is absent in the document, set all its fields to null with confidence 0.0.
3. For medications and diagnoses, return an empty array [] if none are found.
4. Never hallucinate values. If unsure, set value to null and confidence below 0.65.
5. Preserve original formatting of dates, names, and codes exactly as written.
6. Return ONLY valid JSON — no markdown, no explanation, no code fences."""


FEW_SHOT_EXAMPLE = """## Example

INPUT TEXT (abbreviated):
```
Patient: Jane Smith  DOB: 03/15/1968  MRN: 789012
Visit Date: 11/20/2024  Provider: Dr. Emily Chen
Chief Complaint: Follow-up for hypertension

Vitals: BP 138/88  HR 76  Temp 98.6F  Wt 165 lbs

Medications:
- Lisinopril 10mg PO daily (hypertension)

Assessment/Plan:
HTN – continue current regimen, recheck in 3 months.
```

EXPECTED OUTPUT:
```json
{
  "patient": {
    "name": {"value": "Jane Smith", "confidence": 0.99, "source_span": [9, 19]},
    "dob": {"value": "03/15/1968", "confidence": 0.99, "source_span": [26, 36]},
    "mrn": {"value": "789012", "confidence": 0.99, "source_span": [43, 49]},
    "gender": {"value": null, "confidence": 0.0, "source_span": null},
    "insurance_id": {"value": null, "confidence": 0.0, "source_span": null}
  },
  "visit": {
    "visit_date": {"value": "11/20/2024", "confidence": 0.99, "source_span": [62, 72]},
    "provider_name": {"value": "Dr. Emily Chen", "confidence": 0.99, "source_span": [84, 98]},
    "facility_name": {"value": null, "confidence": 0.0, "source_span": null},
    "chief_complaint": {"value": "Follow-up for hypertension", "confidence": 0.98, "source_span": [116, 142]},
    "visit_type": {"value": "follow-up", "confidence": 0.90, "source_span": [116, 125]}
  },
  "vitals": {
    "blood_pressure_systolic": {"value": 138, "confidence": 0.98, "source_span": [153, 159]},
    "blood_pressure_diastolic": {"value": 88, "confidence": 0.98, "source_span": [160, 162]},
    "heart_rate": {"value": 76, "confidence": 0.98, "source_span": [167, 169]},
    "respiratory_rate": {"value": null, "confidence": 0.0, "source_span": null},
    "temperature": {"value": 98.6, "confidence": 0.97, "source_span": [176, 181]},
    "weight": {"value": 165.0, "confidence": 0.97, "source_span": [187, 190]},
    "height": {"value": null, "confidence": 0.0, "source_span": null},
    "bmi": {"value": null, "confidence": 0.0, "source_span": null},
    "oxygen_saturation": {"value": null, "confidence": 0.0, "source_span": null}
  },
  "medications": [
    {
      "name": {"value": "Lisinopril", "confidence": 0.99, "source_span": [210, 220]},
      "dose": {"value": "10mg", "confidence": 0.99, "source_span": [221, 225]},
      "frequency": {"value": "daily", "confidence": 0.99, "source_span": [229, 234]},
      "route": {"value": "oral", "confidence": 0.95, "source_span": [226, 228]},
      "indication": {"value": "hypertension", "confidence": 0.97, "source_span": [236, 248]}
    }
  ],
  "diagnoses": [
    {
      "description": {"value": "Hypertension", "confidence": 0.97, "source_span": [262, 265]},
      "icd10_code": {"value": null, "confidence": 0.0, "source_span": null},
      "status": {"value": "active", "confidence": 0.85, "source_span": null}
    }
  ],
  "assessment_plan": {
    "assessment": {"value": "HTN – continue current regimen, recheck in 3 months.", "confidence": 0.96, "source_span": [261, 313]},
    "plan": {"value": "continue current regimen, recheck in 3 months", "confidence": 0.95, "source_span": [270, 313]}
  }
}
```"""


# ---------------------------------------------------------------------------
# Extraction agent
# ---------------------------------------------------------------------------

class ExtractionAgent:
    """Extracts structured clinical data from OCR text using Claude Sonnet."""

    def __init__(self, anthropic_client: anthropic.Anthropic | None = None):
        self._client = anthropic_client or anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    def extract_clinical_progress_note(
        self,
        ocr_text: str,
        doc_id: str,
        tenant_id: str,
        ocr_engine: str = "tesseract",
        ocr_confidence: float = 1.0,
    ) -> tuple[ClinicalProgressNoteExtraction, TokenUsage]:
        """Extract structured data from a clinical progress note.

        Returns the extraction and token usage (for cost tracking).
        """
        raw_json = self._call_llm(ocr_text)
        data = self._parse_response(raw_json)
        extraction = self._build_extraction(
            data, doc_id, tenant_id, ocr_engine, ocr_confidence
        )
        token_usage = TokenUsage(
            input_tokens=self._last_input_tokens,
            output_tokens=self._last_output_tokens,
            model=EXTRACTION_MODEL,
        )
        return extraction, token_usage

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _last_input_tokens: int = 0
    _last_output_tokens: int = 0

    def _call_llm(self, ocr_text: str) -> str:
        system = f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLE}"
        response = self._client.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": ocr_text}],
        )
        self._last_input_tokens = response.usage.input_tokens
        self._last_output_tokens = response.usage.output_tokens
        return response.content[0].text

    def _parse_response(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        # Strip accidental markdown fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON response: %s\nRaw: %s", exc, raw[:500])
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

    def _build_extraction(
        self,
        data: dict,
        doc_id: str,
        tenant_id: str,
        ocr_engine: str,
        ocr_confidence: float,
    ) -> ClinicalProgressNoteExtraction:
        patient = self._build_patient(data.get("patient", {}))
        visit = self._build_visit(data.get("visit", {}))
        vitals = self._build_vitals(data.get("vitals")) if data.get("vitals") else None
        medications = [self._build_medication(m) for m in data.get("medications", [])]
        diagnoses = [self._build_diagnosis(d) for d in data.get("diagnoses", [])]
        assessment_plan = self._build_assessment_plan(data.get("assessment_plan"))

        overall_confidence = self._compute_overall_confidence(
            patient, visit, vitals, medications, diagnoses
        )

        return ClinicalProgressNoteExtraction(
            doc_id=doc_id,
            tenant_id=tenant_id,
            ocr_engine=ocr_engine,
            ocr_confidence=ocr_confidence,
            patient=patient,
            visit=visit,
            vitals=vitals,
            medications=medications,
            diagnoses=diagnoses,
            assessment_plan=assessment_plan,
            overall_confidence=overall_confidence,
        )

    # -- field builders --

    @staticmethod
    def _ef(raw: dict | None, cls=ExtractedStr) -> Any:
        """Build an ExtractedField subclass from raw dict."""
        if not raw:
            return cls(value=None, confidence=0.0, source_span=None)
        return cls(
            value=raw.get("value"),
            confidence=float(raw.get("confidence", 0.0)),
            source_span=raw.get("source_span"),
        )

    def _build_patient(self, d: dict) -> PatientInfo:
        return PatientInfo(
            name=self._ef(d.get("name")),
            dob=self._ef(d.get("dob")),
            mrn=self._ef(d.get("mrn")),
            gender=self._ef(d.get("gender")),
            insurance_id=self._ef(d.get("insurance_id")) if d.get("insurance_id") else None,
        )

    def _build_visit(self, d: dict) -> VisitInfo:
        return VisitInfo(
            visit_date=self._ef(d.get("visit_date")),
            provider_name=self._ef(d.get("provider_name")),
            facility_name=self._ef(d.get("facility_name")),
            chief_complaint=self._ef(d.get("chief_complaint")),
            visit_type=self._ef(d.get("visit_type")) if d.get("visit_type") else None,
        )

    def _build_vitals(self, d: dict | None) -> VitalsInfo | None:
        if not d:
            return None
        return VitalsInfo(
            blood_pressure_systolic=self._ef(d.get("blood_pressure_systolic"), ExtractedInt),
            blood_pressure_diastolic=self._ef(d.get("blood_pressure_diastolic"), ExtractedInt),
            heart_rate=self._ef(d.get("heart_rate"), ExtractedInt),
            respiratory_rate=self._ef(d.get("respiratory_rate"), ExtractedInt),
            temperature=self._ef(d.get("temperature"), ExtractedFloat),
            weight=self._ef(d.get("weight"), ExtractedFloat),
            height=self._ef(d.get("height"), ExtractedFloat),
            bmi=self._ef(d.get("bmi"), ExtractedFloat),
            oxygen_saturation=self._ef(d.get("oxygen_saturation"), ExtractedFloat),
        )

    def _build_medication(self, d: dict) -> MedicationInfo:
        return MedicationInfo(
            name=self._ef(d.get("name")),
            dose=self._ef(d.get("dose")),
            frequency=self._ef(d.get("frequency")),
            route=self._ef(d.get("route")) if d.get("route") else None,
            indication=self._ef(d.get("indication")) if d.get("indication") else None,
        )

    def _build_diagnosis(self, d: dict) -> DiagnosisInfo:
        return DiagnosisInfo(
            description=self._ef(d.get("description")),
            icd10_code=self._ef(d.get("icd10_code")) if d.get("icd10_code") else None,
            status=self._ef(d.get("status")) if d.get("status") else None,
        )

    def _build_assessment_plan(self, d: dict | None) -> AssessmentPlan | None:
        if not d:
            return None
        return AssessmentPlan(
            assessment=self._ef(d.get("assessment")),
            plan=self._ef(d.get("plan")),
        )

    @staticmethod
    def _confidence_values(field: ExtractedField | None) -> list[float]:
        if field is None:
            return []
        return [field.confidence] if field.confidence > 0 else []

    def _compute_overall_confidence(
        self,
        patient: PatientInfo,
        visit: VisitInfo,
        vitals: VitalsInfo | None,
        medications: list[MedicationInfo],
        diagnoses: list[DiagnosisInfo],
    ) -> float:
        scores: list[float] = []

        # High-weight: patient identity and visit metadata
        for field in [patient.name, patient.dob, patient.mrn, visit.visit_date,
                      visit.provider_name, visit.chief_complaint]:
            scores.extend(self._confidence_values(field))

        # Medium-weight: medications
        for med in medications:
            for field in [med.name, med.dose, med.frequency]:
                scores.extend(self._confidence_values(field))

        # Medium-weight: diagnoses
        for dx in diagnoses:
            scores.extend(self._confidence_values(dx.description))

        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 4)
