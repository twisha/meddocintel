"""Pydantic schemas for clinical document extraction.

These are the spec. Everything downstream (prompts, validation, serialization) flows from here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Core building block: every extracted value carries confidence + provenance
# ---------------------------------------------------------------------------

class ExtractedField(BaseModel):
    """A single extracted value with confidence score and source span."""

    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source_span: Optional[tuple[int, int]] = None  # [start_char, end_char] in OCR text

    @field_validator("source_span", mode="before")
    @classmethod
    def coerce_span(cls, v):
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        raise ValueError("source_span must be [start, end]")


class ExtractedStr(ExtractedField):
    value: Optional[str] = None


class ExtractedFloat(ExtractedField):
    value: Optional[float] = None


class ExtractedInt(ExtractedField):
    value: Optional[int] = None


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class PatientInfo(BaseModel):
    name: ExtractedStr
    dob: ExtractedStr              # Date of birth (string, preserve original format)
    mrn: ExtractedStr              # Medical record number
    gender: ExtractedStr
    insurance_id: Optional[ExtractedStr] = None


class VisitInfo(BaseModel):
    visit_date: ExtractedStr
    provider_name: ExtractedStr
    facility_name: ExtractedStr
    chief_complaint: ExtractedStr
    visit_type: Optional[ExtractedStr] = None  # e.g. "follow-up", "new patient"


class VitalsInfo(BaseModel):
    blood_pressure_systolic: Optional[ExtractedInt] = None
    blood_pressure_diastolic: Optional[ExtractedInt] = None
    heart_rate: Optional[ExtractedInt] = None       # bpm
    respiratory_rate: Optional[ExtractedInt] = None  # breaths/min
    temperature: Optional[ExtractedFloat] = None    # Fahrenheit
    weight: Optional[ExtractedFloat] = None         # lbs
    height: Optional[ExtractedFloat] = None         # inches
    bmi: Optional[ExtractedFloat] = None
    oxygen_saturation: Optional[ExtractedFloat] = None  # percent


class MedicationInfo(BaseModel):
    name: ExtractedStr
    dose: ExtractedStr
    frequency: ExtractedStr
    route: Optional[ExtractedStr] = None       # e.g. oral, IV
    indication: Optional[ExtractedStr] = None


class DiagnosisInfo(BaseModel):
    description: ExtractedStr
    icd10_code: Optional[ExtractedStr] = None
    status: Optional[ExtractedStr] = None  # e.g. "active", "resolved", "chronic"


class AssessmentPlan(BaseModel):
    assessment: ExtractedStr
    plan: ExtractedStr


# ---------------------------------------------------------------------------
# Top-level extraction result
# ---------------------------------------------------------------------------

class ExtractionConfidence(str, Enum):
    HIGH = "high"       # > 0.85 — accept
    MEDIUM = "medium"   # 0.70–0.85 — flag for review
    LOW = "low"         # < 0.70 — reject / manual


class ClinicalProgressNoteExtraction(BaseModel):
    """Top-level result for a clinical progress note extraction."""

    doc_id: str
    tenant_id: str
    document_type: str = "clinical_progress_note"
    ocr_engine: str                     # "tesseract" | "claude_vision"
    ocr_confidence: float

    patient: PatientInfo
    visit: VisitInfo
    vitals: Optional[VitalsInfo] = None
    medications: list[MedicationInfo] = Field(default_factory=list)
    diagnoses: list[DiagnosisInfo] = Field(default_factory=list)
    assessment_plan: Optional[AssessmentPlan] = None

    overall_confidence: float = Field(ge=0.0, le=1.0)

    @property
    def confidence_tier(self) -> ExtractionConfidence:
        if self.overall_confidence > 0.85:
            return ExtractionConfidence.HIGH
        if self.overall_confidence > 0.70:
            return ExtractionConfidence.MEDIUM
        return ExtractionConfidence.LOW


# ---------------------------------------------------------------------------
# API request/response models
# ---------------------------------------------------------------------------

class ExtractionRequest(BaseModel):
    doc_id: str
    tenant_id: str
    file_path: Optional[str] = None
    ocr_text: Optional[str] = None    # Pre-supplied text (skip OCR)
    document_type: str = "clinical_progress_note"


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate based on public Sonnet pricing."""
        # claude-sonnet-4-6: $3/M input, $15/M output
        return (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000


class ExtractionResponse(BaseModel):
    extraction: ClinicalProgressNoteExtraction
    token_usage: TokenUsage
    confidence_tier: str
