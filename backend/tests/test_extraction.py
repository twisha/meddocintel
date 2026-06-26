"""Unit tests for the extraction layer.

These tests use a mock Anthropic client so they don't make real API calls.
Run: pytest tests/test_extraction.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.extraction import ExtractionAgent
from src.schemas import (
    ClinicalProgressNoteExtraction,
    ExtractionConfidence,
    TokenUsage,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "clinical_notes"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(response_json: dict) -> MagicMock:
    """Build a minimal Anthropic client mock that returns canned JSON."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(response_json))]
    mock_response.usage.input_tokens = 800
    mock_response.usage.output_tokens = 300
    mock_client.messages.create.return_value = mock_response
    return mock_client


CANNED_EXTRACTION = {
    "patient": {
        "name": {"value": "Robert Hernandez", "confidence": 0.99, "source_span": [74, 90]},
        "dob": {"value": "07/22/1955", "confidence": 0.99, "source_span": [110, 120]},
        "mrn": {"value": "4821039", "confidence": 0.99, "source_span": [137, 144]},
        "gender": {"value": "Male", "confidence": 0.99, "source_span": [166, 170]},
        "insurance_id": {"value": "BCA-7731-00293", "confidence": 0.97, "source_span": [184, 198]},
    },
    "visit": {
        "visit_date": {"value": "03/10/2025", "confidence": 0.99, "source_span": [213, 223]},
        "provider_name": {"value": "Dr. Sarah Patel, MD", "confidence": 0.99, "source_span": [258, 277]},
        "facility_name": {"value": "Metro Internal Medicine", "confidence": 0.98, "source_span": [289, 312]},
        "chief_complaint": {"value": "Follow-up for Type 2 Diabetes and hypertension; patient also reports worsening fatigue over past 3 weeks.", "confidence": 0.98, "source_span": [330, 432]},
        "visit_type": {"value": "follow-up", "confidence": 0.97, "source_span": [237, 246]},
    },
    "vitals": {
        "blood_pressure_systolic": {"value": 148, "confidence": 0.99, "source_span": [447, 450]},
        "blood_pressure_diastolic": {"value": 92, "confidence": 0.99, "source_span": [451, 453]},
        "heart_rate": {"value": 82, "confidence": 0.99, "source_span": [468, 470]},
        "respiratory_rate": {"value": 16, "confidence": 0.99, "source_span": [490, 492]},
        "temperature": {"value": 98.9, "confidence": 0.98, "source_span": [510, 514]},
        "weight": {"value": 214.0, "confidence": 0.98, "source_span": [527, 530]},
        "height": {"value": 70.0, "confidence": 0.98, "source_span": [540, 542]},
        "bmi": {"value": 30.7, "confidence": 0.97, "source_span": [554, 558]},
        "oxygen_saturation": {"value": 97.0, "confidence": 0.97, "source_span": [571, 573]},
    },
    "medications": [
        {
            "name": {"value": "Metformin", "confidence": 0.99, "source_span": [600, 609]},
            "dose": {"value": "1000mg", "confidence": 0.99, "source_span": [610, 616]},
            "frequency": {"value": "twice daily", "confidence": 0.99, "source_span": [620, 631]},
            "route": {"value": "oral", "confidence": 0.95, "source_span": [617, 619]},
            "indication": {"value": "Type 2 Diabetes", "confidence": 0.97, "source_span": [635, 650]},
        },
        {
            "name": {"value": "Lisinopril", "confidence": 0.99, "source_span": [657, 667]},
            "dose": {"value": "20mg", "confidence": 0.99, "source_span": [668, 672]},
            "frequency": {"value": "once daily", "confidence": 0.99, "source_span": [676, 686]},
            "route": {"value": "oral", "confidence": 0.95, "source_span": [673, 675]},
            "indication": {"value": "Hypertension", "confidence": 0.97, "source_span": [690, 702]},
        },
    ],
    "diagnoses": [
        {
            "description": {"value": "Type 2 Diabetes Mellitus", "confidence": 0.99, "source_span": [760, 784]},
            "icd10_code": {"value": "E11.9", "confidence": 0.99, "source_span": [786, 791]},
            "status": {"value": "active", "confidence": 0.97, "source_span": [796, 802]},
        },
        {
            "description": {"value": "Hypertension, essential", "confidence": 0.98, "source_span": [808, 831]},
            "icd10_code": {"value": "I10", "confidence": 0.99, "source_span": [833, 836]},
            "status": {"value": "active", "confidence": 0.95, "source_span": [841, 847]},
        },
    ],
    "assessment_plan": {
        "assessment": {"value": "HbA1c 8.4% above target; BP poorly controlled; fatigue under evaluation.", "confidence": 0.95, "source_span": [950, 1020]},
        "plan": {"value": "Increase Metformin; increase Lisinopril; add HCTZ; order labs for fatigue.", "confidence": 0.93, "source_span": [1025, 1100]},
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractionAgent:
    def _agent(self) -> ExtractionAgent:
        return ExtractionAgent(anthropic_client=_make_mock_client(CANNED_EXTRACTION))

    def test_returns_correct_type(self):
        agent = self._agent()
        extraction, usage = agent.extract_clinical_progress_note(
            ocr_text="dummy text", doc_id="test-001", tenant_id="demo"
        )
        assert isinstance(extraction, ClinicalProgressNoteExtraction)
        assert isinstance(usage, TokenUsage)

    def test_patient_fields(self):
        agent = self._agent()
        extraction, _ = agent.extract_clinical_progress_note("x", "t1", "d1")
        assert extraction.patient.name.value == "Robert Hernandez"
        assert extraction.patient.mrn.value == "4821039"
        assert extraction.patient.dob.value == "07/22/1955"

    def test_visit_fields(self):
        agent = self._agent()
        extraction, _ = agent.extract_clinical_progress_note("x", "t1", "d1")
        assert extraction.visit.visit_date.value == "03/10/2025"
        assert "Sarah Patel" in extraction.visit.provider_name.value

    def test_vitals_present(self):
        agent = self._agent()
        extraction, _ = agent.extract_clinical_progress_note("x", "t1", "d1")
        assert extraction.vitals is not None
        assert extraction.vitals.blood_pressure_systolic.value == 148
        assert extraction.vitals.blood_pressure_diastolic.value == 92
        assert extraction.vitals.heart_rate.value == 82

    def test_medications_extracted(self):
        agent = self._agent()
        extraction, _ = agent.extract_clinical_progress_note("x", "t1", "d1")
        assert len(extraction.medications) == 2
        assert extraction.medications[0].name.value == "Metformin"
        assert extraction.medications[1].name.value == "Lisinopril"

    def test_diagnoses_extracted(self):
        agent = self._agent()
        extraction, _ = agent.extract_clinical_progress_note("x", "t1", "d1")
        assert len(extraction.diagnoses) == 2
        codes = [d.icd10_code.value for d in extraction.diagnoses if d.icd10_code]
        assert "E11.9" in codes

    def test_source_spans_are_tuples(self):
        agent = self._agent()
        extraction, _ = agent.extract_clinical_progress_note("x", "t1", "d1")
        span = extraction.patient.name.source_span
        assert span is not None
        assert isinstance(span, tuple)
        assert span[0] < span[1]

    def test_confidence_scores_in_range(self):
        agent = self._agent()
        extraction, _ = agent.extract_clinical_progress_note("x", "t1", "d1")
        assert 0.0 <= extraction.patient.name.confidence <= 1.0
        assert 0.0 <= extraction.overall_confidence <= 1.0

    def test_high_confidence_tier(self):
        agent = self._agent()
        extraction, _ = agent.extract_clinical_progress_note("x", "t1", "d1")
        # Canned data has high confidence values — should land in HIGH tier
        assert extraction.confidence_tier == ExtractionConfidence.HIGH

    def test_token_usage(self):
        agent = self._agent()
        _, usage = agent.extract_clinical_progress_note("x", "t1", "d1")
        assert usage.input_tokens == 800
        assert usage.output_tokens == 300
        assert usage.model == "claude-sonnet-4-6"
        assert usage.estimated_cost_usd > 0

    def test_doc_id_and_tenant_propagated(self):
        agent = self._agent()
        extraction, _ = agent.extract_clinical_progress_note(
            "x", doc_id="my-doc-123", tenant_id="tenant-abc"
        )
        assert extraction.doc_id == "my-doc-123"
        assert extraction.tenant_id == "tenant-abc"

    def test_empty_medications_list(self):
        """LLM returns empty medications array → should parse cleanly."""
        canned = {**CANNED_EXTRACTION, "medications": []}
        agent = ExtractionAgent(anthropic_client=_make_mock_client(canned))
        extraction, _ = agent.extract_clinical_progress_note("x", "t1", "d1")
        assert extraction.medications == []

    def test_missing_vitals_returns_none(self):
        canned = {**CANNED_EXTRACTION, "vitals": None}
        agent = ExtractionAgent(anthropic_client=_make_mock_client(canned))
        extraction, _ = agent.extract_clinical_progress_note("x", "t1", "d1")
        assert extraction.vitals is None

    def test_invalid_json_raises(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="this is not json {{{")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        agent = ExtractionAgent(anthropic_client=mock_client)
        with pytest.raises(ValueError, match="invalid JSON"):
            agent.extract_clinical_progress_note("x", "t1", "d1")


class TestExtractionOnFixture:
    """Integration-style tests: parse actual fixture files (no API calls)."""

    def test_fixture_001_is_readable(self):
        path = FIXTURE_DIR / "fixture_001.txt"
        assert path.exists(), f"Fixture not found: {path}"
        text = path.read_text()
        assert "Robert Hernandez" in text
        assert "Metformin" in text

    def test_fixture_002_is_readable(self):
        path = FIXTURE_DIR / "fixture_002.txt"
        assert path.exists(), f"Fixture not found: {path}"
        text = path.read_text()
        assert "Priya Kapoor" in text
        assert "Lisinopril" not in text  # Fixture 002 doesn't have Lisinopril
        assert "Loratadine" in text
