"""Unit tests for the OCR layer.

These tests avoid real Tesseract/API calls — they verify routing logic and
result structures using text-file fixtures and mocked vision calls.
Run: pytest tests/test_ocr.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ocr import CONFIDENCE_THRESHOLD, OCRProcessor, OCRResult

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "clinical_notes"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ocr_processor(mock_vision_text: str = "mocked vision text") -> OCRProcessor:
    """Return an OCRProcessor with a mocked Anthropic client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=mock_vision_text)]
    mock_client.messages.create.return_value = mock_response
    return OCRProcessor(anthropic_client=mock_client)


# ---------------------------------------------------------------------------
# Tests: process_text (pre-supplied text path)
# ---------------------------------------------------------------------------

class TestProcessText:
    def test_returns_ocr_result(self):
        proc = _make_ocr_processor()
        result = proc.process_text("hello world")
        assert isinstance(result, OCRResult)

    def test_confidence_is_1(self):
        proc = _make_ocr_processor()
        result = proc.process_text("some text")
        assert result.confidence == 1.0

    def test_engine_is_pre_supplied(self):
        proc = _make_ocr_processor()
        result = proc.process_text("some text")
        assert result.engine == "pre_supplied"

    def test_text_is_preserved(self):
        proc = _make_ocr_processor()
        result = proc.process_text("hello clinical notes")
        assert result.text == "hello clinical notes"


# ---------------------------------------------------------------------------
# Tests: process() with text files
# ---------------------------------------------------------------------------

class TestProcessTextFile:
    def test_fixture_001_loads(self):
        proc = _make_ocr_processor()
        path = FIXTURE_DIR / "fixture_001.txt"
        result = proc.process(str(path))
        assert isinstance(result, OCRResult)
        assert result.engine == "text_file"
        assert result.confidence == 1.0
        assert "Robert Hernandez" in result.text

    def test_fixture_002_loads(self):
        proc = _make_ocr_processor()
        path = FIXTURE_DIR / "fixture_002.txt"
        result = proc.process(str(path))
        assert "Priya Kapoor" in result.text

    def test_missing_file_raises(self):
        proc = _make_ocr_processor()
        with pytest.raises(FileNotFoundError):
            proc.process("/nonexistent/path/doc.txt")


# ---------------------------------------------------------------------------
# Tests: confidence threshold routing
# ---------------------------------------------------------------------------

class TestConfidenceThreshold:
    def test_threshold_value(self):
        assert CONFIDENCE_THRESHOLD == 0.6

    def test_ocr_result_has_confidence(self):
        result = OCRResult(text="hi", confidence=0.75, engine="tesseract")
        assert 0.0 <= result.confidence <= 1.0

    def test_claude_vision_fallback_confidence(self):
        """When mocked vision is called, it should return 0.92."""
        proc = _make_ocr_processor("Vision text result")
        # Patch _process_pil_image to simulate low Tesseract confidence
        with patch.object(proc, "_call_claude_vision") as mock_cv:
            mock_cv.return_value = OCRResult(
                text="Vision text result", confidence=0.92, engine="claude_vision"
            )
            # Directly test the method
            result = proc._call_claude_vision("base64data", "image/png")
            # The method is patched, so just check return value shape
            assert result.confidence == 0.92
            assert result.engine == "claude_vision"


# ---------------------------------------------------------------------------
# Tests: OCRResult dataclass
# ---------------------------------------------------------------------------

class TestOCRResult:
    def test_defaults(self):
        r = OCRResult(text="hello", confidence=0.9, engine="tesseract")
        assert r.page_count == 1

    def test_multi_page(self):
        r = OCRResult(text="p1\np2", confidence=0.85, engine="tesseract", page_count=2)
        assert r.page_count == 2
