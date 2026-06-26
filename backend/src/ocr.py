"""OCR layer: Tesseract first, Claude Vision fallback.

Decision boundary: escalate to Claude Vision when Tesseract confidence < 0.6.
Cost: ~$0.001/page (Tesseract) vs ~$0.01/page (Claude Vision).
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

# Tesseract is an optional dependency — import lazily so unit tests can run without it
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


CONFIDENCE_THRESHOLD = 0.6  # Below this → escalate to Claude Vision


@dataclass
class OCRResult:
    text: str
    confidence: float   # 0-1
    engine: str         # "tesseract" | "claude_vision"
    page_count: int = 1


class OCRProcessor:
    """Two-tier OCR: Tesseract (cheap) → Claude Vision (expensive fallback)."""

    def __init__(self, anthropic_client: anthropic.Anthropic | None = None):
        self._client = anthropic_client or anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, file_path: str | Path) -> OCRResult:
        """Process a document, escalating from Tesseract to Claude Vision if needed."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._read_text_file(path)

        if suffix == ".pdf":
            return self._process_pdf(path)

        # Image formats
        return self._process_image(path)

    def process_text(self, text: str) -> OCRResult:
        """Wrap pre-supplied text as a high-confidence OCR result (skip OCR stage)."""
        return OCRResult(text=text, confidence=1.0, engine="pre_supplied")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_text_file(self, path: Path) -> OCRResult:
        text = path.read_text(encoding="utf-8")
        return OCRResult(text=text, confidence=1.0, engine="text_file")

    def _process_pdf(self, path: Path) -> OCRResult:
        """Convert PDF pages to images, then OCR each page."""
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise RuntimeError("pdf2image not installed. Run: pip install pdf2image")

        images = convert_from_path(str(path))
        page_results: list[OCRResult] = [self._process_pil_image(img) for img in images]

        combined_text = "\n\n--- PAGE BREAK ---\n\n".join(r.text for r in page_results)
        avg_confidence = sum(r.confidence for r in page_results) / len(page_results)

        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            engine=page_results[0].engine,
            page_count=len(images),
        )

    def _process_image(self, path: Path) -> OCRResult:
        if not TESSERACT_AVAILABLE:
            logger.warning("pytesseract not available; falling back to Claude Vision")
            return self._claude_vision_ocr_path(path)

        try:
            img = Image.open(path)
            return self._process_pil_image(img)
        except Exception as exc:
            logger.warning("Tesseract failed (%s); escalating to Claude Vision", exc)
            return self._claude_vision_ocr_path(path)

    def _process_pil_image(self, img) -> OCRResult:
        """Run Tesseract on a PIL image; escalate if confidence < threshold."""
        if not TESSERACT_AVAILABLE:
            return self._claude_vision_ocr_pil(img)

        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confidences = [c for c in data["conf"] if c != -1]
            raw_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            normalized = raw_confidence / 100.0  # Tesseract returns 0-100

            text = pytesseract.image_to_string(img)
            logger.debug("Tesseract confidence: %.2f", normalized)

            if normalized < CONFIDENCE_THRESHOLD:
                logger.info(
                    "Tesseract confidence %.2f < %.2f — escalating to Claude Vision",
                    normalized,
                    CONFIDENCE_THRESHOLD,
                )
                return self._claude_vision_ocr_pil(img)

            return OCRResult(text=text, confidence=normalized, engine="tesseract")

        except Exception as exc:
            logger.warning("Tesseract error (%s); escalating to Claude Vision", exc)
            return self._claude_vision_ocr_pil(img)

    def _claude_vision_ocr_path(self, path: Path) -> OCRResult:
        """Call Claude Vision with an image file."""
        import base64
        image_data = path.read_bytes()
        b64 = base64.standard_b64encode(image_data).decode()
        suffix = path.suffix.lower().lstrip(".")
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
            suffix, "image/png"
        )
        return self._call_claude_vision(b64, media_type)

    def _claude_vision_ocr_pil(self, img) -> OCRResult:
        """Call Claude Vision with a PIL image (convert to PNG bytes first)."""
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        return self._call_claude_vision(b64, "image/png")

    def _call_claude_vision(self, b64_data: str, media_type: str) -> OCRResult:
        """Call Claude Vision API and return extracted text."""
        response = self._client.messages.create(
            model="claude-haiku-4-5-20251001",  # Haiku: cheapest vision model
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Please extract ALL text from this medical document image. "
                                "Preserve the original layout as much as possible, including "
                                "headings, lists, tables, and paragraph breaks. "
                                "Output only the extracted text, nothing else."
                            ),
                        },
                    ],
                }
            ],
        )

        extracted_text = response.content[0].text
        logger.debug("Claude Vision extracted %d characters", len(extracted_text))

        # Claude Vision doesn't give a per-character confidence; we use a fixed high value
        # because it's our premium fallback and handles complex layouts reliably.
        return OCRResult(text=extracted_text, confidence=0.92, engine="claude_vision")
