"""Celery async task pipeline.

Two tasks:
  process_document(document_id)
    → OCR → Extraction → saves results to DB → triggers verify_extraction

  verify_extraction(extraction_id)
    → Rule gates → Opus judge (if needed) → saves verdict to DB
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from celery import Celery
from sqlalchemy.orm import Session

from .db import (
    AuditLog,
    Document,
    Extraction,
    SessionLocal,
    Verification,
)
from .extraction import ExtractionAgent
from .ocr import OCRProcessor
from .verification import verify

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "meddocintel",
    broker=REDIS_URL,
    backend=REDIS_URL,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,          # re-queue on worker crash
    worker_prefetch_multiplier=1, # one task at a time per worker (fair for LLM calls)
)


def _log(db: Session, tenant_id: str, action: str, resource_type: str, resource_id, metadata: dict):
    db.add(AuditLog(
        tenant_id=tenant_id,
        actor="system",
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        event_metadata=metadata,
    ))
    db.commit()


# ---------------------------------------------------------------------------
# Task 1: OCR + Extraction
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def process_document(self, document_id: str) -> dict:
    """OCR a document and extract structured data. Saves results to DB."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        doc.status = "processing"
        db.commit()

        _log(db, str(doc.tenant_id), "PROCESSING_STARTED", "document", doc.id, {})

        # --- OCR ---
        ocr = OCRProcessor()
        if doc.ocr_text:
            ocr_result = ocr.process_text(doc.ocr_text)
        else:
            ocr_result = ocr.process(doc.file_path)

        doc.ocr_text = ocr_result.text
        doc.ocr_engine = ocr_result.engine
        doc.ocr_confidence = ocr_result.confidence
        db.commit()

        logger.info("OCR complete — doc=%s engine=%s confidence=%.2f", document_id, ocr_result.engine, ocr_result.confidence)

        # --- Extraction ---
        agent = ExtractionAgent()
        extraction_obj, token_usage = agent.extract_clinical_progress_note(
            ocr_text=ocr_result.text,
            doc_id=document_id,
            tenant_id=str(doc.tenant_id),
            ocr_engine=ocr_result.engine,
            ocr_confidence=ocr_result.confidence,
        )

        extraction_dict = extraction_obj.model_dump()

        # Persist extraction
        existing_versions = db.query(Extraction).filter(Extraction.document_id == document_id).count()
        extraction_row = Extraction(
            document_id=document_id,
            version=existing_versions + 1,
            extraction_json=extraction_dict,
            overall_confidence=extraction_obj.overall_confidence,
            confidence_tier=extraction_obj.confidence_tier.value,
            model=token_usage.model,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
        )
        db.add(extraction_row)
        doc.status = "extracted"
        doc.processed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(extraction_row)

        logger.info(
            "Extraction saved — doc=%s extraction=%s confidence=%.2f tier=%s cost=$%.4f",
            document_id, str(extraction_row.id),
            extraction_obj.overall_confidence,
            extraction_obj.confidence_tier.value,
            token_usage.estimated_cost_usd,
        )

        _log(db, str(doc.tenant_id), "EXTRACTION_COMPLETE", "extraction", extraction_row.id, {
            "confidence": extraction_obj.overall_confidence,
            "tier": extraction_obj.confidence_tier.value,
            "tokens_in": token_usage.input_tokens,
            "tokens_out": token_usage.output_tokens,
        })

        # Trigger verification asynchronously
        verify_extraction.delay(str(extraction_row.id))

        return {"status": "extracted", "extraction_id": str(extraction_row.id)}

    except Exception as exc:
        logger.exception("process_document failed for %s", document_id)
        db.query(Document).filter(Document.id == document_id).update({"status": "failed"})
        db.commit()
        raise self.retry(exc=exc)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 2: Verification
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def verify_extraction(self, extraction_id: str) -> dict:
    """Run rule gates + Opus judge on an extraction. Saves verdict to DB."""
    db = SessionLocal()
    try:
        row = db.query(Extraction).filter(Extraction.id == extraction_id).first()
        if not row:
            raise ValueError(f"Extraction not found: {extraction_id}")

        doc = db.query(Document).filter(Document.id == row.document_id).first()
        ocr_text = doc.ocr_text or ""

        result = verify(
            ocr_text=ocr_text,
            extraction=row.extraction_json,
            extraction_confidence=row.overall_confidence,
        )

        verification_row = Verification(
            extraction_id=extraction_id,
            verdict=result.verdict,
            overall_score=result.overall_score,
            field_scores=result.field_scores,
            rule_flags=result.rule_flags,
            verified_by_model=result.model if result.ran_opus else "rule_gates_only",
        )
        db.add(verification_row)

        # Update document status based on verdict
        new_status = {
            "ACCEPT": "verified",
            "FLAG": "flagged",
            "REJECT": "rejected",
        }.get(result.verdict, "flagged")
        doc.status = new_status
        db.commit()

        _log(db, str(doc.tenant_id), "VERIFICATION_COMPLETE", "verification", verification_row.id, {
            "verdict": result.verdict,
            "overall_score": result.overall_score,
            "ran_opus": result.ran_opus,
            "rule_flags": result.rule_flags,
        })

        logger.info("Verification complete — extraction=%s verdict=%s score=%.2f", extraction_id, result.verdict, result.overall_score)
        return {"verdict": result.verdict, "score": result.overall_score}

    except Exception as exc:
        logger.exception("verify_extraction failed for %s", extraction_id)
        raise self.retry(exc=exc)
    finally:
        db.close()
