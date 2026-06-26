"""FastAPI application — production pipeline with auth, file upload, and async processing."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .auth import AuthContext, create_tenant, create_user, get_auth, login_user
from .db import Document, Extraction, SessionLocal, Verification, get_db, init_db, set_tenant_context
from .tasks import process_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/meddocintel/uploads"))


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    logger.info("MedDocIntel API ready")
    yield


app = FastAPI(
    title="MedDocIntel",
    description="Clinical document intelligence — production API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(int((time.perf_counter() - start) * 1000))
    return response


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TenantSignupRequest(BaseModel):
    name: str

class TenantSignupResponse(BaseModel):
    tenant_id: str
    name: str
    api_key: str
    api_key_prefix: str

class UserSignupRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class DocumentResponse(BaseModel):
    document_id: str
    status: str
    message: str

class ExtractionSummary(BaseModel):
    id: str
    document_id: str
    version: int
    overall_confidence: float
    confidence_tier: str
    verdict: str | None
    extracted_at: str

class DocumentDetail(BaseModel):
    id: str
    original_filename: str
    status: str
    document_type: str
    ocr_engine: str | None
    ocr_confidence: float | None
    uploaded_at: str
    extractions: list[ExtractionSummary]


# ---------------------------------------------------------------------------
# Public routes (no auth)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/auth/signup/tenant", response_model=TenantSignupResponse, status_code=201)
def signup_tenant(req: TenantSignupRequest, db: Session = Depends(get_db)):
    """Create a new tenant and return its API key (shown once — save it)."""
    try:
        tenant, raw_key = create_tenant(req.name, db)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return TenantSignupResponse(
        tenant_id=str(tenant.id),
        name=tenant.name,
        api_key=raw_key,
        api_key_prefix=tenant.api_key_prefix,
    )


@app.post("/auth/signup/user", status_code=201)
def signup_user(
    req: UserSignupRequest,
    auth: AuthContext = Depends(get_auth),
    db: Session = Depends(get_db),
):
    """Create a human user under the authenticated tenant."""
    user = create_user(auth.tenant_id, req.email, req.password, db)
    return {"user_id": str(user.id), "email": user.email}


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    token = login_user(req.email, req.password, db)
    return LoginResponse(access_token=token)


# ---------------------------------------------------------------------------
# Document routes (auth required)
# ---------------------------------------------------------------------------

@app.post("/documents", response_model=DocumentResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    auth: AuthContext = Depends(get_auth),
    db: Session = Depends(get_db),
):
    """Upload a document (PDF, image, or .txt). Processing is async."""
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".txt"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

    tenant_dir = UPLOAD_DIR / auth.tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid.uuid4())
    dest = tenant_dir / f"{doc_id}{suffix}"
    content = await file.read()
    dest.write_bytes(content)

    set_tenant_context(db, auth.tenant_id)
    doc = Document(
        id=doc_id,
        tenant_id=auth.tenant_id,
        original_filename=file.filename or "unknown",
        file_path=str(dest),
        status="pending",
    )
    db.add(doc)
    db.commit()

    process_document.delay(doc_id)
    logger.info("Document queued — tenant=%s doc=%s file=%s", auth.tenant_id, doc_id, file.filename)

    return DocumentResponse(document_id=doc_id, status="pending", message="Document queued for processing")


@app.get("/documents", response_model=list[DocumentDetail])
def list_documents(
    status_filter: str | None = None,
    limit: int = 50,
    auth: AuthContext = Depends(get_auth),
    db: Session = Depends(get_db),
):
    set_tenant_context(db, auth.tenant_id)
    query = db.query(Document).filter(Document.tenant_id == auth.tenant_id)
    if status_filter:
        query = query.filter(Document.status == status_filter)
    docs = query.order_by(Document.uploaded_at.desc()).limit(limit).all()
    return [_doc_to_detail(doc, db) for doc in docs]


@app.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: str,
    auth: AuthContext = Depends(get_auth),
    db: Session = Depends(get_db),
):
    set_tenant_context(db, auth.tenant_id)
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.tenant_id == auth.tenant_id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_to_detail(doc, db)


@app.get("/documents/{document_id}/extraction")
def get_extraction(
    document_id: str,
    auth: AuthContext = Depends(get_auth),
    db: Session = Depends(get_db),
):
    """Return the latest extraction + verification for a document."""
    set_tenant_context(db, auth.tenant_id)
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.tenant_id == auth.tenant_id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    extraction = (
        db.query(Extraction)
        .filter(Extraction.document_id == document_id)
        .order_by(Extraction.version.desc())
        .first()
    )
    if not extraction:
        raise HTTPException(status_code=404, detail="No extraction available yet")

    verification = (
        db.query(Verification)
        .filter(Verification.extraction_id == extraction.id)
        .order_by(Verification.created_at.desc())
        .first()
    )

    return {
        "extraction_id": str(extraction.id),
        "version": extraction.version,
        "overall_confidence": extraction.overall_confidence,
        "confidence_tier": extraction.confidence_tier,
        "data": extraction.extraction_json,
        "verification": {
            "verdict": verification.verdict,
            "overall_score": verification.overall_score,
            "field_scores": verification.field_scores,
            "rule_flags": verification.rule_flags,
        } if verification else None,
    }


@app.get("/review-queue")
def review_queue(
    auth: AuthContext = Depends(get_auth),
    db: Session = Depends(get_db),
):
    """Documents in flagged or rejected status — for the manual review UI."""
    set_tenant_context(db, auth.tenant_id)
    docs = (
        db.query(Document)
        .filter(
            Document.tenant_id == auth.tenant_id,
            Document.status.in_(["flagged", "rejected"]),
        )
        .order_by(Document.uploaded_at.desc())
        .all()
    )
    return [_doc_to_detail(doc, db) for doc in docs]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _doc_to_detail(doc: Document, db: Session) -> DocumentDetail:
    extractions = (
        db.query(Extraction)
        .filter(Extraction.document_id == str(doc.id))
        .order_by(Extraction.version.desc())
        .all()
    )
    summaries = []
    for ext in extractions:
        v = (
            db.query(Verification)
            .filter(Verification.extraction_id == str(ext.id))
            .order_by(Verification.created_at.desc())
            .first()
        )
        summaries.append(ExtractionSummary(
            id=str(ext.id),
            document_id=str(doc.id),
            version=ext.version,
            overall_confidence=ext.overall_confidence,
            confidence_tier=ext.confidence_tier,
            verdict=v.verdict if v else None,
            extracted_at=ext.extracted_at.isoformat(),
        ))

    return DocumentDetail(
        id=str(doc.id),
        original_filename=doc.original_filename,
        status=doc.status,
        document_type=doc.document_type,
        ocr_engine=doc.ocr_engine,
        ocr_confidence=doc.ocr_confidence,
        uploaded_at=doc.uploaded_at.isoformat(),
        extractions=summaries,
    )
