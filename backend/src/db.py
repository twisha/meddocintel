"""SQLAlchemy models and database session management.

Tables:
  tenants       - organisations (one API key each)
  users         - human users belonging to a tenant
  documents     - uploaded files (stored on disk, metadata here)
  extractions   - versioned extraction results (JSONB)
  verifications - Opus judge verdicts (immutable audit trail)
  audit_logs    - every state-changing event, append-only
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://meddoc:meddoc@localhost:5432/meddocintel"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    api_key_hash = Column(String(255), nullable=False)   # bcrypt hash
    api_key_prefix = Column(String(8), nullable=False)   # first 8 chars for display
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="tenant", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    tenant = relationship("Tenant", back_populates="users")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    original_filename = Column(String(512), nullable=False)
    file_path = Column(Text, nullable=False)         # path on server disk
    document_type = Column(String(64), nullable=False, default="clinical_progress_note")
    status = Column(String(32), nullable=False, default="pending")
    # pending | processing | extracted | verified | failed
    ocr_engine = Column(String(32))
    ocr_confidence = Column(Float)
    ocr_text = Column(Text)                          # raw OCR output
    uploaded_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    processed_at = Column(DateTime(timezone=True))

    tenant = relationship("Tenant", back_populates="documents")
    extractions = relationship("Extraction", back_populates="document", cascade="all, delete-orphan")


class Extraction(Base):
    __tablename__ = "extractions"
    __table_args__ = (
        UniqueConstraint("document_id", "version", name="uq_extraction_doc_version"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    extraction_json = Column(JSON, nullable=False)   # full ClinicalProgressNoteExtraction dict
    overall_confidence = Column(Float, nullable=False)
    confidence_tier = Column(String(16), nullable=False)  # high | medium | low
    model = Column(String(64), nullable=False)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    extracted_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    document = relationship("Document", back_populates="extractions")
    verifications = relationship("Verification", back_populates="extraction", cascade="all, delete-orphan")


class Verification(Base):
    """Immutable record of an Opus verification verdict. Never updated, only inserted."""

    __tablename__ = "verifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    extraction_id = Column(UUID(as_uuid=True), ForeignKey("extractions.id", ondelete="CASCADE"), nullable=False)
    verdict = Column(String(16), nullable=False)     # ACCEPT | FLAG | REJECT
    overall_score = Column(Float, nullable=False)
    field_scores = Column(JSON, nullable=False)      # {field_name: score}
    rule_flags = Column(JSON, default=list)          # list of rule violations
    verified_by_model = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    extraction = relationship("Extraction", back_populates="verifications")


class AuditLog(Base):
    """Append-only audit trail. Rows are never updated or deleted."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    actor = Column(String(255))                      # user email or "system"
    action = Column(String(64), nullable=False)      # e.g. DOCUMENT_UPLOADED, EXTRACTION_COMPLETE
    resource_type = Column(String(64))               # document | extraction | verification
    resource_id = Column(UUID(as_uuid=True))
    event_metadata = Column(JSON, default=dict)      # action-specific payload (no PHI)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


# ---------------------------------------------------------------------------
# Row-level security helpers
# ---------------------------------------------------------------------------

def set_tenant_context(db: Session, tenant_id: str) -> None:
    """Set PostgreSQL session variable used by RLS policies."""
    db.execute(text("SET LOCAL app.current_tenant = :tid"), {"tid": str(tenant_id)})


def create_rls_policies(conn) -> None:
    """Apply RLS policies — called once after table creation."""
    statements = [
        "ALTER TABLE documents ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE extractions ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE verifications ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY",

        """CREATE POLICY tenant_documents ON documents
           USING (tenant_id = current_setting('app.current_tenant', true)::uuid)""",

        """CREATE POLICY tenant_extractions ON extractions
           USING (document_id IN (
               SELECT id FROM documents
               WHERE tenant_id = current_setting('app.current_tenant', true)::uuid
           ))""",

        """CREATE POLICY tenant_verifications ON verifications
           USING (extraction_id IN (
               SELECT e.id FROM extractions e
               JOIN documents d ON d.id = e.document_id
               WHERE d.tenant_id = current_setting('app.current_tenant', true)::uuid
           ))""",

        """CREATE POLICY tenant_audit_logs ON audit_logs
           USING (tenant_id = current_setting('app.current_tenant', true)::uuid)""",
    ]
    for stmt in statements:
        try:
            conn.execute(text(stmt))
        except Exception:
            pass  # policy already exists


def init_db() -> None:
    """Create all tables. Run once at startup or via Alembic."""
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        create_rls_policies(conn)
        conn.commit()
