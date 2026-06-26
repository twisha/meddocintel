"""Authentication: tenant API keys + user JWTs.

Flow:
  1. Tenant signs up → receives a generated API key (sk-...)
  2. API key is hashed with bcrypt and stored; only the prefix is kept for display
  3. Machine-to-machine calls: Authorization: Bearer sk-...
  4. Human users: POST /auth/login → JWT → Authorization: Bearer <jwt>
  5. Every request resolves to a (tenant_id, actor) pair injected into the route
"""

from __future__ import annotations

import os
import secrets
import string
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .db import Tenant, User, get_db

JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# API key utilities
# ---------------------------------------------------------------------------

API_KEY_LENGTH = 40
API_KEY_PREFIX = "sk-"
_CHARS = string.ascii_letters + string.digits


def generate_api_key() -> str:
    body = "".join(secrets.choice(_CHARS) for _ in range(API_KEY_LENGTH))
    return f"{API_KEY_PREFIX}{body}"


def hash_secret(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_secret(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# JWT utilities
# ---------------------------------------------------------------------------

def create_jwt(tenant_id: str, user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------

class AuthContext:
    def __init__(self, tenant_id: str, actor: str, is_api_key: bool = False):
        self.tenant_id = tenant_id
        self.actor = actor
        self.is_api_key = is_api_key


def _resolve_api_key(token: str, db: Session) -> AuthContext | None:
    """Try to authenticate as a tenant API key (sk-...)."""
    if not token.startswith(API_KEY_PREFIX):
        return None
    tenant = db.query(Tenant).filter(Tenant.is_active == True).all()
    for t in tenant:
        if verify_secret(token, t.api_key_hash):
            return AuthContext(tenant_id=str(t.id), actor=f"api_key:{t.api_key_prefix}...", is_api_key=True)
    return None


def _resolve_jwt(token: str, db: Session) -> AuthContext | None:
    """Try to authenticate as a human user JWT."""
    try:
        payload = decode_jwt(token)
    except HTTPException:
        return None
    user = db.query(User).filter(User.id == payload["sub"], User.is_active == True).first()
    if not user:
        return None
    return AuthContext(tenant_id=payload["tenant_id"], actor=payload["email"])


def get_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    db: Session = Depends(get_db),
) -> AuthContext:
    """FastAPI dependency — resolves Bearer token to AuthContext or raises 401."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")

    token = credentials.credentials
    ctx = _resolve_api_key(token, db) or _resolve_jwt(token, db)
    if not ctx:
        raise HTTPException(status_code=401, detail="Invalid or expired credentials")
    return ctx


# ---------------------------------------------------------------------------
# Tenant + user management helpers
# ---------------------------------------------------------------------------

def create_tenant(name: str, db: Session) -> tuple[Tenant, str]:
    """Create a tenant and return (tenant, raw_api_key)."""
    raw_key = generate_api_key()
    tenant = Tenant(
        name=name,
        api_key_hash=hash_secret(raw_key),
        api_key_prefix=raw_key[:8],
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant, raw_key


def create_user(tenant_id: str, email: str, password: str, db: Session) -> User:
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        tenant_id=tenant_id,
        email=email,
        password_hash=hash_secret(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def login_user(email: str, password: str, db: Session) -> str:
    """Verify credentials and return a signed JWT."""
    user = db.query(User).filter(User.email == email, User.is_active == True).first()
    if not user or not verify_secret(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return create_jwt(str(user.tenant_id), str(user.id), user.email)
