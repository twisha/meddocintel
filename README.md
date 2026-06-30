# MedDocIntel

**Clinical Document Intelligence as a Service** — extracts structured data from medical documents (PDFs, scans, images) using a multi-stage AI pipeline with verification, multi-tenant isolation, and a full audit trail.

```
Document (PDF/image/text)
    ↓
[OCR]  Tesseract → Claude Vision fallback
    ↓
[Extraction]  Claude Sonnet → structured JSON + confidence + source spans
    ↓
[Verification]  Rule gates + Claude Opus judge → ACCEPT / FLAG / REJECT
    ↓
[Storage]  PostgreSQL with row-level security (per-tenant isolation)
    ↓
[UI]  Next.js dashboard — upload, review queue, extraction viewer
```

> 📐 **Architecture:** see [docs/HLD.md](docs/HLD.md) for the high-level design — system context, pipeline, data model, and sequence diagrams (Mermaid).

> 🧩 **Subproject — self-hosted fine-tuned tier:** [`clinical-lora-adapters/`](clinical-lora-adapters/README.md) adds a parameter-efficient (LoRA) system — one open-weights 7B base with swappable, specialty-specific adapters (cardiology summarization, radiology extraction) — as a self-hosted alternative to the Claude extraction slot. Same memory footprint serves N specialties. See its [README](clinical-lora-adapters/README.md).

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Python 3.11 |
| Async queue | Celery + Redis |
| Database | PostgreSQL 16 + SQLAlchemy + Alembic |
| Auth | PyJWT + bcrypt (rolled, zero-cost) |
| OCR | Tesseract (local) → Claude Haiku Vision (fallback) |
| Extraction | Claude Sonnet (`claude-sonnet-4-6`) |
| Verification | Claude Opus (`claude-opus-4-8`) — sampled |
| Fine-tuned tier | Mistral-7B + LoRA adapters (HuggingFace `peft`) — see [`clinical-lora-adapters/`](clinical-lora-adapters/README.md) |
| UI | Next.js 15 + Tailwind CSS |
| Reverse proxy | Nginx |
| Containers | Docker Compose |

---

## Local Setup

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### 1. Clone and configure

```bash
git clone <repo-url>
cd meddocintel

cp .env.example .env
```

Open `.env` and fill in:

```env
ANTHROPIC_API_KEY=sk-ant-...          # required
POSTGRES_PASSWORD=pick_a_password     # any string
JWT_SECRET=<random string>            # generate: python3 -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Start everything

```bash
docker compose up --build
```

First build takes ~3 minutes (downloads images, installs deps). Subsequent starts take ~15 seconds.

### 3. Verify it's running

```
http://localhost       → UI (Next.js)
http://localhost/api/health → {"status": "ok"}
http://localhost:8000/docs  → FastAPI Swagger UI
```

### 4. Create a tenant + user

```bash
# Create a tenant (save the api_key — shown only once)
curl -s -X POST http://localhost:8000/auth/signup/tenant \
  -H "Content-Type: application/json" \
  -d '{"name": "Demo Clinic"}' | python3 -m json.tool

# Create a user (use the api_key from above as the Bearer token)
curl -s -X POST http://localhost:8000/auth/signup/user \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{"email": "doctor@demo.com", "password": "secret123"}' | python3 -m json.tool
```

### 5. Upload a document

```bash
# Login to get a JWT
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "doctor@demo.com", "password": "secret123"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Upload a sample clinical note
curl -s -X POST http://localhost:8000/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@backend/fixtures/clinical_notes/fixture_001.txt" | python3 -m json.tool

# List documents (poll until status = verified/flagged)
curl -s http://localhost:8000/documents \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

### 6. Open the UI

Go to `http://localhost` and sign in with `doctor@demo.com / secret123`.

---

## Project Structure

```
meddocintel/
├── docker-compose.yml
├── .env                    ← secrets (git-ignored)
├── .env.example            ← template
│
├── backend/
│   ├── src/
│   │   ├── main.py         ← FastAPI routes
│   │   ├── schemas.py      ← Pydantic models (the spec)
│   │   ├── ocr.py          ← Tesseract + Claude Vision
│   │   ├── extraction.py   ← Claude Sonnet extraction agent
│   │   ├── verification.py ← Opus judge + rule gates
│   │   ├── tasks.py        ← Celery async pipeline
│   │   ├── db.py           ← SQLAlchemy models + RLS
│   │   └── auth.py         ← JWT + API key auth
│   ├── alembic/            ← DB migrations
│   ├── tests/              ← pytest suite
│   ├── fixtures/           ← de-identified sample notes
│   └── Dockerfile
│
├── web/
│   ├── src/app/
│   │   ├── page.tsx        ← Dashboard
│   │   ├── upload/         ← Upload page
│   │   ├── review/         ← Review queue
│   │   └── login/          ← Login page
│   ├── src/lib/api.ts      ← typed API client
│   └── Dockerfile
│
├── infra/
│   └── nginx.conf          ← reverse proxy
│
└── clinical-lora-adapters/ ← self-hosted PEFT subproject (own README)
    ├── common/             ← base model + adapter registry + prompt contract
    ├── data/               ← synthetic clinical notes + Claude-based generator
    ├── training/           ← train.py (QLoRA) + eval.py (base vs adapter)
    └── inference/          ← adapter_manager.py + api.py (FastAPI, hot-swap)
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | — | Liveness probe |
| POST | `/auth/signup/tenant` | — | Create tenant, get API key |
| POST | `/auth/signup/user` | API key | Add user to tenant |
| POST | `/auth/login` | — | Get JWT |
| POST | `/documents` | JWT/API key | Upload document (async) |
| GET | `/documents` | JWT/API key | List documents |
| GET | `/documents/{id}` | JWT/API key | Document detail |
| GET | `/documents/{id}/extraction` | JWT/API key | Latest extraction + verification |
| GET | `/review-queue` | JWT/API key | Flagged/rejected documents |

Full interactive docs: `http://localhost:8000/docs`

---

## Architecture Decisions

### Why Tesseract first, Claude Vision fallback?
Tesseract is free and local (no PHI leaves the machine). For clean text documents it achieves 85–95% confidence. Claude Vision costs ~$0.01/page and handles complex layouts. The threshold is 0.6 — below that we escalate.

### Why source spans on every field?
Healthcare regulations (HIPAA, 21 CFR Part 11) require you to defend extraction decisions years later. Every extracted field includes `source_span: [start, end]` — character indices in the original OCR text. A reviewer can click a field and see exactly what the model saw.

### Why human-authored routing (not LLM-as-orchestrator)?
Rule: `classifier → fixed spec → LLM in extraction slot`. With LLM orchestration, two LLMs can fail together. With rule-based routing, you can model reliability mathematically: 99% classifier × 90% extraction = 89.1%. You know where errors come from.

### Why sampled verification?
Opus is 10× more expensive than Sonnet. Extractions with confidence > 0.85 and no rule gate violations are accepted without Opus. Only low-confidence or flagged extractions run the full Opus judge — reducing verification cost by ~80%.

### Why rolled auth (PyJWT + bcrypt)?
Zero cost, zero external dependency. Two patterns: API keys (machine-to-machine, hashed with bcrypt, prefixed `sk-`) and JWTs (human users, 24-hour expiry). Every request resolves to `(tenant_id, actor)`.

### Why PostgreSQL row-level security?
Tenant isolation at the database layer. `SET LOCAL app.current_tenant = '<id>'` is called on every request; RLS policies automatically filter all queries to that tenant's rows.

### Why a self-hosted LoRA tier alongside Claude?
The Claude extraction slot is the default — best quality, zero ops. But some deployments need PHI to stay on-prem or want lower per-document cost at volume. [`clinical-lora-adapters/`](clinical-lora-adapters/README.md) fine-tunes small task-specific adapters on top of one open-weights 7B base, so N clinical specialties cost ~one base model + N tiny adapters (~84–90% less serving memory than N full models). It slots in behind the same extraction interface (`extraction.py`), and `eval.py` benchmarks adapter-vs-base on F1/latency/cost so the quality trade-off is measured, not assumed.

---

## Running Tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

Tests use mocked Anthropic clients — no API calls, no cost.

---

## Stopping

```bash
docker compose down          # stop containers, keep data
docker compose down -v       # stop + delete all data (fresh start)
```
