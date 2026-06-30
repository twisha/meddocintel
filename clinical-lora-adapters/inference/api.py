"""FastAPI inference service — task-routed, dynamic-adapter clinical inference.

Routing model mirrors MedDocIntel: human-authored routing, not LLM-as-orchestrator.
The client names the task (or the adapter) and we select the right adapter on the shared
base. Every response carries the adapter used, latency, and token counts; per-adapter
aggregates are exposed at /metrics for monitoring.

Run:
    uvicorn inference.api:app --host 0.0.0.0 --port 8000
    # or: python inference/api.py
Then:
    curl localhost:8000/adapters
    curl -X POST localhost:8000/summarize -H 'content-type: application/json' \
         -d '{"text": "CARDIOLOGY CONSULT ..."}'
    curl -X POST localhost:8000/extract -H 'content-type: application/json' \
         -d '{"text": "EXAMINATION: CT chest ..."}'
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import BASE_MODEL, TASK_EXTRACTION, TASK_SUMMARIZATION  # noqa: E402
from inference.adapter_manager import AdapterManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("api")

# Map a task to its adapter. Adding a specialty = registering an adapter + one row here.
TASK_TO_ADAPTER = {
    TASK_SUMMARIZATION: "cardiology-summary",
    TASK_EXTRACTION: "radiology-extract",
}

manager: AdapterManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    logger.info("Initializing AdapterManager (base=%s)", BASE_MODEL)
    manager = AdapterManager()  # base loads once at startup; adapters load lazily
    # Warm-load whatever is already trained so the first request isn't slow.
    for adapter in set(TASK_TO_ADAPTER.values()):
        try:
            manager.load_adapter(adapter)
        except FileNotFoundError as e:
            logger.warning("Skipping warm-load: %s", e)
    yield
    logger.info("Shutting down")


app = FastAPI(title="Clinical Multi-Adapter LoRA API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class InferRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw clinical note or report text")
    adapter: str | None = Field(None, description="Override adapter selection (else routed by task)")
    max_new_tokens: int = Field(320, ge=16, le=1024)


class InferResponse(BaseModel):
    adapter: str
    task_type: str
    output: str
    parsed: dict | None = None
    latency_ms: float
    input_tokens: int
    output_tokens: int


def _require_manager() -> AdapterManager:
    if manager is None:
        raise HTTPException(503, "Model not initialized yet")
    return manager


def _run(task_type: str, req: InferRequest) -> InferResponse:
    mgr = _require_manager()
    adapter = req.adapter or TASK_TO_ADAPTER[task_type]
    try:
        result = mgr.generate(adapter, req.text, max_new_tokens=req.max_new_tokens)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))
    return InferResponse(
        adapter=result.adapter, task_type=task_type, output=result.output_text,
        parsed=result.parsed, latency_ms=result.latency_ms,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "initialized": manager is not None, "base_model": BASE_MODEL}


@app.get("/adapters")
def adapters() -> dict:
    return {"adapters": _require_manager().registry()}


@app.get("/metrics")
def metrics() -> dict:
    """Per-adapter request counts, mean latency, token totals + memory footprint."""
    mgr = _require_manager()
    return {
        "memory": mgr.memory_summary(),
        "per_adapter": {a["name"]: a["stats"] for a in mgr.registry()},
    }


@app.post("/summarize", response_model=InferResponse)
def summarize(req: InferRequest) -> InferResponse:
    return _run(TASK_SUMMARIZATION, req)


@app.post("/extract", response_model=InferResponse)
def extract(req: InferRequest) -> InferResponse:
    return _run(TASK_EXTRACTION, req)


@app.post("/infer/{adapter_name}", response_model=InferResponse)
def infer(adapter_name: str, req: InferRequest) -> InferResponse:
    """Explicit-adapter route for clients that want to pick the adapter directly."""
    from common.config import get_adapter

    try:
        spec = get_adapter(adapter_name)
    except KeyError as e:
        raise HTTPException(404, str(e))
    req.adapter = adapter_name
    return _run(spec.task_type, req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference.api:app", host="0.0.0.0", port=8000, reload=False)
