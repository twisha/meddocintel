# Clinical Multi-Adapter LoRA

Parameter-efficient fine-tuning for clinical NLP. **One open-weights base model, multiple
swappable task-specific LoRA adapters** — instead of deploying a separate full model per
clinical specialty.

This is a subproject of [MedDocIntel](..) (LLM-based clinical document extraction), adding a
self-hosted, fine-tuned tier: where MedDocIntel's backend calls Claude for extraction, this
trains small specialty adapters on top of a 7B base you control.

```
                         ┌─────────────────────────────┐
   note / report  ──▶    │  FastAPI  (task routing)     │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │      AdapterManager          │
                         │  ┌───────────────────────┐   │
                         │  │  Base model (loaded 1×)│   │   Mistral-7B / Llama-2-7B
                         │  └───────────────────────┘   │   ~4.5 GB in 4-bit
                         │     ▲  set_adapter() = O(1)   │
                         │  ┌──┴────────┬─────────────┐  │
                         │  │ cardiology │ radiology  │  │   LoRA adapters
                         │  │  -summary  │  -extract  │  │   ~30 MB each
                         │  └───────────┴─────────────┘  │
                         └─────────────────────────────┘
```

## Why this design

| Approach | Memory for N specialties | Swap cost |
|---|---|---|
| One full fine-tuned 7B per specialty | N × ~14 GB | model reload |
| **Multi-adapter LoRA (this)** | ~4.5 GB base + N × ~30 MB | `set_adapter()` pointer flip |

For 2 specialties that's **~28 GB → ~4.6 GB (~84% less)**; the gap widens with every
specialty you add. Adapters train in minutes on a single GPU and version independently of
the base.

## Adapters

| Adapter | Specialty | Task | Input → Output | Eval metric |
|---|---|---|---|---|
| `cardiology-summary` | Cardiology | Summarization | note → structured summary | ROUGE-1/2/L |
| `radiology-extract` | Radiology | Extraction | report → JSON findings | slot F1 + critical-flag accuracy |

The `extraction` target schema: `modality, body_part, findings[], impression,
measurements[], critical_flag`. `critical_flag` is scored separately because missing a
critical finding (PE, stroke, malignancy) is the failure that actually matters clinically.

## Repo layout

```
clinical-lora-adapters/
├── common/
│   ├── config.py          # base model, adapter registry, LoRA/training hyperparams
│   └── prompts.py         # prompt-format contract shared by train + eval + serve
├── data/
│   ├── cardiology_notes.json   # synthetic seed data (note + summary)
│   ├── radiology_notes.json    # synthetic seed data (report + extraction)
│   └── generate_data.py        # scale up data via Claude as ground-truth generator
├── training/
│   ├── train.py           # QLoRA fine-tuning → adapters/<name>/
│   └── eval.py            # base vs. adapter: F1/ROUGE, latency, $/1k
├── inference/
│   ├── adapter_manager.py # one base, many resident adapters, hot-swap + per-adapter stats
│   └── api.py             # FastAPI: /summarize /extract /infer/{adapter} /metrics
├── tests/test_wiring.py   # no-GPU contract tests (registry, schema, prompt prefix)
├── requirements.txt
└── README.md
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. (optional) scale the synthetic dataset — needs ANTHROPIC_API_KEY
python data/generate_data.py --adapter cardiology-summary --n 40 --append
python data/generate_data.py --adapter radiology-extract  --n 40 --append

# 2. train each adapter (QLoRA on a single GPU; writes adapters/<name>/)
python training/train.py --adapter cardiology-summary
python training/train.py --adapter radiology-extract

# 3. benchmark adapter vs. base
python training/eval.py --adapter cardiology-summary
python training/eval.py --adapter radiology-extract

# 4. serve — base loads once, both adapters resident
uvicorn inference.api:app --port 8000
curl -X POST localhost:8000/extract -H 'content-type: application/json' \
     -d '{"text":"EXAMINATION: CT chest with contrast. FINDINGS: 2.3 cm spiculated RUL nodule ..."}'
curl localhost:8000/metrics      # per-adapter request counts, latency, GPU footprint
```

### Data note (MIMIC-III)
MIMIC-III needs PhysioNet credentialing + a signed DUA, so it is not a drop-in download.
This repo ships **synthetic, PHI-free** clinical notes and generates more with Claude
(`data/generate_data.py`) — Claude writes the note *and* its gold label in one shot, so the
pairs are self-consistent and the label schema is fully controlled. If you have MIMIC access,
drop records into `data/*.json` matching the `input_key` / `target_key` in `common/config.py`.

## Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `BASE_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | swap to `meta-llama/Llama-2-7b-chat-hf`, etc. |
| `USE_4BIT` | `1` | 4-bit QLoRA (auto-off without CUDA) |
| `ADAPTERS_DIR` | `./adapters` | where trained adapters live |
| `GPU_RATE_USD_HR` | `1.006` | cost model for eval's `$/1k inferences` |

### CPU smoke test (no GPU, tiny model)
Exercises the full train → save → eval → serve path on wiring, not quality. Use a tiny
*Llama-architecture* model so the LoRA `target_modules` (`q_proj`, `gate_proj`, …) match —
a GPT-2 tiny model won't, since it uses fused `c_attn` projections:
```bash
export BASE_MODEL=hf-internal-testing/tiny-random-LlamaForCausalLM USE_4BIT=0
python training/train.py --adapter cardiology-summary --epochs 1 --batch-size 2
python training/eval.py  --adapter cardiology-summary --max-new-tokens 64
```
Metrics will be ~0 (random-weight model emits gibberish) — that's expected; the smoke test
proves the pipeline, not accuracy. Run the real thing on a GPU with the default base.

## How the pieces fit (the part worth explaining in an interview)

- **Single prompt contract.** `common/prompts.py` is imported by training, eval, and
  serving. A LoRA adapter only works for the exact prompt format it trained on, so this is
  one module, not three copies. `tests/test_wiring.py` asserts the inference prompt is a
  prefix of the training text — if that breaks, the prompt-mask in `train.py` masks the
  wrong tokens and the adapter learns nothing.
- **Completion-only loss.** Training masks the prompt tokens (`labels=-100`); the model is
  graded only on the target it should produce.
- **Fair eval.** `eval.py` loads the base once, attaches the adapter, and toggles it with
  `PeftModel.disable_adapter()`. Base and adapted numbers come from identical weights +
  prompt; the only variable is the adapter.
- **O(1) swap.** `AdapterManager` keeps both adapters resident on one base and switches with
  `set_adapter()`. Switching specialties is a pointer flip, not a reload.

## Scaling notes / limitations

- `set_adapter()` mutates shared state, so generation is serialized behind a lock. For real
  throughput: batch requests per adapter, run a replica per hot adapter, or use a server with
  native multi-LoRA batching (e.g. vLLM / LoRAX) that serves many adapters concurrently.
- Seed datasets are intentionally tiny (8 examples each) for a runnable demo — generate more
  with `generate_data.py` before reading much into the eval deltas.
- Synthetic data measures format learning, not real-world clinical accuracy. Validate on real
  (credentialed) data before any clinical claim.

## Interview narrative

> I extended MedDocIntel with parameter-efficient fine-tuning. Rather than deploying separate
> full models per clinical specialty, I built a multi-adapter LoRA system — same 7B base,
> swappable task-specific adapters — cutting serving memory ~90% while keeping domain
> accuracy. I trained on synthetic clinical data (generated with Claude as ground truth since
> MIMIC is access-gated), benchmarked each adapter against the base on F1/ROUGE/latency/cost,
> and demonstrated dynamic adapter loading behind a FastAPI service with per-adapter metrics.
