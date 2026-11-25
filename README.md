# KrafNova AI layer

FastAPI service with a LangGraph agent for the AI part of the project. The
full workflow runs end to end: real LLM intent extraction, image candidates,
deterministic vectorization, and a plain-language explainer.

## Run

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
uvicorn app.main:app --reload
```

Try it: `POST /design` with `{"message": "a cedar tree for a 10cm wood sign"}`.

Without a key the app still runs: set `IMAGE_PROVIDER=stub` (default) and the
explainer falls back to a template, but the intent node needs a real
`GEMINI_API_KEY`.

## The workflow

```
message -> intent -> images -> vectorize -> explain
```

1. **intent** (`nodes/intent.py`): Gemini (`GEMINI_MODEL`, structured output
   via `response_schema`) returns JSON that must validate against the
   `DesignIntent` Pydantic schema. On validation failure the model gets one
   retry with the error fed back; after that the API returns 422. This schema
   is the boundary: only a validated intent reaches the deterministic steps.
2. **images** (`nodes/image_gen.py`): candidate rasters. `IMAGE_PROVIDER=gemini`
   calls a real image model and saves PNGs under `generated/` (served at
   `/generated`); `stub` returns placeholders so development works offline.
3. **vectorize** (`nodes/vectorize.py`): deterministic, no AI. A rule-based
   threshold + rect-merge tracer turns each raster into an SVG. Same input,
   same SVG, every time. Swap `trace_to_svg` for potrace/vtracer later
   without touching the graph.
4. **explain** (`nodes/explain.py`): Gemini writes 2-4 plain sentences about
   what was made and why (thesis: explanation-quality evaluation). Falls back
   to a deterministic template without a key.

## Structure

```
app/main.py                  FastAPI endpoints (/health, /design, /generated)
app/config.py                env-based settings, no secrets in the repo
app/schemas/design_intent.py Pydantic schemas: the LLM -> deterministic boundary
app/agent/state.py           graph state
app/agent/graph.py           LangGraph wiring
app/agent/nodes/intent.py    message -> validated DesignIntent (Gemini)
app/agent/nodes/image_gen.py image candidates (stub | Gemini image model)
app/agent/nodes/vectorize.py deterministic raster -> SVG (rule-based tracer)
app/agent/nodes/explain.py   plain-language explainer (Gemini, template fallback)
```

## Mapping to the plan (solo AI track)

- LangGraph assistant: `app/agent/graph.py`
- Structured design intent: `app/schemas/design_intent.py`, `nodes/intent.py`
- Image/vector candidates: `nodes/image_gen.py`
- Vectorization (deterministic): `nodes/vectorize.py`
- Explainer: `nodes/explain.py`

## Testing

`python tests/test_workflow.py` runs the graph with the model calls mocked:
intent validation + retry + 422 path, image stub, real tracer determinism,
and the explainer fallback. Calls that need a live key (Gemini intent, image
generation, LLM explainer) are untested until a real `GEMINI_API_KEY` is set.

## Rules

- This repo is public until it is made private. Never commit keys; use `.env`.
- No code copied from upstream projects (for example FreazyKam2) is included.
  Check that project's license before reusing any of it.
