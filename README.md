# KrafNova AI layer (scaffold)

FastAPI service with a LangGraph agent for the AI part of the project. Everything that needs a real model is a stub marked `TODO`, so the app runs end to end today.

## Run

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Try it: `POST /design` with `{"message": "a cedar tree for a 10cm wood sign"}`.

## Structure

```
app/main.py                  FastAPI endpoints (/health, /design)
app/config.py                env-based settings, no secrets in the repo
app/schemas/design_intent.py Pydantic schemas: the LLM -> deterministic boundary
app/agent/state.py           graph state
app/agent/graph.py           LangGraph wiring
app/agent/nodes/intent.py    message -> validated DesignIntent (LLM stub)
app/agent/nodes/image_gen.py image candidates (stub)
app/agent/nodes/vectorize.py deterministic raster -> SVG (stub)
app/agent/nodes/explain.py   plain-language explainer (stub)
```

## Flow

`intent -> images -> vectorize -> explain`. If the intent fails validation, the graph stops and the API returns 422. Only a validated `DesignIntent` reaches the deterministic steps.

## Mapping to the plan (solo AI track)

- LangGraph assistant: `app/agent/graph.py`
- Structured design intent: `app/schemas/design_intent.py`, `nodes/intent.py`
- Image/vector candidates: `nodes/image_gen.py`
- Vectorization: `nodes/vectorize.py`
- Explainer: `nodes/explain.py`

## Rules

- This repo is public until it is made private. Never commit keys; use `.env`.
- No code copied from upstream projects (for example FreazyKam2) is included. Check that project's license before reusing any of it.
