"""Node 4: explain what was made and why, in plain language.

Uses Gemini when a key is configured and EXPLAIN_WITH_LLM is on, so the
explanation reflects the actual intent and candidates (and the thesis can
evaluate explanation quality). Falls back to a deterministic template when
no LLM is available, so the graph still completes in development.
"""
import logging

from google import genai

from app.config import settings

log = logging.getLogger(__name__)

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _template(intent, candidates) -> str:
    return (
        f"Made {len(candidates)} {intent.style} options for '{intent.subject}' at "
        f"{intent.dimensions.width_mm}x{intent.dimensions.height_mm} mm."
    )


def _llm_explain(intent, candidates) -> str:
    prompt = (
        "You are the explainer for a CNC / vector-art design service. "
        "In 2-4 plain sentences, tell the user what was designed for them and "
        "why these choices fit their request. No markdown, no bullet points.\n\n"
        f"Intent: {intent.model_dump_json()}\n"
        f"Number of candidates produced: {len(candidates)}"
    )
    response = _get_client().models.generate_content(
        model=settings.gemini_model, contents=prompt
    )
    return response.text.strip()


def explain(state: dict) -> dict:
    intent = state["intent"]
    candidates = state["candidates"]
    if settings.explain_with_llm and settings.gemini_api_key:
        try:
            return {"explanation": _llm_explain(intent, candidates)}
        except Exception:
            log.exception("explainer LLM call failed, falling back to template")
    return {"explanation": _template(intent, candidates)}
