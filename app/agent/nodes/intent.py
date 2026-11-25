"""Node 1: turn the user's message into a validated DesignIntent.

Real Gemini call with structured output (response_schema=DesignIntent).
Validation is the boundary between the LLM and the deterministic pipeline:
if the model's JSON fails DesignIntent validation we retry once with the
error fed back, then give up and the API layer returns 422.
"""
import logging

from google import genai
from google.genai import types
from pydantic import ValidationError

from app.config import settings
from app.schemas.design_intent import DesignIntent

log = logging.getLogger(__name__)

_SYSTEM = (
    "You extract a design intent for a CNC / vector-art design service. "
    "Read the user's request and return the structured fields only. "
    "Dimensions are physical millimeters of the final piece; infer a "
    "reasonable size when the user does not state one."
)

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _call_llm(message: str, feedback: str | None = None) -> str:
    """One Gemini call; returns the raw JSON text of the structured response."""
    contents = message
    if feedback:
        contents = (
            f"{message}\n\n"
            f"Your previous answer failed validation with this error:\n{feedback}\n"
            "Return a corrected JSON object only."
        )
    response = _get_client().models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            response_mime_type="application/json",
            response_schema=DesignIntent,
        ),
    )
    return response.text


def parse_intent(state: dict) -> dict:
    message = state["message"]
    feedback = None
    for attempt in (1, 2):
        try:
            raw = _call_llm(message, feedback)
            return {"intent": DesignIntent.model_validate_json(raw), "error": None}
        except ValidationError as exc:
            log.warning("intent validation failed (attempt %s): %s", attempt, exc)
            feedback = str(exc)
    return {"intent": None, "error": f"invalid design intent after retry: {feedback}"}
