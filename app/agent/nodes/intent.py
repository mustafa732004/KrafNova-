"""Node 1: turn the user's message into a validated DesignIntent.

TODO: replace the stub with a real LLM call that returns JSON, then validate
it with DesignIntent.model_validate(...). Keep validation here, it is the
boundary between the LLM and the deterministic pipeline.
"""
from pydantic import ValidationError

from app.schemas.design_intent import Dimensions, DesignIntent


def _call_llm(message: str) -> dict:
    # PLACEHOLDER: real model call goes here.
    return {
        "subject": message.strip() or "untitled",
        "style": "line_art",
        "dimensions": {"width_mm": 100, "height_mm": 100},
    }


def parse_intent(state: dict) -> dict:
    raw = _call_llm(state["message"])
    try:
        return {"intent": DesignIntent.model_validate(raw), "error": None}
    except ValidationError as exc:
        return {"intent": None, "error": f"invalid design intent: {exc.error_count()} errors"}
