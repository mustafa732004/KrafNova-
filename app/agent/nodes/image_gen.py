"""Node 2: generate candidate images from the validated intent.

Two providers behind settings.image_provider:
  - "stub" (default): placeholder candidates, no key needed, so the graph
    runs end to end in development.
  - "gemini": real image model via the google-genai SDK. Rasters are saved
    under settings.generated_dir and served at /generated.

TODO: persist candidates per request id instead of unique files forever,
and add a cleanup job.
"""
import logging
import os
import uuid

from google import genai
from google.genai import types

from app.config import settings
from app.schemas.design_intent import Candidate, DesignIntent

log = logging.getLogger(__name__)

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _prompt(intent: DesignIntent, i: int) -> str:
    return (
        f"Black and white {intent.style} of {intent.subject}, "
        f"clean single-weight strokes on a plain white background, "
        f"suitable for CNC vector tracing, no shading, no text. Variation {i}."
        + (f" Note: {intent.notes}" if intent.notes else "")
    )


def _stub_candidates(intent: DesignIntent) -> list[Candidate]:
    return [
        Candidate(id=f"cand-{i}", image_url=f"placeholder://{intent.subject}/{i}")
        for i in range(1, settings.image_candidates + 1)
    ]


def _gemini_candidates(intent: DesignIntent) -> list[Candidate]:
    os.makedirs(settings.generated_dir, exist_ok=True)
    batch = uuid.uuid4().hex[:8]
    candidates: list[Candidate] = []
    for i in range(1, settings.image_candidates + 1):
        response = _get_client().models.generate_content(
            model=settings.image_model,
            contents=_prompt(intent, i),
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        png = next(
            (p.inline_data.data for p in response.candidates[0].content.parts if p.inline_data),
            None,
        )
        if png is None:
            log.warning("image candidate %s returned no image data, skipping", i)
            continue
        filename = f"{batch}-cand-{i}.png"
        path = os.path.join(settings.generated_dir, filename)
        with open(path, "wb") as fh:
            fh.write(png)
        candidates.append(
            Candidate(id=f"cand-{i}", image_url=f"/generated/{filename}", local_path=path)
        )
    return candidates


def generate_images(state: dict) -> dict:
    intent = state["intent"]
    if settings.image_provider == "gemini":
        return {"candidates": _gemini_candidates(intent)}
    return {"candidates": _stub_candidates(intent)}
