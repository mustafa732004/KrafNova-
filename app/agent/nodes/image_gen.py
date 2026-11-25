"""Node 2: generate candidate images from the validated intent.

Three providers behind settings.image_provider:
  - "stub" (default): placeholder candidates, no key needed, so the graph
    runs end to end in development.
  - "gemini": real image model via the google-genai SDK. Rasters are saved
    under settings.generated_dir and served at /generated.
  - "pollinations": free, no API key, pollinations.ai image endpoint.
    Same output as gemini (PNG files under generated_dir).

TODO: persist candidates per request id instead of unique files forever,
and add a cleanup job.
"""
import logging
import os
import re
import functools
import io
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image
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


@functools.lru_cache(maxsize=64)
def _english_subject(subject: str) -> str:
    """Image models follow English prompts far better. Translate non-ASCII
    subjects (e.g. Arabic) to a short English visual description; fall back
    to the original text if the call fails."""
    if subject.isascii():
        return subject
    try:
        resp = _get_client().models.generate_content(
            model=settings.gemini_model,
            contents=(
                "Translate this design subject into a short English visual "
                "description for an image generator. Output only the English "
                f"text, no quotes: {subject}"
            ),
        )
        return (resp.text or "").strip() or subject
    except Exception as exc:
        log.warning("subject translation failed, using original: %s", exc)
        return subject


def _prompt(intent: DesignIntent, i: int) -> str:
    subject = _english_subject(intent.subject)
    return (
        f"Black and white {intent.style} of {subject}, flat 2D drawing, "
        f"clean single-weight black strokes on a plain white background, "
        f"suitable for CNC vector tracing, no shading, no color, no text, "
        f"no people, no photo. Variation {i}."
        + (f" Note: {intent.notes}" if intent.notes and intent.notes.isascii() else "")
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


def _fetch_pollinations(prompt: str, seed: int) -> bytes:
    url = (
        f"{settings.pollinations_base_url}/{urllib.parse.quote(prompt, safe='')}"
        f"?width=1024&height=1024&nologo=true&seed={seed}"
        f"&model={urllib.parse.quote(settings.pollinations_model)}"
    )
    # Anonymous use is rate limited per IP (HTTP 402/429 "Retry after Ns"),
    # so wait for the advertised delay and retry.
    last: Exception | None = None
    for attempt in range(6):
        delay = 3 * (attempt + 1)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KrafNova/0.1"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
            Image.open(io.BytesIO(data)).verify()  # make sure it is a real image
            return data
        except urllib.error.HTTPError as exc:
            last = exc
            body = exc.read().decode("utf-8", "ignore")
            m = re.search(r"[Rr]etry after (\d+)s", body)
            if m:
                delay = int(m.group(1)) + 1
            log.warning("pollinations attempt %s: HTTP %s, retry in %ss", attempt + 1, exc.code, delay)
        except Exception as exc:  # network, bad image
            last = exc
            log.warning("pollinations attempt %s failed: %s", attempt + 1, exc)
        time.sleep(delay)
    raise RuntimeError(f"pollinations failed: {last}")


def _pollinations_candidates(intent: DesignIntent) -> list[Candidate]:
    os.makedirs(settings.generated_dir, exist_ok=True)
    batch = uuid.uuid4().hex[:8]
    candidates: list[Candidate] = []
    for i in range(1, settings.image_candidates + 1):
        try:
            raw = _fetch_pollinations(_prompt(intent, i), seed=i)
        except RuntimeError as exc:
            log.warning("image candidate %s skipped: %s", i, exc)
            continue
        filename = f"{batch}-cand-{i}.png"
        path = os.path.join(settings.generated_dir, filename)
        Image.open(io.BytesIO(raw)).convert("RGB").save(path, "PNG")
        candidates.append(
            Candidate(id=f"cand-{i}", image_url=f"/generated/{filename}", local_path=path)
        )
    return candidates


def generate_images(state: dict) -> dict:
    intent = state["intent"]
    if settings.image_provider == "gemini":
        return {"candidates": _gemini_candidates(intent)}
    if settings.image_provider == "pollinations":
        return {"candidates": _pollinations_candidates(intent)}
    return {"candidates": _stub_candidates(intent)}
