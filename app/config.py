"""Settings read from environment variables. No secrets live in the repo.

Copy .env.example to .env and fill it in; python-dotenv loads it here.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # Gemini text model: design intent (structured output) + explainer.
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # Image generation. "stub" (default) returns placeholder candidates and
    # needs no key; "gemini" calls a real image model; "pollinations" is
    # free and needs no key; "horde" is AI Horde, free, with negative-prompt
    # support for clean line-art output.
    image_provider: str = os.getenv("IMAGE_PROVIDER", "stub")
    image_model: str = os.getenv("IMAGE_MODEL", "gemini-2.5-flash-image")
    # "pollinations" provider: free, no key. Model names: flux, turbo, ...
    pollinations_model: str = os.getenv("POLLINATIONS_MODEL", "flux")
    pollinations_base_url: str = os.getenv(
        "POLLINATIONS_BASE_URL", "https://image.pollinations.ai/prompt"
    )
    # "horde" provider: AI Horde (stablehorde.net), free; the default key is
    # the anonymous one. Only free provider with negative-prompt support.
    horde_api_key: str = os.getenv("HORDE_API_KEY", "0000000000")
    horde_base_url: str = os.getenv("HORDE_BASE_URL", "https://stablehorde.net/api/v2")
    horde_model: str = os.getenv("HORDE_MODEL", "SDXL 1.0")
    horde_timeout: int = int(os.getenv("HORDE_TIMEOUT", "600"))
    image_candidates: int = int(os.getenv("IMAGE_CANDIDATES", "3"))

    # Where generated rasters are saved (also served at /generated).
    generated_dir: str = os.getenv("GENERATED_DIR", "generated")

    # Explainer: use the LLM when a key is present, otherwise a template.
    explain_with_llm: bool = _bool("EXPLAIN_WITH_LLM", True)

    # Vectorization: max raster side in px before tracing (keeps SVGs small).
    vector_max_px: int = int(os.getenv("VECTOR_MAX_PX", "512"))
    # Tracer backend: "vtracer" (smooth curves, needs the vtracer package,
    # falls back to rects when missing) or "rects" (dependency-free blocks).
    vector_backend: str = os.getenv("VECTOR_BACKEND", "vtracer")
    # vtracer: speckle filter in px (drops tiny blobs), and the fraction of
    # the bottom edge cropped before tracing to cut provider watermarks.
    vector_filter_speckle: int = int(os.getenv("VECTOR_FILTER_SPECKLE", "16"))
    vector_watermark_crop: float = float(os.getenv("VECTOR_WATERMARK_CROP", "0.06"))


settings = Settings()
