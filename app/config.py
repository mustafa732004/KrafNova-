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
    # needs no key; "gemini" calls a real image model.
    image_provider: str = os.getenv("IMAGE_PROVIDER", "stub")
    image_model: str = os.getenv("IMAGE_MODEL", "gemini-2.5-flash-image")
    image_candidates: int = int(os.getenv("IMAGE_CANDIDATES", "3"))

    # Where generated rasters are saved (also served at /generated).
    generated_dir: str = os.getenv("GENERATED_DIR", "generated")

    # Explainer: use the LLM when a key is present, otherwise a template.
    explain_with_llm: bool = _bool("EXPLAIN_WITH_LLM", True)

    # Vectorization: max raster side in px before tracing (keeps SVGs small).
    vector_max_px: int = int(os.getenv("VECTOR_MAX_PX", "512"))


settings = Settings()
