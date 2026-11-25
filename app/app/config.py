"""Settings read from environment variables. No secrets live in the repo."""
import os


class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "stub")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    image_provider: str = os.getenv("IMAGE_PROVIDER", "stub")
    image_api_key: str = os.getenv("IMAGE_API_KEY", "")


settings = Settings()
