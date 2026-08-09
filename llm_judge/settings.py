
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    token: str
    base_url: str
    model: str
    use_rlm: bool = False
    use_vertex_ai: bool = False     # refreshed Vertex AI tokens vs static API key
    use_strict_json: bool = True    # set to false for Gemini / Groq
    max_concurrency: int = 5       # max simultaneous API calls
    max_retries: int = 20            # retry attempts on rate-limit / server errors
    max_tokens: int = 65536
    reasoning_effort: str | None = None  # set to "low"/"medium"/"high" for thinking models (Gemini, o-series); leave unset for Groq etc.
    enable_thinking: bool = True            # False disables model thinking

