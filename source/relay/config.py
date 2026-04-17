import os
from pathlib import Path
from dataclasses import dataclass


def _load_dotenv():
    """Load .env file into os.environ if it exists."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not os.environ.get(key):  # Don't override existing env vars
                os.environ[key] = value


_load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Local LLM (Ollama)
    local_model: str = ""
    local_host: str = ""

    # External provider (required for hybrid inference)
    external_provider: str = ""   # anthropic | gemini | openai
    claude_model: str = ""
    gemini_model: str = ""

    # OAuth (optional — enables browser-based auth)
    anthropic_oauth_client_id: str = ""
    anthropic_oauth_client_secret: str = ""
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    oauth_redirect_base: str = ""  # e.g. "http://localhost:8000"

    # Server
    demo_mode: bool = False
    log_dir: str = ""

    def __post_init__(self):
        self.local_model = self.local_model or os.getenv("LOCAL_MODEL", "qwen3.5:9b")
        self.local_host = self.local_host or os.getenv("LOCAL_HOST", "http://localhost:11434")
        external_provider = self.external_provider or os.getenv("EXTERNAL_PROVIDER", "anthropic")
        if external_provider == "claude":
            external_provider = "anthropic"
        self.external_provider = external_provider
        self.claude_model = self.claude_model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        self.gemini_model = self.gemini_model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        self.log_dir = self.log_dir or os.getenv("LOG_DIR", "logs")
        if not self.demo_mode:
            self.demo_mode = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")

        # OAuth
        self.anthropic_oauth_client_id = self.anthropic_oauth_client_id or os.getenv("ANTHROPIC_OAUTH_CLIENT_ID", "")
        self.anthropic_oauth_client_secret = self.anthropic_oauth_client_secret or os.getenv("ANTHROPIC_OAUTH_CLIENT_SECRET", "")
        self.google_oauth_client_id = self.google_oauth_client_id or os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
        self.google_oauth_client_secret = self.google_oauth_client_secret or os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")
        self.oauth_redirect_base = self.oauth_redirect_base or os.getenv("OAUTH_REDIRECT_BASE", "http://localhost:8000")


# Singleton
_settings = None


def get_settings(**overrides) -> Settings:
    global _settings
    if _settings is None or overrides:
        _settings = Settings(**overrides)
    return _settings
