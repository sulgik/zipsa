from typing import Optional
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI Codex CLI Provider.
    Uses `codex` command with OPENAI_API_KEY environment variable.
    
    Install: npm install -g @openai/codex
    Auth: export OPENAI_API_KEY=sk-...
    """
    
    def generate(self, prompt: str) -> Optional[str]:
        """Call OpenAI Codex CLI with the given prompt."""
        # Codex CLI uses -p for prompt in non-interactive mode
        command = ["codex", "-p", prompt]
        return self._run_cli(command)
