import os
from typing import Optional
from .base import BaseLLMProvider


class ClaudeProvider(BaseLLMProvider):
    """
    Claude API Provider using Anthropic SDK.

    Auth: Set ANTHROPIC_API_KEY in .env or environment.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

    def generate(self, prompt: str) -> Optional[str]:
        """Call Claude API synchronously."""
        if not self.api_key:
            print("[ClaudeProvider] ANTHROPIC_API_KEY not set")
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            print(f"[ClaudeProvider Error] {e}")
            return None

    async def generate_async(self, prompt: str, language: str = "ko") -> Optional[str]:
        """Call Claude API asynchronously with a single prompt."""
        return await self.generate_async_messages([{"role": "user", "content": prompt}])

    async def generate_async_messages(self, messages: list) -> Optional[str]:
        """
        Multi-turn interface. Accepts a full messages list (sanitized_history + current user message).
        System messages are extracted and passed separately per Anthropic API convention.
        """
        if not self.api_key:
            print("[ClaudeProvider] ANTHROPIC_API_KEY not set")
            return None
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            # Separate system message (Anthropic API takes it as a top-level param)
            system_content = None
            chat_messages = []
            for m in messages:
                if m.get("role") == "system":
                    system_content = m["content"]
                else:
                    chat_messages.append({"role": m["role"], "content": m["content"]})

            kwargs = dict(model=self.model, max_tokens=4096, messages=chat_messages)
            if system_content:
                kwargs["system"] = system_content

            message = await client.messages.create(**kwargs)
            return message.content[0].text
        except Exception as e:
            print(f"[ClaudeProvider Error] {e}")
            return None
