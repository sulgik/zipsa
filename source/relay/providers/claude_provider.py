import os
from typing import Optional
from .base import BaseLLMProvider


class ClaudeProvider(BaseLLMProvider):
    """
    Claude API Provider using Anthropic SDK.

    Auth: API key (ANTHROPIC_API_KEY) or OAuth token via TokenStore.
    """

    def __init__(self, api_key: str = None, model: str = None, token_store=None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
        self.token_store = token_store

    def _effective_key(self) -> Optional[str]:
        """Return OAuth access_token if available, else API key."""
        if self.token_store:
            token = self.token_store.get("anthropic")
            if token and not token.is_expired:
                return token.access_token
        return self.api_key or None

    async def _resolve_key_async(self) -> Optional[str]:
        """Like _effective_key but auto-refreshes expired OAuth tokens."""
        if self.token_store:
            token = self.token_store.get("anthropic")
            if token:
                if not token.is_expired:
                    return token.access_token
                if token.refresh_token:
                    try:
                        from source.auth.oauth_flows import AnthropicOAuth
                        from source.relay.config import get_settings
                        s = get_settings()
                        oauth = AnthropicOAuth(
                            s.anthropic_oauth_client_id,
                            s.anthropic_oauth_client_secret,
                            f"{s.oauth_redirect_base}/auth/claude/callback",
                        )
                        refreshed = await oauth.refresh_token(token)
                        self.token_store.set("anthropic", refreshed)
                        return refreshed.access_token
                    except Exception as e:
                        print(f"[ClaudeProvider] Token refresh failed: {e}")
        return self.api_key or None

    def generate(self, prompt: str) -> Optional[str]:
        """Call Claude API synchronously."""
        key = self._effective_key()
        if not key:
            print("[ClaudeProvider] No API key or OAuth token available")
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
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
        key = await self._resolve_key_async()
        if not key:
            print("[ClaudeProvider] No API key or OAuth token available")
            return None
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=key)

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
