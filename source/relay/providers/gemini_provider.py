import os
from typing import Optional
from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """
    Gemini API Provider via google-genai Python SDK.
    Auth: API key (GEMINI_API_KEY) or OAuth token via TokenStore.
    """

    def __init__(self, token_store=None):
        self.token_store = token_store

    def _client(self):
        from google import genai
        # Prefer OAuth token over API key
        if self.token_store:
            token = self.token_store.get("gemini")
            if token and not token.is_expired:
                from google.oauth2.credentials import Credentials
                creds = Credentials(token=token.access_token)
                return genai.Client(credentials=creds)
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _has_credentials(self) -> bool:
        if self.token_store:
            token = self.token_store.get("gemini")
            if token and not token.is_expired:
                return True
        return bool(os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str) -> Optional[str]:
        """Sync call — used as fallback."""
        if not self._has_credentials():
            print("[Gemini Error] No API key or OAuth token available")
            return None
        model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        try:
            from google.genai import types as _types
            thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "-1"))
            config = _types.GenerateContentConfig(
                thinking_config=_types.ThinkingConfig(thinking_budget=thinking_budget)
            ) if thinking_budget != 0 else None
            kwargs = dict(model=model, contents=prompt)
            if config:
                kwargs["config"] = config
            response = self._client().models.generate_content(**kwargs)
            return response.text.strip() if response.text else None
        except Exception as e:
            print(f"[Gemini Error] {e}")
            return None

    async def generate_async(self, prompt: str, language: str = "ko") -> Optional[str]:
        """Async call with a single prompt string."""
        return await self.generate_async_messages([{"role": "user", "content": prompt}])

    async def generate_async_messages(self, messages: list) -> Optional[str]:
        """
        Multi-turn interface. Accepts a full messages list (sanitized_history + current user message).
        Converts to Gemini Content format; system messages become a system_instruction.
        """
        import asyncio

        # Auto-refresh expired OAuth token before entering thread
        if self.token_store:
            token = self.token_store.get("gemini")
            if token and token.is_expired and token.refresh_token:
                try:
                    from source.auth.oauth_flows import GeminiOAuth
                    from source.relay.config import get_settings
                    s = get_settings()
                    oauth = GeminiOAuth(
                        s.google_oauth_client_id,
                        s.google_oauth_client_secret,
                        f"{s.oauth_redirect_base}/auth/gemini/callback",
                    )
                    refreshed = await oauth.refresh_token(token)
                    self.token_store.set("gemini", refreshed)
                except Exception as e:
                    print(f"[Gemini] Token refresh failed: {e}")

        # Capture client builder for use in thread
        _build_client = self._client
        _has_creds = self._has_credentials

        def _call():
            model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
            if not _has_creds():
                print("[Gemini Error] No API key or OAuth token available")
                return None
            try:
                from google import genai
                from google.genai import types

                system_instruction = None
                contents = []
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if role == "system":
                        system_instruction = content
                    else:
                        gemini_role = "model" if role == "assistant" else "user"
                        contents.append(types.Content(role=gemini_role, parts=[types.Part(text=content)]))

                client = _build_client()
                thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "-1"))
                config_kwargs = {}
                if system_instruction:
                    config_kwargs["system_instruction"] = system_instruction
                if thinking_budget != 0:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=thinking_budget
                    )
                config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

                kwargs = dict(model=model, contents=contents)
                if config:
                    kwargs["config"] = config

                response = client.models.generate_content(**kwargs)
                return response.text.strip() if response.text else None
            except Exception as e:
                print(f"[Gemini Error] {e}")
                return None

        return await asyncio.to_thread(_call)
