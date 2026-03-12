import os
from typing import Optional
from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """
    Gemini API Provider via google-genai Python SDK.
    Requires GEMINI_API_KEY in environment.
    """

    def _client(self):
        from google import genai
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str) -> Optional[str]:
        """Sync call — used as fallback."""
        if not os.getenv("GEMINI_API_KEY"):
            print("[Gemini Error] GEMINI_API_KEY not set")
            return None
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        try:
            response = self._client().models.generate_content(model=model, contents=prompt)
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

        def _call():
            api_key = os.getenv("GEMINI_API_KEY")
            model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            if not api_key:
                print("[Gemini Error] GEMINI_API_KEY not set")
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
                        # Gemini uses "model" instead of "assistant"
                        gemini_role = "model" if role == "assistant" else "user"
                        contents.append(types.Content(role=gemini_role, parts=[types.Part(text=content)]))

                client = genai.Client(api_key=api_key)
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                ) if system_instruction else None

                kwargs = dict(model=model, contents=contents)
                if config:
                    kwargs["config"] = config

                response = client.models.generate_content(**kwargs)
                return response.text.strip() if response.text else None
            except Exception as e:
                print(f"[Gemini Error] {e}")
                return None

        return await asyncio.to_thread(_call)
