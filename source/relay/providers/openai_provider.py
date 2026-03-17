import os
from typing import Optional
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API Provider via openai Python SDK.
    Requires OPENAI_API_KEY in environment.
    Model: OPENAI_MODEL env var (default: gpt-4o-mini).
    """

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _model(self) -> str:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str) -> Optional[str]:
        if not os.getenv("OPENAI_API_KEY"):
            print("[OpenAI Error] OPENAI_API_KEY not set")
            return None
        try:
            response = self._client().chat.completions.create(
                model=self._model(),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAI Error] {e}")
            return None

    async def generate_async(self, prompt: str, language: str = "en") -> Optional[str]:
        return await self.generate_async_messages([{"role": "user", "content": prompt}])

    async def generate_async_messages(self, messages: list) -> Optional[str]:
        import asyncio

        def _call():
            if not os.getenv("OPENAI_API_KEY"):
                print("[OpenAI Error] OPENAI_API_KEY not set")
                return None
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                oai_messages = []
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    # OpenAI accepts system/user/assistant
                    oai_messages.append({"role": role, "content": content})
                response = client.chat.completions.create(
                    model=self._model(),
                    messages=oai_messages,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[OpenAI Error] {e}")
                return None

        return await asyncio.to_thread(_call)
