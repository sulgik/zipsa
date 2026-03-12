import asyncio
from typing import Optional
from .base import BaseLLMProvider


class LocalProvider(BaseLLMProvider):
    """
    Local LLM Provider using Ollama.
    Uses the same Ollama instance as the sanitizer for reasoning.
    No external API keys needed.
    """

    def __init__(self, ollama_client=None):
        self._ollama = ollama_client

    def generate(self, prompt: str, language: str = "ko") -> Optional[str]:
        """Synchronous wrapper around async Ollama call."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, self._generate_async(prompt, language)
                    ).result(timeout=600)
                return result
            else:
                return loop.run_until_complete(self._generate_async(prompt, language))
        except Exception as e:
            print(f"[LocalProvider Error] {e}")
            return None

    async def _generate_async(self, prompt: str, language: str = "ko") -> Optional[str]:
        if self._ollama is None:
            from source.relay.ollama import OllamaClient
            self._ollama = OllamaClient()

        if language == "ko":
            system = "You are a helpful assistant. Provide concise, accurate answers. Respond in Korean."
        else:
            # "en", "auto", or any other value: respond in the user's language
            system = "You are a helpful assistant. Provide concise, accurate answers in the same language as the question."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return await self._ollama.chat(messages, temperature=0.4)

    async def generate_async(self, prompt: str, language: str = "ko") -> Optional[str]:
        """Direct async interface for use in async contexts."""
        return await self._generate_async(prompt, language)

    async def generate_async_messages(self, messages: list) -> Optional[str]:
        """
        Multi-turn interface. Accepts a full messages list (raw_history + current user message).
        Prepends a system prompt if the first message is not already a system message.
        """
        if self._ollama is None:
            from source.relay.ollama import OllamaClient
            self._ollama = OllamaClient()

        if messages and messages[0].get("role") == "system":
            full_messages = messages
        else:
            system = {
                "role": "system",
                "content": "You are a helpful assistant. Provide concise, accurate answers in the same language as the question.",
            }
            full_messages = [system] + messages

        return await self._ollama.chat(full_messages, temperature=0.4)
