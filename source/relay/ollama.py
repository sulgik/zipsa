import httpx
import json
import os
import time
from typing import List, Dict, Any

class OllamaClient:
    def __init__(self, model: str = None, host: str = None, api_key: str = None):
        self.model = model or os.getenv("LOCAL_MODEL", "qwen3.5:9b")
        self.host = (host or os.getenv("LOCAL_HOST", "http://localhost:11434")).rstrip("/")
        # API key for hosted Ollama endpoints (e.g. Ollama Cloud, custom deployments)
        self.api_key = api_key or os.getenv("LOCAL_API_KEY", "")
        # Persistent HTTP client — reused across all calls to avoid TCP connection overhead
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Return (or lazily create) a persistent AsyncClient for this host."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=60,
            )
            self._client = httpx.AsyncClient(timeout=600.0, limits=limits)
        return self._client

    async def aclose(self) -> None:
        """Close the persistent client (call on app shutdown)."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _is_openai_compatible(self) -> bool:
        """Detect if host is OpenAI-compatible (OpenRouter, vLLM, etc.) vs native Ollama.
        Ollama native: localhost, 127.0.0.1, host.docker.internal, any host with port 11434
        OpenAI-compatible: openrouter.ai, custom cloud endpoints, etc.
        """
        ollama_indicators = ["localhost", "127.0.0.1", "11434", "ollama.com/api", "host.docker.internal"]
        return not any(x in self.host for x in ollama_indicators)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int | None = None,
    ) -> str:
        if self._is_openai_compatible():
            # OpenAI-compatible endpoint (OpenRouter, vLLM, etc.)
            url = f"{self.host}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            response_path = lambda result: result["choices"][0]["message"]["content"]
        else:
            # Native Ollama API
            url = f"{self.host}/api/chat"
            options: Dict[str, Any] = {"temperature": temperature}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": options,
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response_path = lambda result: result["message"]["content"]

        try:
            client = await self._get_client()
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return response_path(result).strip()
        except Exception as e:
            print(f"[Ollama Error] {e}")
            # Reset client on connection errors so next call gets a fresh one
            if isinstance(e, (httpx.ConnectError, httpx.RemoteProtocolError)):
                self._client = None
            return ""

    async def semantic_sanitize_en(self, text: str) -> str:  # kept for reference / experiments
        """
        English semantic sanitizer: avoid summarization, keep structure, only replace sensitive entities.
        """
        system_prompt = """
You are a PII scrubber. Your job: replace sensitive entities while keeping meaning and structure.

Hard rules:
- NO summarization, NO paraphrasing, NO reordering of sentences or bullets.
- Length must stay within ±10% of the input (ignoring placeholder tokens).
- Preserve all existing [PLACEHOLDER] tokens exactly.
- Keep line breaks, bullet structure, punctuation, and non-PII wording unchanged.
- NEVER rewrite instructions or rephrase the user's request. Copy them verbatim except for PII tokens.

Replace ONLY:
- Private person names -> [NAME]. ALL occurrences of the same name MUST be replaced consistently throughout the entire text. This includes:
  * Names with titles: Dr. Smith -> Dr. [NAME]
  * Names in metadata/headers: "From: Jo Peninsulas" -> "From: [NAME]"
  * Names in instructions: "act as Jo" -> "act as [NAME]"
  * Names in conversation turns: "Human: Bob said..." -> "Human: [NAME] said..."
  * Names with possessives: "Butler's research" -> "[NAME]'s research"
  * Names in signatures, greetings, attribution lines
- Personal/profile URLs -> [URL] (e.g., linkedin.com/in/john-doe, instagram.com/janedoe, facebook.com/john.smith)
- Street address/parcel/folio numbers/coordinates -> [ADDRESS]
- API keys, tokens, secrets -> [API_KEY]
- Passwords in any format -> [PASSWORD] (including plaintext passwords like 'iamatester', 'p@ssw0rd')
- Phone numbers -> [PHONE], Emails -> [EMAIL], SSN/ID -> [ID]
- Money amounts ONLY with currency symbols ($, €, ₩) -> [AMOUNT]. Plain numbers are NOT amounts.

DO NOT REPLACE (preserve exactly):
- Well-known public entities: Google, Apple, Microsoft, Amazon, Meta, JetBrains, DuckDuckGo, Wikipedia, GitHub, Docker, Redis, Node.js, React, etc.
- Public organizations: universities, government agencies, cultural centers, NGOs (e.g., "Singapore Chinese Cultural Centre", "MIT", "FDA")
- Fictional/example characters in creative writing, roleplay, or story prompts (e.g., "Aldred", "Thorne" in a fantasy story)
- Song titles, film titles, book titles, game names
- Generic roles and descriptions: "homeowner", "patient", "customer", "the doctor"
- Error codes and status codes (e.g., 2339, 404, 500, ts(2339))
- Version numbers (e.g., v1.2.3, 8.0, Python 3.11)
- Technical identifiers (e.g., k8s, OAuth2, HTTP/2)
- Years in context (e.g., "founded in 1984", "since 2020")
- Dates that are not birth dates (e.g., "JAN 6, 2021", "Mar 23, 2023")
- Timestamps (e.g., "6:43 AM", "3:02 PM", "14:30")
- CSS/code values (e.g., 25px, 0.3s, 100%, opacity:0, blur(25px), translateZ(0))
- Line numbers and diff markers (e.g., @@ -8,8 +8,8 @@)
- Plain counts without currency (e.g., "10 items", "3 steps", "line 42")
- Code variable values (e.g., categoryEntityId: 23, port 8080, index 0)
- Question/problem numbers (e.g., "Question 81", "#42")
- Example/tutorial emails that are clearly illustrative (e.g., user@example.com in a code tutorial)
- Domain names that are not personal profiles (e.g., example.com, github.com/repo)

If unsure whether something is PII or a public entity, keep the original to avoid meaning loss.

Examples:
- "John Smith from Acme Corp at 123 Main St" -> "[NAME] from [COMPANY] at [ADDRESS]"
- "Dr. Butler's research" -> "Dr. [NAME]'s research"
- "act as Jo Peninsulas" -> "act as [NAME]"
- "linkedin.com/in/john-doe" -> "[URL]"
- "password is iamatester" -> "password is [PASSWORD]"
- "Visit the Google Cloud console" -> keep "Google Cloud" (public entity)
- "Set up DuckDuckGo as default" -> keep "DuckDuckGo" (public entity)
- "Write a story about Aldred the knight" -> keep "Aldred" (fictional character)
- "Singapore Chinese Cultural Centre" -> keep as-is (public organization)
- "Error ts(2339) in file.ts" -> keep as-is (error code)
- "@@ -1,3 +1,4 @@ function render()" -> keep as-is (git diff)
- "opacity:0; blur(25px)" -> keep as-is (CSS values)
- "DISCORD_TOKEN: abc123xyz..." -> "DISCORD_TOKEN: [PASSWORD]"

Return ONLY the rewritten text.
"""
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": text},
        ]
        return await self.chat(messages, temperature=0.0)

    async def reformulate(self, text: str, conversation_context: list = None) -> str | None:
        """
        Generate an external-safe, depersonalized reformulation of the query.

        Called only when the formal planner has already decided hybrid routing.
        The conversation_context should be the external-safe sub_thread (not the
        PII-intact main_thread) so the reformulation builds on sanitized history.

        Returns the reformulated string, or None on failure (caller falls back to local).
        """
        system_prompt = """You are a privacy-preserving query reformulator.

Your job: produce a depersonalized version of the user's query that reveals no identifying
information, so it can be safely sent to an external AI service for knowledge assistance.

If conversation history is provided, it represents the prior external-safe turns.
Use it to understand context, but do not introduce any identifying details into the output.

OUTPUT: A single JSON object — no markdown, no explanation:
{"reformulated": "<depersonalized knowledge request>"}

Reformulation rules:
1. Remove ALL identifiers: names, DOB (use age range: "late 50s"), IDs, contact info, specific dates
2. Replace semantic identifiers with generic categories:
   - "senior ER physician at City General" → "healthcare professional (physician)"
   - "principal cellist at a symphony" → "professional musician (string instrument)"
   - "software engineer at Google" → "technology professional (software engineer)"
   - "Samsung Seoul Hospital" → "a major hospital"
3. Keep ALL task-relevant parameters precisely: lab values, measurements, code, error messages, constraints
4. Reformulate as a follow-up to the sanitized conversation history (if any).

EXAMPLES:
Input:  "My patient John K. (DOB 03/15/1966, MRN 445892) has been on metformin 2000mg + sitagliptin 100mg for 6 months. HbA1c worsened 7.8→8.4%, eGFR 62. What are the escalation options for his T2DM?"
Output: {"reformulated": "A healthcare professional (physician) asks about a patient in their late 50s with T2DM. HbA1c worsening 7.8→8.4% over 6 months on current regimen: metformin 2000mg + DPP-4i (sitagliptin 100mg), eGFR 62. Rank the top escalation strategies with expected HbA1c reduction, renal dosing requirements, and monitoring needs."}

Input:  "John Smith (john@acme.com) is our CTO. Why are our AWS Lambda cold starts over 3s with a 512MB Node.js function?"
Output: {"reformulated": "AWS Lambda cold starts exceeding 3 seconds with a 512MB Node.js function. What are the top causes and ranked mitigation strategies?"}

Now reformulate this query:"""

        import json as _json
        import re as _re

        messages = [{"role": "system", "content": system_prompt.strip()}]
        if conversation_context:
            messages.extend(conversation_context)
        messages.append({"role": "user", "content": text})
        # max_tokens=256: reformulated query is always a single short JSON object
        raw = await self.chat(messages, temperature=0.1, max_tokens=256)

        try:
            clean = _re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("no JSON object found")
            result = _json.loads(clean[start:end])
            return result.get("reformulated") or None
        except Exception as e:
            print(f"[Reformulate Parse Error] {e} | raw={raw[:200]}")
            return None

    async def plan_execution(self, text: str, conversation_context: list = None) -> dict:
        """
        Decide the execution path and, when needed, prepare an external-safe query.

        Returns a dict: {"execution_path": "hybrid" | "local", "external_query": str | None}

        execution_path="local"  → skip external call; local LLM handles the query entirely.
        execution_path="hybrid" → run local (original) + external (external_query) in parallel.

        conversation_context: prior turns from the local main thread. This runs inside the
        trusted zone, so the planner may use original conversation context to decide
        whether an external-safe reformulation is possible and useful.
        Format: [{"role": "user"|"assistant", "content": str}, ...]
        """
        system_prompt = """You are a privacy-preserving query router and reformulator.

Your job: Decide whether external AI knowledge would improve the answer, and if so,
produce a depersonalized reformulation that reveals no identifying information.

If conversation history is provided, use it to understand context inside the trusted
local environment. That history may contain private details. Do not repeat those
details in the reformulated external query unless they can be safely abstracted away.

OUTPUT: A single JSON object — no markdown, no explanation:
{"route": "hybrid", "reformulated": "<depersonalized knowledge request>"}
  or
{"route": "local"}

WHEN to use route="local" (external knowledge not needed or not safe to use):
- The answer IS the private data itself (e.g., "what is my password?", "look up John's record")
- Simple personal tasks where general knowledge adds nothing (e.g., "write a birthday message for my wife Sarah")
- The question is inseparable from the person's identity (re-identification unavoidable)
- Conversational/emotional support where context matters more than external knowledge

WHEN to use route="hybrid" (external knowledge improves the answer):
- Domain knowledge questions (medical, legal, financial, technical) where general expertise helps
- Analysis or comparison tasks where the specific context can be abstracted
- Code or technical problems where the personal details are incidental
- Any question where a depersonalized version can be meaningfully answered by an expert

FOR HYBRID — reformulation rules:
1. Remove ALL identifiers: names, DOB (use age range: "late 50s"), IDs, contact info, specific dates
2. Replace semantic identifiers with generic categories:
   - "senior ER physician at City General" → "healthcare professional (physician)"
   - "principal cellist at a symphony" → "professional musician (string instrument)"
   - "software engineer at Google" → "technology professional (software engineer)"
   - "Samsung Seoul Hospital" → "a major hospital"
   - "the 2023 Itaewon incident" → "a large crowd event"
3. Keep ALL task-relevant parameters precisely: lab values, measurements, code, error messages, constraints
4. Reformulate as a follow-up to the sanitized conversation history above (if any).

EXAMPLES:
Input:  "My patient John K. (DOB 03/15/1966, MRN 445892) has been on metformin 2000mg + sitagliptin 100mg for 6 months. HbA1c worsened 7.8→8.4%, eGFR 62. What are the escalation options for his T2DM?"
Output: {"route": "hybrid", "reformulated": "A healthcare professional (physician) asks about a patient in their late 50s with T2DM. HbA1c worsening 7.8→8.4% over 6 months on current regimen: metformin 2000mg + DPP-4i (sitagliptin 100mg), eGFR 62. Rank the top escalation strategies with expected HbA1c reduction, renal dosing requirements, and monitoring needs."}

Input:  "John Smith (john@acme.com) is our CTO. Why are our AWS Lambda cold starts over 3s with a 512MB Node.js function?"
Output: {"route": "hybrid", "reformulated": "AWS Lambda cold starts exceeding 3 seconds with a 512MB Node.js function. What are the top causes and ranked mitigation strategies? For each: expected latency improvement and implementation complexity."}

Input:  "What is my wife Sarah's birthday? She was born on March 15, 1985."
Output: {"route": "local"}

Input:  "Write a thank-you note to my boss Dr. Kim at Samsung for the promotion."
Output: {"route": "local"}

Now analyze this query:"""

        import json as _json
        import re as _re

        messages = [{"role": "system", "content": system_prompt.strip()}]
        if conversation_context:
            messages.extend(conversation_context)
        messages.append({"role": "user", "content": text})
        raw = await self.chat(messages, temperature=0.1)

        # Parse JSON response
        try:
            # Strip markdown fences if present
            clean = _re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            # Extract first JSON object
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("no JSON object found")
            result = _json.loads(clean[start:end])
            execution_path = result.get("route", "hybrid")
            if execution_path not in ("hybrid", "local"):
                execution_path = "hybrid"
            external_query = result.get("reformulated") or None
            return {"execution_path": execution_path, "external_query": external_query}
        except Exception as e:
            print(f"[Planning Parse Error] {e} | raw={raw[:200]}")
            # Fallback: treat raw output as an external-safe query proposal, go hybrid
            return {"execution_path": "hybrid", "external_query": raw or text}
