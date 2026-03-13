import httpx
import json
import os
import time
from typing import List, Dict, Any

class OllamaClient:
    def __init__(self, model: str = None, host: str = None):
        self.model = model or os.getenv("LOCAL_MODEL", "qwen3.5:9b")
        self.host = host or os.getenv("LOCAL_HOST", "http://localhost:11434")

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.5) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result['message']['content'].strip()
        except Exception as e:
            print(f"[Ollama Error] {e}")
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

    async def plan_execution(self, text: str, sanitized_context: list = None) -> dict:
        """
        Decide the execution path and, when needed, prepare an external-safe query.

        Returns a dict: {"execution_path": "hybrid" | "local", "external_query": str | None}

        execution_path="local"  → skip external call; local LLM handles the query entirely.
        execution_path="hybrid" → run local (original) + external (external_query) in parallel.

        sanitized_context: prior turns already external-safe (safe to include as conversation
        history so the reformulator understands ongoing context without re-exposing PII).
        Format: [{"role": "user"|"assistant", "content": str}, ...]
        """
        system_prompt = """You are a privacy-preserving query router and reformulator.

Your job: Decide whether external AI knowledge would improve the answer, and if so,
produce a depersonalized reformulation that reveals no identifying information.

If conversation history is provided, use it to understand context — it is already
depersonalized. Your reformulation of the new query must be consistent with that history.

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
Input:  "Jane Smith (SSN 123-45-6789) is a senior ER physician. Her patient's HbA1c worsened 7.8→8.4% over 6 months on metformin 2000mg + sitagliptin 100mg (eGFR 62). Next treatment options?"
Output: {"route": "hybrid", "reformulated": "A patient in their late 50s with T2DM. HbA1c worsening 7.8→8.4% over 6 months. Current regimen: metformin 2000mg + DPP-4i (sitagliptin 100mg), eGFR 62. Rank the top escalation strategies with expected HbA1c reduction, renal dosing requirements, and monitoring needs."}

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
        if sanitized_context:
            messages.extend(sanitized_context)
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
