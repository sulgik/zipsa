"""
Selective Query Router for Zipsa Privacy Gateway.

두 가지 라우팅 전략을 모두 지원합니다:

── Selective Routing (route_query_selective) ──────────────────────────────────
  단일 경로: 쿼리를 LOCAL 또는 EXTERNAL 중 하나로 라우팅.
  1. PII severity threshold override  → force LOCAL
  2. Fast heuristic patterns (<1 ms, keyword/regex)
  3. LLM-based classification (Ollama, ~2-5 s) when heuristic is inconclusive

── Dual-Path Synthesis (route_query_dual) ─────────────────────────────────────
  이중 경로: 로컬 LLM이 직접 응답(3a) + 프로토콜 LLM 응답(3b) → 로컬 합성(4).
  LLM 카테고리 분류 없이 PII severity 점수만으로 외부 전송 가능 여부를 결정.
  (LLM classifier 정확도 낮음 → 결정론적 threshold 방식으로 대체)

PII Severity reference (empirically calibrated):
  ID / SSN        10  →  highest-risk; almost always force local
  CARD_NUMBER      9
  ACCOUNT_ID/API_KEY/PASSWORD  8
  PHONE            6
  EMAIL            5
  NAME / ADDRESS / DIAGNOSIS   4
  HOSPITAL / AMOUNT / IP       3
  DRUG / COMPANY / PERCENT     2

Default force-local threshold: 12
  • score ≥ 12  →  force LOCAL
  • score < 12  →  selective: use heuristic / LLM; dual: use external provider

Empirical category win-rates (LOCAL=A, EXTERNAL=B, n=100 pairs):
  roleplay_persona      LOCAL  80%
  pii_dependent         LOCAL  56%
  code_technical        EXTERNAL 84%
  text_rewrite          EXTERNAL 62%
  structured_generation EXTERNAL 62%
  domain_knowledge      EXTERNAL 67%
  analysis_evaluation   EXTERNAL 57%
  information_request   EXTERNAL 57%
  simple_instruction    EXTERNAL 50%
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List


# ── PII severity table ────────────────────────────────────────────────────────

PII_SEVERITY_SCORES: dict[str, int] = {
    "ID":           10,   # 주민등록번호, SSN, passport
    "CARD_NUMBER":   9,   # 신용 / 체크카드
    "ACCOUNT_ID":    8,   # 은행계좌번호
    "API_KEY":       8,   # API keys, tokens
    "PASSWORD":      8,   # 비밀번호
    "PHONE":         6,   # 전화번호
    "EMAIL":         5,   # 이메일
    "NAME":          4,   # 개인 이름
    "ADDRESS":       4,   # 주소
    "DIAGNOSIS":     4,   # 의료 진단명
    "HOSPITAL":      3,   # 병원명
    "AMOUNT":        3,   # 금액
    "IP_ADDRESS":    3,   # IP 주소
    "DRUG":          2,   # 약물명
    "COMPANY":       2,   # 회사명
    "PERCENT":       2,   # 비율
}

DEFAULT_FORCE_LOCAL_THRESHOLD = 12


def compute_pii_sensitivity(pii_types: List[str]) -> int:
    """Return summed severity score for the given list of detected PII types."""
    return sum(PII_SEVERITY_SCORES.get(t.upper().strip("[]"), 3) for t in pii_types)


# ── Routing decision ──────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    route: str           # "local" | "external"
    category: str        # query category string
    confidence: str      # "high" | "medium"
    reason: str          # human-readable explanation
    sensitivity_score: int = 0    # total PII severity score
    forced_local: bool = False    # True if threshold override applied


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: SELECTIVE ROUTING (heuristic + LLM classify)
# ══════════════════════════════════════════════════════════════════════════════

# ── Category → default route ─────────────────────────────────────────────────

CATEGORY_ROUTES: dict[str, str] = {
    "roleplay_persona":      "local",
    "pii_dependent":         "local",
    "crisis_sensitive":      "local",
    "code_technical":        "external",
    "text_rewrite":          "external",
    "structured_generation": "external",
    "domain_knowledge":      "external",
    "analysis_evaluation":   "external",
    "information_request":   "external",
    "simple_instruction":    "external",
    "other":                 "external",
}

# ── Heuristic patterns ────────────────────────────────────────────────────────

CODE_PATTERNS = [
    re.compile(r'```'),
    re.compile(r'\bdef\s+\w+\s*\('),
    re.compile(r'\bfunction\s+\w+\s*\('),
    re.compile(r'\bclass\s+\w+'),
    re.compile(r'\bimport\s+\w+'),
    re.compile(r'\bconst\s+\w+\s*='),
    re.compile(r'boto3|subprocess|httpx|axios|fetch\('),
    re.compile(r'\b(debug|error|bug|fix|코드|디버그|오류)\b', re.IGNORECASE),
]

ROLEPLAY_PATTERNS = [
    re.compile(r'(act\s+as|pretend|roleplay|처럼|인\s*척|역할|캐릭터)', re.IGNORECASE),
    re.compile(r'(you\s+are\s+a|너는\s+.+이다|당신은\s+.+입니다)', re.IGNORECASE),
]

CRISIS_PATTERNS = [
    re.compile(r'(자살|자해|극단적\s*선택|suicide|self[\s\-]?harm)', re.IGNORECASE),
    re.compile(r'(학대|폭력|피해|abuse|violence|victim)', re.IGNORECASE),
]

PII_CONTENT_KEYWORDS = [
    "주민등록번호", "비밀번호", "계좌번호", "카드번호",
    "연락처", "개인정보", "비밀", "기밀", "인사", "연봉", "급여",
    "salary", "ssn", "password", "confidential", "private",
    "이력서", "resume", "cover letter",
]

DOMAIN_KNOWLEDGE_PATTERNS = [
    re.compile(r'(치료|진단|증상|처방|용법|부작용|medication|diagnosis|treatment|symptom|dosage)', re.IGNORECASE),
    re.compile(r'(법률|판례|계약|소송|조항|legal|lawsuit|court|statute|clause)', re.IGNORECASE),
    re.compile(r'(재무|투자|주가|M&A|실사|financial|investment|stock|portfolio|due.?diligence)', re.IGNORECASE),
]

TEXT_REWRITE_PATTERNS = [
    re.compile(r'(rewrite|rephrase|paraphrase|수정|다시\s*써|고쳐|번역|translate)', re.IGNORECASE),
    re.compile(r'(proofread|교정|문법|grammar|edit\s+this)', re.IGNORECASE),
]

STRUCTURED_GEN_PATTERNS = [
    re.compile(r'(표로|테이블|table|csv|json|yaml|스키마|schema|sql)', re.IGNORECASE),
    re.compile(r'(목록|리스트|list\s+of|bullet\s+point|항목)', re.IGNORECASE),
    re.compile(r'(템플릿|template|양식|form\s+letter|format)', re.IGNORECASE),
]

# ── LLM classification prompt ─────────────────────────────────────────────────

CLASSIFY_PROMPT = """Classify this query into exactly ONE category. Output ONLY the category name, nothing else.

Categories:
- roleplay_persona: Acting as a character, persona, role-playing
- pii_dependent: Query where PII/private data IS the answer (contact lookup, salary inquiry)
- crisis_sensitive: Mental health crisis, self-harm, abuse situations
- code_technical: Code debugging, programming, technical implementation
- text_rewrite: Rewriting, translating, proofreading text
- structured_generation: Generate tables, lists, templates, structured output
- domain_knowledge: Specific domain expertise (medical, legal, financial facts)
- analysis_evaluation: Analyze data, evaluate options, compare alternatives
- information_request: Factual questions, how-to, explanations
- simple_instruction: Short tasks, calculations, simple commands

Query: {query}

Category:"""


# ── Heuristic stage ───────────────────────────────────────────────────────────

def route_heuristic(
    query: str,
    pii_detected: bool,
    pii_types: Optional[List[str]] = None,
    sensitivity_threshold: int = DEFAULT_FORCE_LOCAL_THRESHOLD,
) -> Optional[RoutingDecision]:
    """
    Fast heuristic routing (<1 ms).
    Returns None when inconclusive (caller should use LLM classification).

    Priority order:
      1. PII severity threshold override  → force LOCAL
      2. Crisis / sensitive content       → force LOCAL
      3. Code patterns (≥2 matches)       → EXTERNAL  (high confidence)
      4. Roleplay patterns                → LOCAL     (high confidence)
      5. PII-dependent keywords + PII     → LOCAL     (medium)
      6. Domain knowledge patterns        → EXTERNAL  (medium)
      7. Text rewrite patterns            → EXTERNAL  (medium)
      8. Structured generation patterns   → EXTERNAL  (medium)
    """
    pii_types = pii_types or []
    sensitivity = compute_pii_sensitivity(pii_types)

    # 1. PII severity override
    if sensitivity >= sensitivity_threshold and pii_types:
        type_list = ", ".join(sorted(set(pii_types)))
        return RoutingDecision(
            route="local",
            category="pii_dependent",
            confidence="high",
            reason=f"PII severity {sensitivity} ≥ threshold {sensitivity_threshold} ({type_list})",
            sensitivity_score=sensitivity,
            forced_local=True,
        )

    # 2. Crisis content
    for p in CRISIS_PATTERNS:
        if p.search(query):
            return RoutingDecision(
                route="local",
                category="crisis_sensitive",
                confidence="high",
                reason="Crisis / sensitive content detected — always local",
                sensitivity_score=sensitivity,
                forced_local=True,
            )

    # 3. Code patterns
    code_matches = sum(1 for p in CODE_PATTERNS if p.search(query))
    if code_matches >= 2:
        return RoutingDecision(
            route="external",
            category="code_technical",
            confidence="high",
            reason=f"Code patterns detected ({code_matches} signals)",
            sensitivity_score=sensitivity,
        )

    # 4. Roleplay
    for p in ROLEPLAY_PATTERNS:
        if p.search(query):
            return RoutingDecision(
                route="local",
                category="roleplay_persona",
                confidence="high",
                reason="Roleplay / persona pattern detected",
                sensitivity_score=sensitivity,
            )

    # 5. PII-dependent (query IS about private data)
    if pii_detected:
        hits = sum(1 for kw in PII_CONTENT_KEYWORDS if kw.lower() in query.lower())
        if hits >= 1:
            return RoutingDecision(
                route="local",
                category="pii_dependent",
                confidence="medium",
                reason=f"PII-dependent keywords ({hits} matches) with detected PII",
                sensitivity_score=sensitivity,
            )

    # 6. Domain knowledge
    domain_hits = sum(1 for p in DOMAIN_KNOWLEDGE_PATTERNS if p.search(query))
    if domain_hits >= 1:
        return RoutingDecision(
            route="external",
            category="domain_knowledge",
            confidence="medium",
            reason=f"Domain knowledge signals ({domain_hits} matches)",
            sensitivity_score=sensitivity,
        )

    # 7. Text rewrite
    for p in TEXT_REWRITE_PATTERNS:
        if p.search(query):
            return RoutingDecision(
                route="external",
                category="text_rewrite",
                confidence="medium",
                reason="Text rewrite pattern detected",
                sensitivity_score=sensitivity,
            )

    # 8. Structured generation
    struct_hits = sum(1 for p in STRUCTURED_GEN_PATTERNS if p.search(query))
    if struct_hits >= 1:
        return RoutingDecision(
            route="external",
            category="structured_generation",
            confidence="medium",
            reason=f"Structured generation signals ({struct_hits} matches)",
            sensitivity_score=sensitivity,
        )

    return None  # inconclusive


# ── LLM classification stage ──────────────────────────────────────────────────

async def route_with_llm(
    query: str,
    ollama_client,
    sensitivity_score: int = 0,
) -> RoutingDecision:
    """LLM-based query classification via local Ollama (~2-5 s)."""
    prompt = CLASSIFY_PROMPT.format(query=query[:500])
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await ollama_client.chat(messages, temperature=0.0)
        category = response.strip().lower().replace(" ", "_").strip("'\"")

        if category not in CATEGORY_ROUTES:
            for known in CATEGORY_ROUTES:
                if known in category or category in known:
                    category = known
                    break
            else:
                category = "other"

        return RoutingDecision(
            route=CATEGORY_ROUTES[category],
            category=category,
            confidence="medium",
            reason=f"LLM classified as '{category}'",
            sensitivity_score=sensitivity_score,
        )
    except Exception as e:
        return RoutingDecision(
            route="local",
            category="unknown",
            confidence="medium",
            reason=f"Classification error ({e}) — fail-safe: defaulting to local",
            sensitivity_score=sensitivity_score,
        )


# ── Selective routing entry point ─────────────────────────────────────────────

async def route_query_selective(
    query: str,
    pii_detected: bool,
    ollama_client=None,
    pii_types: Optional[List[str]] = None,
    sensitivity_threshold: int = DEFAULT_FORCE_LOCAL_THRESHOLD,
) -> RoutingDecision:
    """
    Selective routing: single-path LOCAL or EXTERNAL decision.

    Two-stage:
      Stage 1 — heuristic  (PII severity + keyword patterns, <1 ms)
      Stage 2 — LLM        (Ollama classification, ~2-5 s), only if Stage 1 inconclusive
    """
    pii_types = pii_types or []
    sensitivity = compute_pii_sensitivity(pii_types)

    decision = route_heuristic(query, pii_detected, pii_types, sensitivity_threshold)
    if decision is not None:
        return decision

    if ollama_client is not None:
        return await route_with_llm(query, ollama_client, sensitivity_score=sensitivity)

    return RoutingDecision(
        route="local",
        category="unknown",
        confidence="medium",
        reason="No classifier available — fail-safe: defaulting to local",
        sensitivity_score=sensitivity,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: DUAL-PATH (PII severity only)
# ══════════════════════════════════════════════════════════════════════════════

def route_query_dual(
    pii_types: Optional[List[str]] = None,
) -> RoutingDecision:
    """
    Dual-path routing: always runs both local (3a) and external (3b) paths.

    No forced-local override — both paths always reach their respective providers.
    PII sensitivity score is computed for display purposes only.
    """
    pii_types = pii_types or []
    sensitivity = compute_pii_sensitivity(pii_types)
    type_list = ", ".join(sorted(set(pii_types))) if pii_types else "none"

    return RoutingDecision(
        route="external",
        category="dual_path",
        confidence="high",
        reason=f"Dual-path synthesis — PII sensitivity={sensitivity} ({type_list})",
        sensitivity_score=sensitivity,
        forced_local=False,
    )


# ── Backward-compatible alias (defaults to selective) ────────────────────────
async def route_query(
    query: str,
    pii_detected: bool,
    ollama_client=None,
    pii_types: Optional[List[str]] = None,
    sensitivity_threshold: int = DEFAULT_FORCE_LOCAL_THRESHOLD,
) -> RoutingDecision:
    """Backward-compatible alias for route_query_selective."""
    return await route_query_selective(
        query, pii_detected, ollama_client, pii_types, sensitivity_threshold
    )
