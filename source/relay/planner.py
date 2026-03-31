"""
Formal Planning Engine for Zipsa Privacy Gateway.

Architecture (3-layer pipeline):
  1. Classifier  — deterministic tag extraction (injection, exfiltration, task type, PII sensitivity)
  2. Planner     — generates structured PlannerProposal (decision enum + reason_code + external task spec)
  3. Validator   — enforces policy invariants; can downgrade or reject the proposal

Contrast with the prior LLM-based planner (ollama.plan_execution):
  - Old:    single LLM call → {"route": "hybrid"|"local", "reformulated": str|None}
              • LLM can be instruction-followed or jailbroken (inj_llm_targeted attack)
              • Sometimes inconclusive — fallback goes external by default
  - Formal: classify → propose → validate → typed FormalPlannerDecision
              • Always conclusive, no LLM → smaller attack surface
              • Monotonic downgrade: validator can only make decisions more conservative

Policy invariants enforced by Validator:
  INV-1  decision=="hybrid" ∧ proposed_external_task==None   → reject → local_only
  INV-2  injection_risk=="high"                              → downgrade to local_only
  INV-3  exfiltration_risk==True                             → downgrade to local_only
  INV-4  crisis_risk==True                                   → downgrade to local_only
  INV-5  injection_risk=="medium" ∧ decision=="hybrid"       → allow but emit warning note
  INV-6  injection_risk=="medium" ∧ PII present ∧ hybrid     → downgrade (assisted-exfiltration risk)
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ── LRU cache for LLM classification results ──────────────────────────────────
# Key: SHA-256 of the redacted query (no PII). Value: (result_dict, timestamp).
# Avoids repeated LLM round-trips for queries with the same structural pattern.
_CLASSIFY_CACHE: OrderedDict = OrderedDict()
_CLASSIFY_CACHE_MAX = 256   # entries
_CLASSIFY_CACHE_TTL = 300   # seconds (5 min)

from source.relay.router import (
    compute_pii_sensitivity,
    CRISIS_PATTERNS,
    CODE_PATTERNS,
    ROLEPLAY_PATTERNS,
    PII_CONTENT_KEYWORDS,
    DOMAIN_KNOWLEDGE_PATTERNS,
    TEXT_REWRITE_PATTERNS,
    STRUCTURED_GEN_PATTERNS,
    DEFAULT_FORCE_LOCAL_THRESHOLD,
    RoutingDecision,
)


# ── Task types that prefer external vs local ───────────────────────────────────

EXTERNAL_PREFERRED_TASKS = frozenset({
    "domain_knowledge",
    "analysis_evaluation",
    "code_technical",
    "text_rewrite",
    "structured_generation",
    "information_request",
    "simple_instruction",
    "other",
})

LOCAL_PREFERRED_TASKS = frozenset({
    "roleplay_persona",
    "pii_dependent",
    "crisis_sensitive",
})

# Maps task_type → external task kind (for proposal schema)
TASK_KIND_MAP: Dict[str, str] = {
    "code_technical":        "code_debug",
    "domain_knowledge":      "knowledge_lookup",
    "text_rewrite":          "text_transform",
    "structured_generation": "generation",
    "analysis_evaluation":   "analysis",
    "information_request":   "knowledge_lookup",
    "simple_instruction":    "knowledge_lookup",
    "other":                 "other",
}


# ── Injection & exfiltration detection patterns ────────────────────────────────

INJECTION_PATTERNS: List[re.Pattern] = [
    # ── English: override / jailbreak ─────────────────────────────────────────
    re.compile(r'ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|constraints?|guidelines?)', re.IGNORECASE),
    re.compile(r'(forget|disregard|override)\s+(your\s+)?(instructions?|rules?|guidelines?|training|constraints?)', re.IGNORECASE),
    re.compile(r'\b(DAN|jailbreak|developer\s+mode|god\s+mode|unrestricted\s+mode|no[\s-]?filter\s+mode)\b', re.IGNORECASE),
    # Persona / mode takeover
    re.compile(r'(you\s+are\s+now|from\s+now\s+on\s+you\s+are|act\s+as\s+if\s+you\s+have\s+no)', re.IGNORECASE),
    # ── Context-boundary injection (LLM template tokens — each a separate hit) ─
    re.compile(r'</?(?:system|assistant|user|instruction)>', re.IGNORECASE),
    re.compile(r'\[INST\]', re.IGNORECASE),
    re.compile(r'\[/INST\]', re.IGNORECASE),
    re.compile(r'<<SYS>>|<</SYS>>', re.IGNORECASE),
    # ── Privilege escalation ──────────────────────────────────────────────────
    re.compile(r'(admin\s+mode|root\s+access|bypass\s+(safety|filter|restrict|security))', re.IGNORECASE),
    re.compile(r'(reveal|leak|expose|print)\s+(system|your|the)\s+(prompt|instructions?|rules?)', re.IGNORECASE),
    # ── Korean injection patterns (particle-agnostic) ─────────────────────────
    re.compile(r'이전.{0,15}(지시|지침|규칙|보안|설정).{0,10}무시', re.IGNORECASE),
    re.compile(r'모든.{0,10}(보안|지침|규칙|제약|제한).{0,10}무시', re.IGNORECASE),
    re.compile(r'시스템\s*(프롬프트|지시|규칙)', re.IGNORECASE),
    re.compile(r'(안전|보안).{0,10}(필터|제약|규칙|지침).{0,10}(우회|해제|무시|없는|없이)', re.IGNORECASE),
    re.compile(r'(필터|제약|제한).{0,5}(없는|없이|해제|우회)', re.IGNORECASE),
    re.compile(r'(자유로운|제한\s*없는|무제한|무필터).{0,10}AI', re.IGNORECASE),
    re.compile(r'모든\s*(제약|제한|규칙)\s*(해제|무시|없이)', re.IGNORECASE),
]

EXFILTRATION_PATTERNS: List[re.Pattern] = [
    # ── English ───────────────────────────────────────────────────────────────
    re.compile(r'(send|email|post|upload|forward)\b.{0,40}(user\s+data|private\s+data|all\s+(?:the\s+)?data|personal\s+(?:info|data)|pii\s+data)', re.IGNORECASE),
    re.compile(r'(send|forward|share|transmit)\b.{0,25}(conversation|context|session|message\s+history)', re.IGNORECASE),
    re.compile(r'(print|output|display|return)\s+(the|all|my|user|binding)\s+(pii|data|map|context|secrets?)', re.IGNORECASE),
    re.compile(r'(what\s+is|tell\s+me|show\s+me)\s+(my|the|user|their)\s+(real|actual|original)\s+(name|id|phone|email)', re.IGNORECASE),
    re.compile(r'\b(binding[\s_]?map|placeholder[\s_]?map|original[\s_]?value)\b', re.IGNORECASE),
    re.compile(r'(leak|expose|exfiltrate|steal)\s+.{0,20}(user|private|pii|personal|original)\s+(data|info|values?)', re.IGNORECASE),
    # ── Korean exfiltration ───────────────────────────────────────────────────
    re.compile(r'(실제\s*값|원래\s*이름|바인딩\s*맵|개인정보\s*전달|원본\s*데이터)', re.IGNORECASE),
    re.compile(r'(전송|유출|노출|외부.{0,5}전달).{0,20}(개인정보|데이터|정보)', re.IGNORECASE),
]

# ── Natural language PII signals (sentence-level, no structured format) ────────
#
# Coverage:
#   KO-1  Salary/compensation with amount      "연봉 1억 2천", "급여가 350만원"
#   KO-2  Korean name + 씨/님 honorific        "최철수씨의", "김영희님이"
#   KO-3  Korean name + subject particle + sensitive verb  "최철수가 치료 중"
#   KO-4  First-person + sensitive noun        "저의 병명", "제 주민번호"
#   KO-5  Possessive name + sensitive topic    "최철수의 진단서", "김 팀장의 연봉"
#   KO-6  Korean address (district + street)   "강남구 테헤란로 123"
#   KO-7  Resident registration number hint    "주민번호가" (without the number)
#   EN-1  Salary with amount                   "salary is $80k", "$120k annual"
#   EN-2  Personal medical attribution         "my diagnosis is", "her prescription"

NL_PII_PATTERNS: List[re.Pattern] = [
    # KO-1: salary/compensation + numeric amount
    re.compile(r'(연봉|급여|월급|임금|연봉액|급여액)\s*[이가은는을를]?\s*\d', re.IGNORECASE),
    re.compile(r'\d+\s*(만원|억원|억|천만|백만|만\s*원)\s*(연봉|급여|월급|임금)', re.IGNORECASE),
    # KO-2: Korean name (2-4 chars) + 씨/님 honorific (strong personal name signal)
    re.compile(r'[가-힣]{2,4}\s*(?:씨|님)\s*(?:의|가|은|는|이|께서|에게|한테)', re.IGNORECASE),
    # KO-3: Korean name-like sequence + subject/topic particle + sensitive context within 40 chars
    re.compile(
        r'[가-힣]{2,4}\s*(?:이|가|은|는)\s*.{0,40}'
        r'(?:치료|진단|처방|수술|입원|퇴원|연봉|급여|재직|근무|의료|처방전|병력|검사결과)',
        re.IGNORECASE
    ),
    # KO-4: first-person pronoun + sensitive noun (personal info disclosure)
    re.compile(
        r'(?:저의|저는|저가|제|나의|내|나는|나의)\s*'
        r'(?:병명|진단|병력|처방|주민|연봉|급여|주소|계좌|비밀번호|개인정보|연락처)',
        re.IGNORECASE
    ),
    # KO-5: Korean name possessive + sensitive topic
    re.compile(
        r'[가-힣]{2,4}의\s*'
        r'(?:병력|병명|진단|처방전?|연봉|급여|주소|계좌|비밀번호|주민|이력서|신상)',
        re.IGNORECASE
    ),
    # KO-6: Korean address pattern (district + road/street)
    re.compile(r'[가-힣]+\s*(?:시|구|동|읍|면)\s+[가-힣0-9]+\s*(?:로|길|대로|번길)', re.IGNORECASE),
    # KO-7: RRN/ID mention without the number (user might be about to share it)
    re.compile(r'주민\s*(?:등록\s*)?(?:번호|번)', re.IGNORECASE),
    # EN-1: salary with amount
    re.compile(r'(?:salary|income|compensation|pay)\s*(?:is|of|was|at|:)?\s*\$?\d', re.IGNORECASE),
    re.compile(r'\$\d+\s*[kKmM]?\s*(?:salary|per\s+year|annual|annually)', re.IGNORECASE),
    # EN-2: personal medical attribution
    re.compile(
        r'(?:my|their|his|her|patient\'?s?)\s+'
        r'(?:diagnosis|prescription|medical\s+record|health\s+record|treatment\s+plan)',
        re.IGNORECASE
    ),
]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ClassifierTags:
    """Layer 1 output: typed tags describing the query's risk and task profile."""
    task_type: str = "unknown"
    task_confidence: str = "medium"     # "high" | "medium"
    task_kind: str = "other"            # external task schema kind

    pii_sensitivity: int = 0
    pii_types: List[str] = field(default_factory=list)

    injection_risk: str = "low"         # "low" | "medium" | "high"
    injection_hit_count: int = 0
    exfiltration_risk: bool = False
    crisis_risk: bool = False

    # Natural language PII signals (sentence-level, independent of regex scan)
    nl_pii_detected: bool = False       # True if NL_PII_PATTERNS matched

    # Semantic harm categories (derived from all signals above)
    # Values: PERSONAL_PRIVACY | ORGANIZATIONAL_CONFIDENTIAL | SECURITY_RISK |
    #         REGULATORY_EXPOSURE | ADVERSARIAL_INTENT
    harm_categories: List[str] = field(default_factory=list)


@dataclass
class PlannerProposal:
    """Layer 2 output: structured decision proposal (not yet validated)."""
    decision: str = "local_only"        # "local_only" | "hybrid"
    reason_code: str = "privacy_risk"   # controlled vocabulary (see below)
    proposed_external_task: Optional[Dict[str, Any]] = None

    # Valid reason_codes:
    #   needs_external_knowledge  — task benefits from external LLM
    #   privacy_risk              — PII severity or sanitization failure
    #   sufficient_local_context  — local model can handle without external help
    #   security_override         — injection / exfiltration / crisis detected
    #   crisis_content            — crisis-sensitive content


@dataclass
class FormalPlannerDecision:
    """
    Layer 3 (final) output.
    route field maps to orchestrator's execution_path:
      "local"    → execution_path = "local"
      "external" → execution_path = "hybrid"
    """
    # Routing-compatible fields
    route: str = "local"                # "local" | "external"
    category: str = "unknown"
    confidence: str = "medium"
    reason: str = ""
    sensitivity_score: int = 0
    forced_local: bool = False

    # Formal planning fields
    decision: str = "local_only"        # "local_only" | "hybrid"
    reason_code: str = "privacy_risk"
    proposed_external_task: Optional[Dict[str, Any]] = None
    classifier_tags: ClassifierTags = field(default_factory=ClassifierTags)
    validation_passed: bool = True
    validation_notes: List[str] = field(default_factory=list)

    def to_routing_decision(self) -> RoutingDecision:
        """Convert to RoutingDecision (for compatibility with router-based code)."""
        return RoutingDecision(
            route=self.route,
            category=self.category,
            confidence=self.confidence,
            reason=self.reason,
            sensitivity_score=self.sensitivity_score,
            forced_local=self.forced_local,
        )


# ── Layer 1: Classifier ────────────────────────────────────────────────────────

def classify(
    query: str,
    pii_types: List[str],
    pii_detected: bool,
    sensitivity_threshold: int = DEFAULT_FORCE_LOCAL_THRESHOLD,
) -> ClassifierTags:
    """
    Deterministic tag extraction — no LLM calls.

    Output tags:
      - task_type / task_kind: coarse task category
      - pii_sensitivity: summed severity score
      - injection_risk: low / medium / high (pattern count)
      - exfiltration_risk: bool
      - crisis_risk: bool
    """
    tags = ClassifierTags(
        pii_types=list(pii_types),
        pii_sensitivity=compute_pii_sensitivity(pii_types),
    )

    # Injection risk
    hits = sum(1 for p in INJECTION_PATTERNS if p.search(query))
    tags.injection_hit_count = hits
    if hits >= 2:
        tags.injection_risk = "high"
    elif hits == 1:
        tags.injection_risk = "medium"
    else:
        tags.injection_risk = "low"

    # Exfiltration risk
    tags.exfiltration_risk = any(p.search(query) for p in EXFILTRATION_PATTERNS)

    # Crisis risk
    tags.crisis_risk = any(p.search(query) for p in CRISIS_PATTERNS)

    # Natural language PII signal (sentence-level, independent of regex scan)
    tags.nl_pii_detected = any(p.search(query) for p in NL_PII_PATTERNS)

    # Task type — priority order: security checks first, then task signals
    if tags.crisis_risk:
        tags.task_type = "crisis_sensitive"
        tags.task_confidence = "high"
        tags.task_kind = "other"

    elif tags.pii_sensitivity >= sensitivity_threshold and pii_types:
        tags.task_type = "pii_dependent"
        tags.task_confidence = "high"
        tags.task_kind = "other"

    elif sum(1 for p in CODE_PATTERNS if p.search(query)) >= 2:
        tags.task_type = "code_technical"
        tags.task_confidence = "high"
        tags.task_kind = TASK_KIND_MAP["code_technical"]

    elif any(p.search(query) for p in ROLEPLAY_PATTERNS):
        tags.task_type = "roleplay_persona"
        tags.task_confidence = "high"
        tags.task_kind = "other"

    elif (pii_detected or tags.nl_pii_detected) and any(
        kw.lower() in query.lower() for kw in PII_CONTENT_KEYWORDS
    ):
        # PII present + sensitive keyword. If domain knowledge signal is also present,
        # let domain_knowledge win — PII is incidental context, reformulator strips it.
        domain_hits = sum(1 for p in DOMAIN_KNOWLEDGE_PATTERNS if p.search(query))
        if domain_hits >= 1:
            tags.task_type = "domain_knowledge"
            tags.task_confidence = "medium"
            tags.task_kind = TASK_KIND_MAP["domain_knowledge"]
        else:
            tags.task_type = "pii_dependent"
            tags.task_confidence = "medium"
            tags.task_kind = "other"

    elif sum(1 for p in DOMAIN_KNOWLEDGE_PATTERNS if p.search(query)) >= 1:
        tags.task_type = "domain_knowledge"
        tags.task_confidence = "medium"
        tags.task_kind = TASK_KIND_MAP["domain_knowledge"]

    elif tags.nl_pii_detected:
        tags.task_type = "pii_dependent"
        tags.task_confidence = "medium"
        tags.task_kind = "other"

    elif any(p.search(query) for p in TEXT_REWRITE_PATTERNS):
        tags.task_type = "text_rewrite"
        tags.task_confidence = "medium"
        tags.task_kind = TASK_KIND_MAP["text_rewrite"]

    elif sum(1 for p in STRUCTURED_GEN_PATTERNS if p.search(query)) >= 1:
        tags.task_type = "structured_generation"
        tags.task_confidence = "medium"
        tags.task_kind = TASK_KIND_MAP["structured_generation"]

    else:
        tags.task_type = "information_request"
        tags.task_confidence = "medium"
        tags.task_kind = TASK_KIND_MAP["information_request"]

    tags.harm_categories = _derive_harm_categories(tags)
    return tags


_PERSONAL_PII = frozenset({
    "NAME", "EMAIL", "PHONE", "SSN", "RRN", "PASSPORT",
    "CARD_NUMBER", "CREDIT_CARD", "PERSON", "DOB",
})
_ORGANIZATIONAL_PII = frozenset({"ORG", "COMPANY", "BIZ_REG"})
_SECURITY_PII = frozenset({"AWS_KEY_ID", "GITHUB_TOKEN", "API_KEY", "PASSWORD", "TOKEN"})
_FINANCIAL_PII = frozenset({"AMOUNT", "CARD_NUMBER", "ACCOUNT_ID", "BANK"})


def _derive_harm_categories(tags: ClassifierTags) -> List[str]:
    """Map low-level classifier signals → semantic harm categories (rubric v1)."""
    cats: List[str] = []
    pii_set = set(tags.pii_types)

    if pii_set & _PERSONAL_PII or tags.crisis_risk or tags.nl_pii_detected:
        cats.append("PERSONAL_PRIVACY")
    if pii_set & _ORGANIZATIONAL_PII:
        cats.append("ORGANIZATIONAL_CONFIDENTIAL")
    if pii_set & _SECURITY_PII:
        cats.append("SECURITY_RISK")
    if pii_set & _FINANCIAL_PII:
        cats.append("REGULATORY_EXPOSURE")
    if tags.injection_risk == "high" or tags.exfiltration_risk:
        cats.append("ADVERSARIAL_INTENT")

    return cats


# ── Layer 2: Planner ───────────────────────────────────────────────────────────

def propose(tags: ClassifierTags, sanitize_safe: bool) -> PlannerProposal:
    """
    Generate a structured planning proposal from classifier tags.

    Decision logic (ordered by priority):
      1. Security risks (injection/exfiltration/crisis) → local_only / security_override
      2. Sanitization failed                            → local_only / privacy_risk
      3. Local-preferred task type (pii_dependent, etc.)→ local_only / sufficient_local_context
      4. External-preferred task type                   → hybrid / needs_external_knowledge
      5. Default                                        → local_only / sufficient_local_context

    PII severity is NOT checked here — enforced downstream via post-reformulation PII scan.
    """
    # 1. Security override (highest priority)
    if tags.injection_risk == "high" or tags.exfiltration_risk:
        return PlannerProposal(
            decision="local_only",
            reason_code="security_override",
            proposed_external_task=None,
        )

    if tags.crisis_risk:
        return PlannerProposal(
            decision="local_only",
            reason_code="crisis_content",
            proposed_external_task=None,
        )

    # 2. Sanitization failure
    if not sanitize_safe:
        return PlannerProposal(
            decision="local_only",
            reason_code="privacy_risk",
            proposed_external_task=None,
        )

    # 3. Local-preferred tasks
    if tags.task_type in LOCAL_PREFERRED_TASKS:
        return PlannerProposal(
            decision="local_only",
            reason_code="sufficient_local_context",
            proposed_external_task=None,
        )

    # 4. External-preferred tasks → hybrid
    if tags.task_type in EXTERNAL_PREFERRED_TASKS:
        return PlannerProposal(
            decision="hybrid",
            reason_code="needs_external_knowledge",
            proposed_external_task={
                "kind": tags.task_kind,
                "task_type": tags.task_type,
                "constraints": ["no_pii", "no_hidden_context", "no_system_prompt"],
            },
        )

    # 5. Default — safe choice
    return PlannerProposal(
        decision="local_only",
        reason_code="sufficient_local_context",
        proposed_external_task=None,
    )


# ── Layer 3: Validator ─────────────────────────────────────────────────────────

def validate(
    proposal: PlannerProposal,
    tags: ClassifierTags,
) -> tuple[PlannerProposal, bool, List[str]]:
    """
    Enforce policy invariants. Returns (validated_proposal, passed, notes).

    Monotonic downgrade principle: validator can only make decisions more
    conservative (hybrid → local_only), never less conservative.

    INV-1: hybrid without task spec  → downgrade to local_only
    INV-2: high injection risk       → downgrade to local_only
    INV-3: exfiltration risk         → downgrade to local_only
    INV-4: crisis risk               → downgrade to local_only
    INV-5: medium injection + hybrid → allow, emit warning
    INV-6: medium injection + PII    → downgrade (assisted-exfiltration risk)
    """
    notes: List[str] = []
    passed = True
    p = proposal  # mutable reference

    # INV-1
    if p.decision == "hybrid" and p.proposed_external_task is None:
        p = PlannerProposal(decision="local_only", reason_code="security_override")
        notes.append("INV-1: hybrid proposed without external task spec — downgraded")
        passed = False

    # INV-2
    if tags.injection_risk == "high":
        p = PlannerProposal(decision="local_only", reason_code="security_override")
        notes.append(f"INV-2: injection_risk=high ({tags.injection_hit_count} patterns) — downgraded")
        passed = False

    # INV-3
    if tags.exfiltration_risk:
        p = PlannerProposal(decision="local_only", reason_code="security_override")
        notes.append("INV-3: exfiltration_risk detected — downgraded")
        passed = False

    # INV-4
    if tags.crisis_risk:
        p = PlannerProposal(decision="local_only", reason_code="crisis_content")
        notes.append("INV-4: crisis_risk — downgraded")
        passed = False

    # INV-5 (non-blocking, warning only)
    if tags.injection_risk == "medium" and p.decision == "hybrid" and not tags.pii_types:
        notes.append(f"INV-5: medium injection risk ({tags.injection_hit_count} pattern) — monitoring")

    # INV-6: medium injection + any PII present → potential assisted-exfiltration attack
    if tags.injection_risk == "medium" and tags.pii_types and p.decision == "hybrid":
        p = PlannerProposal(decision="local_only", reason_code="security_override")
        notes.append(
            f"INV-6: medium injection ({tags.injection_hit_count} pattern) + PII present "
            f"({', '.join(tags.pii_types[:3])}) — assisted-exfiltration risk, downgraded"
        )
        passed = False

    return p, passed, notes


# ── LLM-based classification ───────────────────────────────────────────────────

_LLM_CLASSIFY_PROMPT = """You are a query router for a local privacy gateway. Classify the query and decide whether external AI knowledge is needed.

VALID task_type values (pick exactly one):
  domain_knowledge      medical, legal, financial, scientific expertise questions
  analysis_evaluation   comparing options, evaluating tradeoffs, strategic analysis
  code_technical        code, debugging, technical implementation
  text_rewrite          editing, translating, rephrasing existing text
  structured_generation generating documents, forms, reports, tables
  information_request   factual lookup, general knowledge
  simple_instruction    short, clear request needing no expertise
  pii_dependent         the PII itself IS the answer (e.g. "look up John's SSN")
  roleplay_persona      persona, character, or role-play request
  crisis_sensitive      self-harm, abuse, emergency situations
  other                 none of the above

DECISION rules:
  hybrid     — task type benefits from external expertise AND an anonymized version
               of the question can meaningfully capture the knowledge need
  local_only — the answer IS the private data, the query is identity-bound, or
               external knowledge adds nothing beyond what a local model can do

Output ONLY a JSON object, no explanation:
{"task_type": "<value>", "decision": "hybrid"|"local_only", "reason_code": "needs_external_knowledge"|"sufficient_local_context"|"privacy_risk"}

EXAMPLES:
Query: "Jane Smith (SSN 123-45-6789) has T2DM, HbA1c 8.4% on metformin. Next treatment?"
Output: {"task_type": "domain_knowledge", "decision": "hybrid", "reason_code": "needs_external_knowledge"}

Query: "Look up John's SSN 123-45-6789 and get his insurance policy number."
Output: {"task_type": "pii_dependent", "decision": "local_only", "reason_code": "sufficient_local_context"}

Query: "My name is Sarah, email sarah@acme.com. Write a Python CSV parser for employee records."
Output: {"task_type": "code_technical", "decision": "hybrid", "reason_code": "needs_external_knowledge"}

Query: "My salary is $120k. Is that fair for a senior engineer in NYC?"
Output: {"task_type": "information_request", "decision": "hybrid", "reason_code": "needs_external_knowledge"}

Now classify:"""

_VALID_TASK_TYPES = frozenset({
    "domain_knowledge", "analysis_evaluation", "code_technical", "text_rewrite",
    "structured_generation", "information_request", "simple_instruction",
    "pii_dependent", "roleplay_persona", "crisis_sensitive", "other",
})
_VALID_DECISIONS = frozenset({"hybrid", "local_only"})
_VALID_REASON_CODES = frozenset({
    "needs_external_knowledge", "sufficient_local_context", "privacy_risk",
})


_CREDENTIAL_PATTERNS: List[re.Pattern] = [
    re.compile(r'(password|secret[\s_]?key|access[\s_]?key|api[\s_]?key|auth[\s_]?key|private[\s_]?key|client[\s_]?secret)\s*[:=]\s*\S+', re.IGNORECASE),
    re.compile(r'\b(sk-[A-Za-z0-9\-_]{8,}|AKIA[A-Z0-9]{10,}|vs_tok_\S+)\b'),
]


def _has_credentials(query: str) -> bool:
    """Return True if query contains raw credential values."""
    return any(p.search(query) for p in _CREDENTIAL_PATTERNS)


def _redact_for_classifier(query: str) -> str:
    """
    Redact credential values before passing to LLM classifier.
    Keeps structural labels (e.g. "Password:") so the classifier understands
    the query type, but removes actual values that trigger model safety refusals.
    """
    # Label: value patterns (keep label, redact value)
    label_value = re.compile(
        r'((?:password|secret[\s_]?key|access[\s_]?key|api[\s_]?key|token|auth[\s_]?key'
        r'|private[\s_]?key|client[\s_]?secret|passphrase|credential)\s*[:=]\s*)\S+',
        re.IGNORECASE
    )
    query = label_value.sub(r'\1[REDACTED]', query)

    # Standalone credential-like tokens (sk-..., AKIA..., vs_tok_..., long hex/base64)
    standalone = re.compile(
        r'\b(sk-[A-Za-z0-9\-_]{8,}|AKIA[A-Z0-9]{16,}|vs_tok_\S+|[A-Za-z0-9/+]{40,}=*)\b'
    )
    query = standalone.sub('[REDACTED]', query)

    return query


async def _classify_llm(query: str, ollama_client) -> Optional[Dict[str, str]]:
    """
    Call local LLM to classify task_type and propose decision.
    Returns {"task_type", "decision", "reason_code"} or None on failure.

    Security fields (injection_risk, exfiltration_risk, crisis_risk) are NOT
    delegated to the LLM — they are always computed deterministically in classify().
    Results are LRU-cached by redacted-query fingerprint to skip redundant LLM calls.
    """
    import json as _json
    import re as _re

    redacted = _redact_for_classifier(query)

    # ── Cache lookup ──────────────────────────────────────────────────────────
    cache_key = hashlib.sha256(redacted.encode()).hexdigest()
    if cache_key in _CLASSIFY_CACHE:
        result, ts = _CLASSIFY_CACHE[cache_key]
        if time.monotonic() - ts < _CLASSIFY_CACHE_TTL:
            _CLASSIFY_CACHE.move_to_end(cache_key)   # LRU bump
            print(f"[Stage 1] LLM classifier cache HIT ({cache_key[:8]})")
            return result
        else:
            del _CLASSIFY_CACHE[cache_key]

    messages = [
        {"role": "system", "content": _LLM_CLASSIFY_PROMPT.strip()},
        {"role": "user", "content": redacted},
    ]
    # max_tokens=80: JSON classifier output is always short ({"task_type":..., ...})
    raw = await ollama_client.chat(messages, temperature=0.0, max_tokens=80)
    if not raw:
        return None
    try:
        clean = _re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("no JSON object")
        result = _json.loads(clean[start:end])
        task_type   = result.get("task_type", "")
        decision    = result.get("decision", "")
        reason_code = result.get("reason_code", "")
        # Normalize common LLM shorthand to valid task types
        _TASK_ALIASES = {
            "legal": "domain_knowledge",
            "medical": "domain_knowledge",
            "financial": "domain_knowledge",
            "scientific": "domain_knowledge",
            "technical": "code_technical",
            "rewrite": "text_rewrite",
            "generation": "structured_generation",
            "instruction": "simple_instruction",
        }
        task_type = _TASK_ALIASES.get(task_type, task_type)
        if task_type not in _VALID_TASK_TYPES:
            raise ValueError(f"unknown task_type: {task_type!r}")
        if decision not in _VALID_DECISIONS:
            raise ValueError(f"unknown decision: {decision!r}")
        if reason_code not in _VALID_REASON_CODES:
            reason_code = "needs_external_knowledge" if decision == "hybrid" else "sufficient_local_context"
        result = {"task_type": task_type, "decision": decision, "reason_code": reason_code}

        # ── Cache store ───────────────────────────────────────────────────────
        _CLASSIFY_CACHE[cache_key] = (result, time.monotonic())
        _CLASSIFY_CACHE.move_to_end(cache_key)
        if len(_CLASSIFY_CACHE) > _CLASSIFY_CACHE_MAX:
            _CLASSIFY_CACHE.popitem(last=False)   # evict oldest

        return result
    except Exception as e:
        print(f"[LLM Classify Error] {e} | raw={raw[:120]}")
        return None


# ── Entry point ────────────────────────────────────────────────────────────────

def plan(
    query: str,
    pii_types: List[str],
    pii_detected: bool,
    sanitize_safe: bool = True,
    sensitivity_threshold: int = DEFAULT_FORCE_LOCAL_THRESHOLD,
) -> FormalPlannerDecision:
    """
    Full 3-layer planning pipeline: classify → propose → validate → finalize.

    Returns FormalPlannerDecision with:
      decision == "hybrid"     → orchestrator calls reformulate() then runs external
      decision == "local_only" → orchestrator skips external call entirely
    """
    tags = classify(query, pii_types, pii_detected, sensitivity_threshold)
    proposal = propose(tags, sanitize_safe)
    validated, validation_passed, validation_notes = validate(proposal, tags)

    if validated.decision == "hybrid":
        route = "external"
        forced_local = False
    else:
        route = "local"
        forced_local = (
            tags.injection_risk == "high"
            or tags.exfiltration_risk
            or tags.crisis_risk
            or (tags.pii_sensitivity >= sensitivity_threshold and bool(tags.pii_types))
            or not sanitize_safe
        )

    reason_parts = [f"[Formal/{validated.reason_code}] {tags.task_type} → {validated.decision}"]
    if validation_notes:
        reason_parts.append("| " + "; ".join(validation_notes))

    return FormalPlannerDecision(
        route=route,
        category=tags.task_type,
        confidence=tags.task_confidence,
        reason=" ".join(reason_parts),
        sensitivity_score=tags.pii_sensitivity,
        forced_local=forced_local,
        decision=validated.decision,
        reason_code=validated.reason_code,
        proposed_external_task=validated.proposed_external_task,
        classifier_tags=tags,
        validation_passed=validation_passed,
        validation_notes=validation_notes,
    )


async def plan_async(
    query: str,
    pii_types: List[str],
    pii_detected: bool,
    ollama_client,
    sanitize_safe: bool = True,
    sensitivity_threshold: int = DEFAULT_FORCE_LOCAL_THRESHOLD,
) -> FormalPlannerDecision:
    """
    Hybrid planning pipeline: LLM classification + deterministic validation.

    Layer 1a (LLM): classify task_type and propose decision (intelligent, context-aware)
    Layer 1b (deterministic): compute security fields — injection_risk, exfiltration_risk,
                               crisis_risk, nl_pii_detected, pii_sensitivity
                               These are NEVER delegated to the LLM (attack surface).
    Layer 2 (deterministic): validate — enforce invariants, monotonic downgrade only
    Falls back to plan() if the LLM call fails.
    """
    # Layer 1b: deterministic security fields (always — regardless of LLM)
    tags = classify(query, pii_types, pii_detected, sensitivity_threshold)

    # Layer 1a: LLM classification (task_type + decision proposal)
    # If query contains raw credentials, pass a redacted version to the classifier
    # to avoid model safety refusals — the classifier only needs task structure, not values.
    llm_result = await _classify_llm(query, ollama_client)

    if llm_result:
        # Override task_type and initial decision with LLM output
        task_type   = llm_result["task_type"]
        decision    = llm_result["decision"]
        reason_code = llm_result["reason_code"]
        task_kind   = TASK_KIND_MAP.get(task_type, "other")

        # Rebuild tags with LLM task_type (security fields stay deterministic)
        tags.task_type       = task_type
        tags.task_kind       = task_kind
        tags.task_confidence = "high"  # LLM classification

        # Build proposal from LLM decision
        if decision == "hybrid":
            proposal = PlannerProposal(
                decision="hybrid",
                reason_code=reason_code,
                proposed_external_task={
                    "kind": task_kind,
                    "task_type": task_type,
                    "constraints": ["no_pii", "no_hidden_context", "no_system_prompt"],
                },
            )
        else:
            proposal = PlannerProposal(
                decision="local_only",
                reason_code=reason_code,
                proposed_external_task=None,
            )

        print(f"[Stage 1] LLM classifier → task={task_type}, decision={decision}")
    else:
        # LLM failed — fall back to deterministic propose()
        print("[Stage 1] LLM classifier failed — falling back to deterministic")
        proposal = propose(tags, sanitize_safe)

    # Layer 2: deterministic validation (invariants enforced on deterministic security fields)
    validated, validation_passed, validation_notes = validate(proposal, tags)

    if validated.decision == "hybrid":
        route = "external"
        forced_local = False
    else:
        route = "local"
        forced_local = (
            tags.injection_risk == "high"
            or tags.exfiltration_risk
            or tags.crisis_risk
            or (tags.pii_sensitivity >= sensitivity_threshold and bool(tags.pii_types))
            or not sanitize_safe
        )

    prefix = "LLM" if llm_result else "Formal"
    reason_parts = [f"[{prefix}/{validated.reason_code}] {tags.task_type} → {validated.decision}"]
    if validation_notes:
        reason_parts.append("| " + "; ".join(validation_notes))

    return FormalPlannerDecision(
        route=route,
        category=tags.task_type,
        confidence=tags.task_confidence,
        reason=" ".join(reason_parts),
        sensitivity_score=tags.pii_sensitivity,
        forced_local=forced_local,
        decision=validated.decision,
        reason_code=validated.reason_code,
        proposed_external_task=validated.proposed_external_task,
        classifier_tags=tags,
        validation_passed=validation_passed,
        validation_notes=validation_notes,
    )
