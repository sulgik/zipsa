"""Tests for routing logic — heuristic and fallback behavior."""
import pytest
from source.relay.router import (
    route_heuristic,
    compute_pii_sensitivity,
    RoutingDecision,
    DEFAULT_FORCE_LOCAL_THRESHOLD,
)


# ── PII severity ──────────────────────────────────────────────────────────────

def test_high_severity_forces_local():
    # ID(10) + CARD_NUMBER(9) = 19 >= threshold(12)
    # Query is Korean ("process this payment") — testing PII type sensitivity, not language
    decision = route_heuristic(
        query="결제 처리해줘",
        pii_detected=True,
        pii_types=["ID", "CARD_NUMBER"],
    )
    assert decision is not None
    assert decision.route == "local"
    assert decision.forced_local is True


def test_low_severity_does_not_force_local():
    # COMPANY(2) + PERCENT(2) = 4 < threshold(12)
    # Query is Korean ("what is this company's growth rate?") — testing low-severity pass-through
    decision = route_heuristic(
        query="이 회사 성장률이 얼마야?",
        pii_detected=False,
        pii_types=["COMPANY", "PERCENT"],
    )
    # Should not force local (may return None or external)
    if decision:
        assert decision.forced_local is False


# ── Crisis detection ──────────────────────────────────────────────────────────

def test_crisis_always_local():
    # Korean: "I want to kill myself" — crisis keyword detection test
    decision = route_heuristic(
        query="자살하고 싶어",
        pii_detected=False,
        pii_types=[],
    )
    assert decision is not None
    assert decision.route == "local"
    assert decision.category == "crisis_sensitive"


# ── Code routing ──────────────────────────────────────────────────────────────

def test_code_goes_external():
    # Mixed Korean/code: "debug this code" — testing code category routing
    decision = route_heuristic(
        query="def calculate(x):\n    return x * 2\n이 코드 디버그해줘",
        pii_detected=False,
        pii_types=[],
    )
    assert decision is not None
    assert decision.route == "external"
    assert decision.category == "code_technical"


# ── Roleplay ──────────────────────────────────────────────────────────────────

def test_roleplay_stays_local():
    # Korean: "You are a doctor. Play the patient role." — roleplay detection test
    decision = route_heuristic(
        query="너는 의사야. 환자 역할을 해줘.",
        pii_detected=False,
        pii_types=[],
    )
    assert decision is not None
    assert decision.route == "local"
    assert decision.category == "roleplay_persona"


# ── Fail-safe ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_classifier_defaults_to_local():
    """When no ollama_client is available, should default to local (fail-safe)."""
    from source.relay.router import route_query_selective
    # Korean: "explain what this is" — testing fail-safe with no ollama client
    decision = await route_query_selective(
        query="이게 뭔지 설명해줘",
        pii_detected=False,
        ollama_client=None,
        pii_types=[],
    )
    assert decision.route == "local"
    assert "fail-safe" in decision.reason
