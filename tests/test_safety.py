"""Tests for SafetyGuard PII detection and redaction."""
import pytest
from source.relay.safety import SafetyGuard


@pytest.fixture
def guard():
    return SafetyGuard()


# ── Korean PII ────────────────────────────────────────────────────────────────

def test_korean_rrn_with_dash(guard):
    text = "주민등록번호는 800101-1234567입니다."
    redacted, found, _, bmap = guard.scan_and_redact(text)
    assert found
    assert "[ID]" in redacted
    assert "800101-1234567" not in redacted


def test_korean_rrn_no_dash(guard):
    text = "RRN: 8001011234567"
    redacted, found, _, bmap = guard.scan_and_redact(text)
    assert found
    assert "8001011234567" not in redacted


def test_korean_mobile(guard):
    text = "연락처: 010-1234-5678"
    redacted, found, _, _ = guard.scan_and_redact(text)
    assert found
    assert "010-1234-5678" not in redacted


def test_korean_landline(guard):
    text = "대표번호: 02-1234-5678"
    redacted, found, _, _ = guard.scan_and_redact(text)
    assert found
    assert "02-1234-5678" not in redacted


def test_korean_bank_account(guard):
    text = "계좌번호: 110-123-456789"
    redacted, found, _, _ = guard.scan_and_redact(text)
    assert found
    assert "110-123-456789" not in redacted


def test_korean_biz_reg(guard):
    text = "사업자등록번호: 123-45-67890"
    redacted, found, _, _ = guard.scan_and_redact(text)
    assert found
    assert "123-45-67890" not in redacted


# ── International PII ─────────────────────────────────────────────────────────

def test_email(guard):
    text = "이메일: user@example.com"
    redacted, found, _, _ = guard.scan_and_redact(text)
    assert found
    assert "user@example.com" not in redacted


def test_credit_card(guard):
    text = "카드번호: 4111-1111-1111-1111"
    redacted, found, _, _ = guard.scan_and_redact(text)
    assert found
    assert "4111-1111-1111-1111" not in redacted


def test_github_token(guard):
    text = "token: ghp_" + "a" * 36
    redacted, found, _, _ = guard.scan_and_redact(text)
    assert found


def test_no_pii(guard):
    text = "오늘 날씨가 좋네요. React 최적화 방법이 궁금합니다."
    _, found, _, bmap = guard.scan_and_redact(text)
    assert not found
    assert len(bmap) == 0


# ── Reversibility ─────────────────────────────────────────────────────────────

def test_binding_map_reversible(guard):
    text = "환자 SSN: 800101-1234567, 연락처: 010-9999-8888"
    redacted, found, _, bmap = guard.scan_and_redact(text)
    assert found
    # Restore original
    restored = redacted
    for placeholder, original in bmap.items():
        restored = restored.replace(placeholder, original)
    assert "800101-1234567" in restored
    assert "010-9999-8888" in restored


# ── Leakage check ─────────────────────────────────────────────────────────────

def test_leakage_detected(guard):
    original = "SSN: 800101-1234567"
    sanitized = "환자의 SSN은 800101-1234567입니다."  # leaked!
    leaked, items = guard.check_leakage(original, sanitized)
    assert leaked
    assert len(items) > 0


def test_no_leakage(guard):
    original = "SSN: 800101-1234567"
    sanitized = "환자의 SSN은 [ID]입니다."  # properly redacted
    leaked, items = guard.check_leakage(original, sanitized)
    assert not leaked
