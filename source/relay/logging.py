import json
import logging
import hashlib
import os
from datetime import datetime
from typing import Optional, Dict, Any

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

# Configure structured logger
logger = logging.getLogger("relay_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs/relay.jsonl")
logger.addHandler(file_handler)

def hash_content(content) -> str:
    """Returns SHA256 hash of the content for privacy."""
    if not content:
        return ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def log_event(
    trace_id: str,
    step_name: str,
    raw_input: Optional[str] = None,
    raw_output: Optional[str] = None,
    provider: str = "internal",
    latency_ms: float = 0.0,
    safety_flags: Dict[str, Any] = None
):
    """
    Logs an event without storing raw PII user content.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "trace_id": trace_id,
        "step_name": step_name,
        "input_hash": hash_content(raw_input) if raw_input else None,
        "output_hash": hash_content(raw_output) if raw_output else None,
        "provider": provider,
        "latency_ms": round(latency_ms, 2),
        "safety_flags": safety_flags or {}
    }
    
    # Write as JSON line
    logger.info(json.dumps(entry))
