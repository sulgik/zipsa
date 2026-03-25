#!/usr/bin/env python3
"""
Zipsa Setup Wizard
Interactive CLI to configure .env for first-time users.
"""

import os
import sys
import urllib.request
import json
import re

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"
DIM = "\033[2m"


def print_header():
    print(f"""
{BOLD}🔒 Zipsa Setup Wizard{RESET}
{DIM}─────────────────────────────────────{RESET}
""")


def ask(prompt, default=None):
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    try:
        val = input(full_prompt).strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print(f"\n{YELLOW}Setup cancelled.{RESET}")
        sys.exit(0)


def ask_choice(prompt, choices):
    """Present numbered choices, return selected index (0-based)."""
    print(prompt)
    for i, choice in enumerate(choices, 1):
        print(f"  [{i}] {choice}")
    while True:
        try:
            raw = input("  > ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return idx
            print(f"  Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print(f"  Please enter a number between 1 and {len(choices)}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Setup cancelled.{RESET}")
            sys.exit(0)


def check_ollama(host):
    """Check if Ollama is reachable and return list of model names."""
    try:
        req = urllib.request.Request(
            f"{host}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return None


def mask_token(token):
    if token and len(token) > 10:
        return token[:15] + "..." + token[-4:]
    return token or ""


def read_env(path=".env"):
    """Read existing .env preserving all lines."""
    if not os.path.exists(path):
        if os.path.exists(".env.example"):
            with open(".env.example") as f:
                return f.readlines()
        return []
    with open(path) as f:
        return f.readlines()


def set_env_key(lines, key, value):
    """Set a key in env lines, replacing if exists, appending if not."""
    pattern = re.compile(rf"^{re.escape(key)}\s*=")
    new_line = f"{key}={value}\n"
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = new_line
            return lines
    lines.append(new_line)
    return lines


def write_env(lines, path=".env"):
    with open(path, "w") as f:
        f.writelines(lines)


def main():
    print_header()

    env_lines = read_env()
    updates = {}

    # ── Step 1: Provider ──────────────────────────────────────────────────────
    print(f"{BOLD}Step 1/3: External LLM Provider{RESET}")
    print("Which provider do you want for cloud queries?\n")

    providers = [
        ("Claude (Anthropic)", "claude", "ANTHROPIC_API_KEY"),
        ("OpenAI (GPT)", "openai", "OPENAI_API_KEY"),
        ("Gemini (Google)", "gemini", "GEMINI_API_KEY"),
    ]

    idx = ask_choice("", [p[0] for p in providers])
    provider_name, provider_id, key_env = providers[idx]
    updates["EXTERNAL_PROVIDER"] = provider_id
    print(f"  {GREEN}✓ {provider_name}{RESET}\n")

    # ── Step 2: Auth ──────────────────────────────────────────────────────────
    print(f"{BOLD}Step 2/3: Authentication{RESET}")

    if provider_id == "claude":
        print("How do you want to authenticate?\n")
        auth_idx = ask_choice("", [
            "API Key  (pay-per-use)  — get from console.anthropic.com",
            "Claude Max Subscription (no extra charges)",
        ])

        if auth_idx == 1:
            print(f"""
  {CYAN}Claude Max Setup:{RESET}
  Run this command in your terminal to generate a setup token:

    {BOLD}claude setup-token{RESET}

  Then paste the token below.
""")
        else:
            print(f"\n  Get your API key at: {CYAN}https://console.anthropic.com/{RESET}\n")

        token = ask("  Paste your token (sk-ant-...)")
        if not token or not token.startswith("sk-ant-"):
            print(f"  {YELLOW}⚠ Token doesn't look right (expected sk-ant-...). Saving anyway.{RESET}")
        else:
            print(f"  {GREEN}✓ Token received: {mask_token(token)}{RESET}")
        updates["ANTHROPIC_API_KEY"] = token

    elif provider_id == "openai":
        print(f"\n  Get your API key at: {CYAN}https://platform.openai.com/api-keys{RESET}\n")
        token = ask("  Paste your API key (sk-...)")
        print(f"  {GREEN}✓ Token received: {mask_token(token)}{RESET}")
        updates["OPENAI_API_KEY"] = token

    elif provider_id == "gemini":
        print(f"\n  Get your API key at: {CYAN}https://aistudio.google.com/apikey{RESET}\n")
        token = ask("  Paste your API key (AIza...)")
        print(f"  {GREEN}✓ Token received: {mask_token(token)}{RESET}")
        updates["GEMINI_API_KEY"] = token

    print()

    # ── Step 3: Local Ollama ──────────────────────────────────────────────────
    print(f"{BOLD}Step 3/3: Local LLM (Ollama){RESET}")

    ollama_host = "http://localhost:11434"
    print(f"  Checking Ollama at {ollama_host}...", end=" ", flush=True)
    models = check_ollama(ollama_host)

    if models is None:
        # Try Docker host
        docker_host = "http://host.docker.internal:11434"
        print(f"{YELLOW}not found{RESET}")
        print(f"  Trying Docker host ({docker_host})...", end=" ", flush=True)
        models = check_ollama(docker_host)
        if models:
            ollama_host = docker_host
            print(f"{GREEN}✅ Connected{RESET}")
        else:
            print(f"{RED}not found{RESET}")
            print(f"""
  {YELLOW}⚠ Ollama not reachable.{RESET}
  Make sure Ollama is running: https://ollama.ai
  Or pull a model first:
    ollama pull qwen3.5:9b
""")
            ollama_host = ask("  Ollama host", default="http://localhost:11434")
            models = []
    else:
        print(f"{GREEN}✅ Connected{RESET}")

    updates["LOCAL_HOST"] = ollama_host

    if models:
        print(f"  Available models: {DIM}{', '.join(models[:5])}{RESET}")
        default_model = models[0] if models else "qwen3.5:9b"
        chosen = ask(f"  Which model for private queries?", default=default_model)
        updates["LOCAL_MODEL"] = chosen
        print(f"  {GREEN}✓ {chosen}{RESET}")
    else:
        chosen = ask("  Local model name", default="qwen3.5:9b")
        updates["LOCAL_MODEL"] = chosen

    print()

    # ── Write .env ─────────────────────────────────────────────────────────────
    for key, value in updates.items():
        env_lines = set_env_key(env_lines, key, value)

    write_env(env_lines)

    print(f"""{DIM}─────────────────────────────────────{RESET}
{GREEN}{BOLD}✅ .env configured!{RESET}

Next steps:
  {BOLD}docker compose up -d{RESET}
  open {CYAN}http://localhost:8000{RESET}

To verify:
  curl http://localhost:8000/health
""")


if __name__ == "__main__":
    main()
