# Zipsa — Local-First Privacy Gateway for LLM Apps

> Use frontier models without handing over your raw private context, internal knowledge, or secrets.

**TL;DR:** Zipsa is an OpenAI-compatible gateway that sits between your app and Claude, ChatGPT, or Gemini. It inspects each request locally, decides which parts are identity-bound, company-confidential, or safe to abstract, and routes only the prompts that genuinely need cloud knowledge. When it does call the cloud, it rewrites the prompt to remove identifying or proprietary details while preserving the technical, legal, clinical, or operational facts that matter. You get the upside of cloud models without sending raw personal or internal context upstream.

![Privacy First](https://img.shields.io/badge/privacy-local--first-1f6feb)
![OpenAI Compatible](https://img.shields.io/badge/api-openai--compatible-0a7f5a)
![Ollama](https://img.shields.io/badge/local%20llm-Ollama-222222)
![Hybrid Routing](https://img.shields.io/badge/routing-local%20%7C%20hybrid-b35c00)

## Why This Matters

You've got patient records, customer lists, contracts, salary info, SSNs, emails, API keys, incident reports, source code, pricing notes, board drafts, and internal strategy docs — stuff you *don't* want on someone else's cloud servers. But cloud AI is too useful to ignore.

Zipsa solves this by:
- **Local-first:** your personal data, internal business context, and security-sensitive details stay within your private zone (your device, your network, your on-prem infrastructure)
- **Smart routing:** automatically decides what can stay local, what can be abstracted, and what actually needs cloud knowledge
- **Semantic reformulation:** rewrites the entire question to abstract identity and proprietary context (name → profession, hospital → institution type, customer name → enterprise account, product codename → internal system) while preserving the knowledge context cloud needs to actually help
- **Hybrid mode:** gets the best of both — local privacy + cloud knowledge

Drop it in front of any OpenAI-compatible client (like OpenClaw) and forget about it.

> ⚠️ **Experimental software.** APIs might change. Use for evaluation, not production yet.

## How It Works

**Your query comes in →**

1. **Local scan** — Zipsa checks for PII, credentials, and other high-risk markers before anything happens
2. **Classify** — Is this about *private identity, internal company context,* or *general knowledge*?
3. **Route decision:**
   - **Private-only?** → handled entirely within your private zone, done. No cloud call.
   - **Hybrid?** → reformulate (rewrite entire query to remove identity, proprietary context, and secrets) → send to cloud → bring the answer back
4. **Safety scan** — Did anything slip through? If yes, use local answer instead
5. **Synthesize** — Apply cloud answer to *your* actual context (if hybrid)

**Concrete example (semantic reformulation in action):**
- **Your query (local):** "Jane Smith (SSN 123-45-6789), senior ER physician. HbA1c 8.4% on metformin 2000mg + sitagliptin 100mg (eGFR 62). Treatment options?"
- **What cloud sees:** "Healthcare professional, physician, late 30s. Diabetes, HbA1c 8.4%. Current drugs: metformin 2000mg + DPP-4i (100mg). eGFR 62. What escalation strategies?"
- **Cloud answers with:** Generic escalation protocols (renal-safe options, likelihood of HbA1c reduction, monitoring needs)
- **Private side applies back to:** Jane's full profile — her hospital, her insurance, her prior decisions
- **You get:** Best clinical answer *without* Jane's SSN or identity ever touching cloud servers

The same pattern works for internal contracts, support escalations, security incidents, customer strategy, source code, and other proprietary context.

### Examples

**Case 1: Sensitive record lookup (stays local)**
- You: "Look up John's SSN 123-45-6789 and payroll record"
- Cloud: 🚫 (never sees it)
- Zipsa: Uses local model, keeps the personal record local

**Case 2: Domain knowledge with personal data (goes hybrid)**
- You: "Jane Smith (DOB 1985, SSN 123-45-6789) is a physician with HbA1c 8.4%. Treatment options?"
- Zipsa reformulates (not just strip names, rewrite it):
  - Original: *"Jane Smith (DOB 1985, SSN 123-45-6789) is a senior ER physician with HbA1c 8.4% on metformin 2000mg + sitagliptin 100mg (eGFR 62)"*
  - Sent to cloud: *"Healthcare professional, late 30s, with diabetes. HbA1c 8.4%. Current: metformin 2000mg + DPP-4i (sitagliptin 100mg), eGFR 62. What escalation strategies exist?"*
- Cloud: Returns generic escalation strategies (no idea it's Jane)
- Private side: Takes cloud answer, applies back to Jane's full profile
- Result: Best of both — cloud knowledge + Jane's actual context

**Case 3: Internal business context (goes hybrid)**
- You: "Acme renewal is blocked on a 17% discount floor, custom SLA language, and a churn warning from our sales team. Gross margin floor is 61%. What should we offer?"
- Zipsa reformulates (rewrite proprietary context, not just names):
  - Original: *"Acme renewal is blocked on a 17% discount floor, custom SLA language, and a churn warning from our sales team. Gross margin floor is 61%."*
  - Sent to cloud: *"An enterprise renewal is at risk over discount pressure, SLA demands, and churn signals. What negotiation options preserve margin while improving close probability?"*
- Cloud: Returns generic negotiation playbooks
- Private side: Applies them back to the real account, margin floor, contract history, and approval chain
- Result: Better strategy without exposing the customer name or internal deal terms

**Case 4: Pure knowledge (could go hybrid)**
- You: "How do I optimize React components?"
- Zipsa: Routes to cloud (no sensitive context detected)
- Cloud: Answers
- You: Get better answer than local model alone

## Multi-Turn Conversations (The Hard Problem)

One question is easy. But what about a conversation?

Over several turns, you drop hints. Your name, your company, your customer, your condition, the incident, what you've tried. If every assistant message gets sent to Claude with all that history, you're leaking context even if individual turns looked "safe."

But if you *strip* the history before each call, Claude loses track and gives worse answers.

**Zipsa's solution:** Keep two separate conversation threads:
1. **Full thread (private zone)** — everything, including all your personal, business, and operational context
2. **Cloud thread (safe)** — only the turns where you went hybrid

They diverge when you ask local-only questions, re-converge on hybrid ones. Both stay in sync.

So when you ask "What discount did we approve for Acme last quarter?" (local-only), it doesn't appear in what Claude sees. But when you ask "What negotiation levers work when an enterprise customer wants a termination clause?" (hybrid), it shows up and Claude has the strategic context it needs.

This way you get privacy *and* continuity.

## Does It Actually Work?

Ran it on 100 real conversations with sensitive workloads — medical, legal, finance, security, and internal ops.

**Results:**

| | Privacy Risk | Your Data Safe | Answer Quality |
| ---- | ---- | ---- | ---- |
| Local-only | 0% | 100% | 73/100 |
| **Zipsa** | **9%** | **91%** | **78/100** |
| Cloud-direct | 26% | 72% | 79/100 |

**What this means:**
- Zipsa keeps 36% of queries 100% local (zero privacy risk)
- Reformulates the other 64% before sending
- Cuts privacy exposure risk by 2/3
- Barely loses anything on answer quality

## What You Get

- ✅ **Local model = privacy shield** — your raw queries, internal context, and secrets never leave the box
- ✅ **Smart routing** — decides what's safe to send, what stays local, and what must be abstracted first
- ✅ **Full sentence rewriting** — not just PII token-swapping, real reformulation of private and proprietary context
- ✅ **Dual-thread sessions** — keeps local history private, cloud history safe
- ✅ **Private side has final say** — cloud is just a knowledge provider
- ✅ **OpenAI-compatible API** — drop-in replacement for any LLM client
- ✅ **Multi-provider** — works with Claude, Gemini, or OpenAI

## Quick Start

### What You Need

- Docker & Docker Compose
- Ollama running locally (`localhost:11434`)
- A local Ollama model (default: `qwen3.5:9b`)
- API key for Claude, Gemini, or OpenAI

### Docker (Easy)

```bash
git clone https://github.com/sulgik/zipsa.git
cd zipsa
cp .env.example .env
# Edit .env — pick your provider (claudeAPI key, etc)
ollama pull qwen3.5:9b
docker-compose up -d
curl http://localhost:8000/health
```

### Or Local (Native Python)

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Using Zipsa

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="zipsa-key"
)

response = client.chat.completions.create(
    model="zipsa",
    messages=[{
        "role": "user",
        "content": "Acme renewal is asking for a 17% discount and a custom SLA. Our internal floor is 61% margin. What negotiation options do we have?"
    }]
)
print(response.choices[0].message.content)
```

**For multi-turn conversations** (keeps history in sync):

```python
response = client.chat.completions.create(
    model="zipsa",
    messages=[{"role": "user", "content": "They also want a 30-day termination clause. What tradeoffs should we consider?"}],
    extra_body={"session_id": "session-abc123"}
)
```

### cURL

```bash
curl -X POST http://localhost:8000/relay \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "Acme renewal is asking for a 17% discount and a custom SLA. Our internal floor is 61% margin. What negotiation options do we have?",
    "session_id": "session-abc123"
  }'
```

## Using With OpenClaw (Or Any LLM App)

Want Zipsa to handle privacy for your OpenClaw setup? Easy:

**1. Start Zipsa**
```bash
docker-compose up -d
curl http://localhost:8000/health
```

**2. Point OpenClaw to Zipsa**

Instead of sending requests directly to Claude/Gemini, use Zipsa instead:
```env
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=zipsa-key
OPENAI_MODEL=zipsa
```

**3. That's it**

Send your normal OpenClaw queries. Zipsa decides locally what leaves your machine.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="zipsa-key"
)

# Your normal OpenClaw prompt
response = client.chat.completions.create(
    model="zipsa",
    messages=[{
        "role": "user",
        "content": "Acme renewal is asking for a 17% discount and a custom SLA. Our internal floor is 61% margin. What negotiation options do we have?"
    }]
)
```

For **multi-turn workflows**, keep a session_id to maintain conversation continuity:

```python
response = client.chat.completions.create(
    model="zipsa",
    messages=[{"role": "user", "content": "They also want a 30-day termination clause. What tradeoffs should we consider?"}],
    extra_body={"session_id": "case-001"}
)
```

## Authentication

Zipsa supports two ways to authenticate with cloud providers: **API keys** (simple) and **OAuth 2.0** (browser-based, no keys in `.env`).

### Option 1: API Keys (Simple)

Set the provider's API key in `.env` and you're done:

```env
ANTHROPIC_API_KEY=sk-ant-...
# or
GEMINI_API_KEY=AIza...
```

### Option 2: OAuth 2.0 (PKCE)

Use browser-based OAuth to connect without storing API keys. Supports Anthropic (Claude) and Google (Gemini).

**Setup:**

1. Register an OAuth app with the provider and get a client ID/secret
2. Set the OAuth env vars (see Configuration table below)
3. Start Zipsa, then open the authorize URL in your browser:

```bash
# Claude
open http://localhost:8000/auth/claude

# Gemini
open http://localhost:8000/auth/gemini
```

4. Complete the consent flow — Zipsa stores the token automatically

**OAuth endpoints:**

| Endpoint | Method | Description |
| --- | --- | --- |
| `/auth/claude` | GET | Redirect to Anthropic OAuth consent |
| `/auth/claude/callback` | GET | Handle Anthropic OAuth callback |
| `/auth/claude/status` | GET | Check Claude connection status |
| `/auth/claude/revoke` | POST | Revoke stored Claude token |
| `/auth/gemini` | GET | Redirect to Google OAuth consent |
| `/auth/gemini/callback` | GET | Handle Google OAuth callback |
| `/auth/gemini/status` | GET | Check Gemini connection status |
| `/auth/gemini/revoke` | POST | Revoke stored Gemini token |

Tokens are kept in memory by default. Set `TOKEN_ENCRYPTION_KEY` to persist them to an encrypted file (`.tokens.enc`).

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `LOCAL_MODEL` | `qwen3.5:9b` | Ollama model |
| `LOCAL_HOST` | `http://localhost:11434` | Ollama server |
| `EXTERNAL_PROVIDER` | `anthropic` | Choose: `anthropic`, `gemini`, `openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | If using OpenAI |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | If using Anthropic |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | If using Gemini |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic (API key mode) |
| `GEMINI_API_KEY` | — | Required for Gemini (API key mode) |
| `OPENAI_API_KEY` | — | Required for OpenAI |
| `ANTHROPIC_OAUTH_CLIENT_ID` | — | For Anthropic OAuth |
| `ANTHROPIC_OAUTH_CLIENT_SECRET` | — | For Anthropic OAuth |
| `GOOGLE_OAUTH_CLIENT_ID` | — | For Gemini OAuth |
| `GOOGLE_OAUTH_CLIENT_SECRET` | — | For Gemini OAuth |
| `OAUTH_REDIRECT_BASE` | `http://localhost:8000` | Base URL for OAuth callbacks |
| `TOKEN_ENCRYPTION_KEY` | — | Fernet key; enables encrypted token persistence |
| `DEMO_MODE` | `true` | Skip auth checks when true |

## License

**Business Source License 1.1 (BSL 1.1)**

- 🟢 **Free for:** Personal use, research, open-source projects, evaluation
- 🔴 **Commercial use:** Requires a paid license
- 📅 **Change date:** 2029-03-12 — then it automatically becomes Apache 2.0

Questions? Email: [sulgik@gmail.com](mailto:sulgik@gmail.com)

---

## FAQ

**Q: Does Zipsa send my data to the cloud?**
A: Only the minimal, reformulated version when absolutely necessary. Personal context stays within your private zone by design.

**Q: What if I don't trust even the reformulated query?**
A: Set local-only mode and Zipsa will never touch the cloud.

**Q: Can I audit what gets sent?**
A: Yes, logs show every routing decision and what the reformulated query looks like.

**Q: Works with X LLM?**
A: If you can hit it with an OpenAI-compatible API call, Zipsa works with it.
