# Zipsa Onboarding for Ollama Users

You're already running Ollama locally for privacy. But sometimes you need Claude or GPT for complex tasks. Zipsa sits between your app and the cloud — filtering PII before anything leaves your machine.

**Time to set up:** ~5 minutes

---

## Step 0: Prerequisites Check

Before starting, verify you have:

### Ollama running locally
```bash
curl http://localhost:11434/api/tags
```
You should see a list of your models. If not, [install Ollama](https://ollama.ai) first.

### Docker installed
```bash
docker --version
docker-compose --version
```
Need Docker? [Get Docker Desktop](https://www.docker.com/products/docker-desktop/).

---

## Step 1: Clone and Configure

```bash
git clone https://github.com/sulgik/zipsa.git
cd zipsa
cp .env.example .env
```

---

## Step 2: Pick Your External Provider

Open `.env` and set your cloud provider API key. Pick ONE:

### Claude (Anthropic) — Recommended

**Option A: API Key** (pay-per-use)
```env
EXTERNAL_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```
Get your key: [console.anthropic.com](https://console.anthropic.com/)

**Option B: Claude Max Subscription** (no extra API charges)

If you already have a Claude Max subscription, you can use a setup token instead of an API key — no additional billing.

```bash
# Generate a setup token (requires Claude Code CLI)
claude setup-token
# → outputs: sk-ant-oat01-...
```

```env
EXTERNAL_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-oat01-your-setup-token-here
```

> **How it works:** Claude setup tokens are OAuth access tokens that Anthropic's API accepts in the same `Authorization: Bearer` header as regular API keys. Your Max subscription covers the usage — no extra cost.

### GPT (OpenAI)
```env
EXTERNAL_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```
Get your key: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### Gemini (Google)
```env
EXTERNAL_PROVIDER=gemini
GEMINI_API_KEY=AIza-your-key-here
```
Get your key: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

That's all you need to change. The defaults work with your local Ollama.

---

## Step 3: Start Zipsa

```bash
docker-compose up -d
```

Check it's running:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

---

## Step 4: Change Your App Endpoint

### The key change
```
Before: http://localhost:11434
After:  http://localhost:8000
```

### OpenClaw Example

In your OpenClaw config or `.env`:
```env
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=zipsa-key
OPENAI_MODEL=zipsa
```

### Generic curl Example
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer zipsa-key" \
  -d '{
    "model": "zipsa",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

### Python Example
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="zipsa-key"
)

response = client.chat.completions.create(
    model="zipsa",
    messages=[{"role": "user", "content": "What treatment options exist for diabetes?"}]
)
print(response.choices[0].message.content)
```

---

## Step 5: Verify It Works

### Health check
```bash
curl http://localhost:8000/health
```

### Test query with PII
Send a query with obvious PII to see redaction in action:

```bash
curl -X POST http://localhost:8000/relay \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "John Smith (SSN 123-45-6789) needs treatment advice for his diabetes. HbA1c is 8.4%.",
    "session_id": "test-001"
  }'
```

What happens:
1. Zipsa detects PII (name, SSN)
2. Reformulates the query: "A patient needs diabetes treatment advice. HbA1c is 8.4%."
3. Sends the reformulated version to Claude/GPT
4. Returns the cloud answer to you

Check the logs to see the routing decision:
```bash
docker-compose logs -f zipsa
```

---

## Troubleshooting

### "Connection refused" to Ollama

**Symptom:** Zipsa can't reach your local Ollama.

**Fix:** Make sure Ollama is running and accessible:
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If using Docker, Ollama needs to be accessible from the container
# Edit .env:
LOCAL_HOST=http://host.docker.internal:11434  # macOS/Windows
# or
LOCAL_HOST=http://172.17.0.1:11434  # Linux
```

### "Invalid API key" errors

**Symptom:** Cloud provider rejects requests.

**Fix:** Check your `.env` file:
```bash
# Make sure the key is uncommented and correct
cat .env | grep API_KEY

# Verify the provider matches your key
cat .env | grep EXTERNAL_PROVIDER
```

### Port 8000 already in use

**Symptom:** `docker-compose up` fails with port conflict.

**Fix:** Either stop the conflicting service or change Zipsa's port:
```bash
# Find what's using port 8000
lsof -i :8000

# Or change Zipsa's port in docker-compose.yml
# ports:
#   - "8001:8000"  # Use 8001 instead
```

### Models not found

**Symptom:** Zipsa starts but queries fail with model errors.

**Fix:** Pull the default model:
```bash
ollama pull qwen3.5:9b

# Or change LOCAL_MODEL in .env to a model you have:
ollama list  # See available models
```

---

## Next Steps

- **Multi-turn conversations:** Add `session_id` to maintain context across queries
- **Monitor routing decisions:** Check logs or enable the dashboard
- **Fine-tune privacy settings:** See [README.md](../README.md) for configuration options

---

Questions? [Open an issue](https://github.com/sulgik/zipsa/issues) or email sulgik@gmail.com.
