# Show HN Draft

## Title Options

**Primary:**
> Show HN: Zipsa – I added PII filtering between my Ollama and Claude in 5 minutes

**Alternatives:**
> Show HN: Zipsa – Privacy layer for Ollama users who still need Claude/GPT

> Show HN: Zipsa – Use Claude without sending your raw private data

---

## Post Body

I run Ollama locally for privacy, but I still need Claude for complex reasoning tasks. The problem: those complex queries often contain exactly the stuff I don't want on someone else's servers — patient records, internal company context, customer names, salary info.

So I built Zipsa. It sits between your app and the cloud LLM, rewrites your prompts to remove PII and proprietary context, gets the answer, and applies it back to your actual situation.

**Setup for Ollama users:**

```bash
git clone https://github.com/sulgik/zipsa.git
cd zipsa
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env
docker-compose up -d
```

Then point your app to `localhost:8000` instead of `localhost:11434`. That's it.

**What happens:**
- Your query: "Jane Smith (SSN 123-45-6789), senior physician. HbA1c 8.4% on metformin. Treatment options?"
- What cloud sees: "Healthcare professional. Diabetes, HbA1c 8.4%. Current: metformin. What escalation strategies?"
- Cloud returns generic medical advice
- Zipsa applies it back to Jane's actual profile on your end

**Benchmark (100 real sensitive conversations):**

| | Privacy Risk | Answer Quality |
| --- | --- | --- |
| Local-only | 0% | 73/100 |
| **Zipsa** | **9%** | **78/100** |
| Cloud-direct | 26% | 79/100 |

Zipsa keeps 36% of queries fully local (zero privacy risk) and reformulates the rest before sending. Privacy exposure drops by 2/3 with minimal quality loss.

**Honest caveats:**
- Experimental. APIs may change.
- BSL licensed (free for personal/research/OSS, commercial needs a license)
- The reformulation isn't perfect — some context loss is inherent

**Tech:** Python/FastAPI, OpenAI-compatible API, works with Claude/GPT/Gemini.

GitHub: https://github.com/sulgik/zipsa

Would love feedback from r/LocalLLaMA and r/selfhosted folks who are already privacy-conscious but need cloud LLM capabilities for certain tasks.

---

## Notes for posting

**Best times to post on HN:**
- Weekday mornings (Pacific time), around 6-9 AM PT
- Avoid weekends

**Subreddits to cross-post:**
- r/LocalLLaMA (primary audience)
- r/selfhosted
- r/privacy
- r/machinelearning (if it gains traction)

**Key talking points for comments:**
1. This isn't about avoiding cloud entirely — it's about controlling *what* goes to cloud
2. The reformulation is semantic, not just token-swapping names
3. Local model has final say on whether something is sensitive
4. Logs show exactly what was sent to cloud (auditable)
