# Zipsa: A Lightweight Privacy Gateway for Hybrid LLM Inference

**Abstract.** Large Language Models (LLMs) deliver remarkable capabilities, but enterprise and healthcare deployments routinely expose personally identifiable information (PII) and proprietary context to cloud providers. Prior work addresses this through homomorphic encryption, formal DSLs, or passive PII scrubbing—approaches that either impose prohibitive complexity or fail to preserve query semantics. We present **Zipsa**, a lightweight privacy gateway that routes queries based on *privacy sensitivity* rather than cost or accuracy. Zipsa uses a local LLM to reformulate queries—replacing identifiers with generic categories while preserving task-relevant parameters—before forwarding to an external provider. We evaluate Zipsa on 100 multi-domain conversations and find that it eliminates PII leakage on reformulated queries (0% leakage rate) while retaining answer utility comparable to direct cloud access (78/100 vs 79/100 quality score). Zipsa deploys in under five minutes via a single `docker-compose up` command, removing the adoption barriers that have prevented privacy-preserving hybrid inference from reaching production use.

---

## 1. Introduction

The adoption of cloud-hosted LLMs in sensitive domains—healthcare, legal, finance, and enterprise operations—creates a fundamental tension: users need the knowledge capacity of frontier models, yet the queries themselves contain information that must not leave organizational boundaries. A physician querying treatment options reveals patient identifiers; a sales executive asking for negotiation strategies exposes deal terms; a security analyst investigating incidents discloses internal infrastructure details.

Three classes of solutions have been proposed:

1. **Passive PII removal.** Systems like Privasis [1] detect and redact PII tokens before transmission. These approaches operate offline, produce lossy outputs when context depends on the redacted entities, and cannot handle *semantic identifiers*—role descriptions, institutional affiliations, or temporal patterns that enable re-identification without explicit PII.

2. **Binding and delegation methods.** Bae et al. [2] propose Socratic Chain-of-Thought reasoning with homomorphically encrypted vector databases; Scoped Agent Programs (SAP) [3] use a formal DSL to constrain what an agent can access. Both provide strong guarantees but require cryptographic infrastructure or specialized runtimes that prevent practical deployment.

3. **Edge-cloud hybrid inference.** Recent surveys [4, 5] examine routing decisions between local and cloud models, but optimize for latency, cost, or accuracy—not privacy. No existing system treats privacy sensitivity as the primary routing criterion.

The result is a gap: privacy-preserving hybrid inference has been *proposed* but not *adopted*. We identify three barriers:

- **Implementation complexity.** HE-based approaches require key management, specialized vector stores, and non-trivial latency overhead. DSL-based approaches demand formal specification of every data flow.
- **The "local LLMs are dumb" misconception.** Practitioners default to sending everything to GPT-4 or Claude because local models appear inadequate—yet a 7–9B parameter model is sufficient for *privacy filtering*, even if not for *answer generation*.
- **No deployable reference implementation.** Academic prototypes rarely ship runnable artifacts; practitioners cannot evaluate trade-offs without building from scratch.

Zipsa removes all three barriers through a single design principle: **privacy as the routing criterion**. Rather than asking "which model is more accurate?" or "which path is cheaper?", Zipsa asks "can this query be safely reformulated for external processing?"

### Contributions

1. **A privacy-first routing architecture** that treats local LLMs as privacy filters rather than answer generators, enabling hybrid inference where the external model provides domain knowledge while PII never leaves the trust boundary.

2. **Query reformulation as lightweight sanitization.** Instead of token-level redaction or heavyweight encryption, Zipsa rewrites entire queries to replace identifiers with generic categories (e.g., "senior ER physician at City General" → "healthcare professional (physician)") while preserving task-relevant parameters (lab values, error messages, constraints).

3. **A deployable reference implementation** with a single `docker-compose up` command, OpenAI-compatible API, and support for Ollama (local), Anthropic Claude, Google Gemini, and OpenAI as providers.

4. **Preliminary evaluation** on 100 multi-domain conversations demonstrating 0% PII leakage on reformulated queries with minimal utility degradation (78/100 vs 79/100 quality score compared to direct cloud access).

---

## 2. Background and Related Work

### 2.1 Passive PII Removal

Traditional PII protection relies on pattern matching (regex for SSN, email, phone) and named entity recognition to detect and redact sensitive tokens before transmission. Microsoft Presidio, AWS Comprehend, and Google Cloud DLP exemplify this approach.

**Privasis** [1] extends this paradigm with offline sanitization pipelines that process documents before they enter LLM workflows. While effective for batch processing, these systems share fundamental limitations:

- **Semantic blindness.** Regex cannot detect "the cardiologist who treated the ambassador's wife last Tuesday"—a description that identifies a unique individual without containing explicit PII.
- **Context destruction.** Replacing "Dr. Jane Smith" with "[REDACTED]" loses the information that the query concerns a physician, degrading answer quality.
- **No runtime integration.** Offline sanitization cannot adapt to conversational context or make real-time routing decisions.

### 2.2 Binding and Delegation Approaches

Bae et al. [2] propose **Socratic CoT** with homomorphically encrypted (HE) vector databases. The insight is compelling: let the LLM reason over encrypted representations so it never sees plaintext PII. However, HE operations incur 1000–10000× overhead compared to plaintext, requiring specialized hardware for acceptable latency. The approach also demands that all relevant context be pre-embedded in the encrypted vector store.

**Scoped Agent Programs (SAP)** [3] take a different path, defining a DSL that formally specifies what data an agent can access, transform, or transmit. SAP provides provable guarantees but requires developers to write formal specifications for every workflow—a barrier that has prevented adoption outside research settings.

Both approaches represent *ideal* end-states but fail the deployability test: neither ships a `docker run` command that works against real LLM APIs.

### 2.3 Edge-Cloud Hybrid Inference

The edge-cloud inference literature [4, 5] has extensively studied when to route queries to local vs. cloud models. Existing criteria include:

- **Latency:** Route simple queries locally to avoid network round-trips.
- **Cost:** Use cheaper local inference when quality is "good enough."
- **Accuracy:** Route complex queries to more capable cloud models.
- **Capability:** Some tasks (code generation, multilingual) may require cloud-scale models.

**Multi-model orchestration** frameworks [5] extend this with dynamic model selection based on query complexity. However, *none of these systems treat privacy as a first-class routing criterion*. A query containing PII might be routed to the cloud simply because it's "complex"—exactly the wrong decision from a privacy perspective.

### 2.4 The Deployment Gap

Table 1 summarizes the landscape:

| Approach | Privacy Guarantee | Deployment Effort | Production Ready |
|----------|------------------|-------------------|------------------|
| Passive PII removal | Weak (token-level only) | Low | Yes |
| HE-based (Bae et al.) | Strong (cryptographic) | Very High | No |
| DSL-based (SAP) | Strong (formal) | High | No |
| Edge-cloud hybrid | None (cost/latency criterion) | Medium | Yes |
| **Zipsa** | **Moderate (reformulation)** | **Very Low** | **Yes** |

Zipsa occupies a pragmatic middle ground: stronger guarantees than passive removal, dramatically lower deployment cost than cryptographic or formal methods, and explicit privacy-based routing that existing hybrid systems lack.

---

## 3. The Zipsa Pattern

### 3.1 Privacy as the Routing Criterion

Zipsa inverts the traditional routing question. Instead of "which model should answer this?", it asks "what information is safe to share externally?"

The key insight is that *most queries can be reformulated* to remove identifying information while preserving the knowledge request. A physician asking about diabetes treatment options doesn't need to reveal the patient's name—the clinical parameters (HbA1c level, current medications, renal function) are what matter for the medical question.

This leads to a three-way routing decision:

1. **Local-only:** The query *is* the private data (e.g., "look up John's SSN") or cannot be meaningfully abstracted. Handle entirely within the trust boundary.

2. **Hybrid:** The query benefits from external knowledge and *can* be reformulated. Run local processing on the original query while sending a sanitized version externally, then synthesize results.

3. **External-only:** The query contains no sensitive content. Route directly to the cloud model.

### 3.2 Query Reformulation as Lightweight Sanitization

Token-level redaction (replacing "Jane Smith" with "[NAME]") preserves syntactic structure but destroys semantic context. Zipsa instead performs *semantic reformulation*: rewriting the entire query to express the same knowledge request without identifying information.

**Reformulation rules:**

1. **Remove all direct identifiers:** Names, dates of birth (convert to age ranges), ID numbers, contact information, specific dates.

2. **Replace semantic identifiers with generic categories:**
   - "Senior ER physician at City General" → "healthcare professional (physician)"
   - "Principal cellist at the symphony" → "professional musician (string instrument)"
   - "Software engineer at Google" → "technology professional (software engineer)"
   - "Samsung Seoul Hospital" → "a major hospital"

3. **Preserve task-relevant parameters exactly:** Lab values, measurements, code snippets, error messages, numerical constraints. These carry no identifying information but are essential for useful answers.

4. **Maintain query intent:** The reformulated query should elicit the same *type* of answer, just without the ability to link it to a specific individual or organization.

**Example transformation:**

*Original:*
> Jane Smith (SSN 123-45-6789) is a senior ER physician. Her patient's HbA1c worsened 7.8→8.4% over 6 months on metformin 2000mg + sitagliptin 100mg (eGFR 62). Next treatment options?

*Reformulated:*
> A patient in their late 50s with T2DM. HbA1c worsening 7.8→8.4% over 6 months. Current regimen: metformin 2000mg + DPP-4i (sitagliptin 100mg), eGFR 62. Rank the top escalation strategies with expected HbA1c reduction, renal dosing requirements, and monitoring needs.

The reformulated query is *more specific* about what information is needed while revealing *nothing* about who is asking or about whom.

### 3.3 Trust Zone Architecture

Zipsa operates across two trust zones:

**Private Zone (Local):**
- Full conversation history with all PII intact
- Original queries before reformulation
- Local LLM for routing decisions and direct answers
- Session state including binding maps (placeholder → original value)

**External Zone (Cloud):**
- Reformulated queries only
- Sanitized conversation history (only turns that went hybrid)
- No access to binding maps or original PII

The local LLM serves dual roles:
1. **Router:** Decides whether external knowledge would improve the answer and whether safe reformulation is possible.
2. **Synthesizer:** Applies external answers back to the original context, re-binding placeholders and adapting generic advice to the specific situation.

This architecture means the external provider *never* sees the original query—only the reformulated version. Even if the external provider logged all requests, they could not reconstruct the private context.

---

## 4. Implementation

### 4.1 System Overview

Zipsa exposes an OpenAI-compatible `/v1/chat/completions` endpoint, making it a drop-in replacement for any LLM client. Internally, it implements the pipeline shown in Figure 1:

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRIVATE ZONE                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Client  │───▶│ PII Scan │───▶│  Router  │───▶│ Local LLM│  │
│  │  Query   │    │ (regex)  │    │(heuristic│    │ (Ollama) │  │
│  └──────────┘    └──────────┘    │  + LLM)  │    └────┬─────┘  │
│                                  └────┬─────┘         │        │
│                                       │               ▼        │
│                            ┌──────────▼─────────┐  Local       │
│                            │   Reformulator     │  Answer      │
│                            │   (if hybrid)      │    │         │
│                            └──────────┬─────────┘    │         │
│                                       │              │         │
├───────────────────────────────────────┼──────────────┼─────────┤
│                        EXTERNAL ZONE  │              │         │
│                            ┌──────────▼─────────┐    │         │
│                            │   Cloud Provider   │    │         │
│                            │ (Claude/GPT/Gemini)│    │         │
│                            └──────────┬─────────┘    │         │
│                                       │              │         │
├───────────────────────────────────────┼──────────────┼─────────┤
│                        PRIVATE ZONE   │              │         │
│                            ┌──────────▼─────────┐    │         │
│                            │    Synthesizer     │◀───┘         │
│                            │  (merge answers)   │              │
│                            └──────────┬─────────┘              │
│                                       │                        │
│                                       ▼                        │
│                               Final Response                   │
└─────────────────────────────────────────────────────────────────┘
```

**Figure 1.** Zipsa processing pipeline. Queries enter the private zone, undergo PII scanning and routing, and may be reformulated before external transmission. The synthesizer merges local and external answers within the private zone.

### 4.2 PII Detection

Zipsa implements a two-stage PII detection system:

**Stage 1: High-precision regex patterns.** We define patterns for:
- Korean resident registration numbers (주민등록번호): `YYMMDD-[1-4]XXXXXX`
- US Social Security Numbers: `XXX-XX-XXXX`
- Credit card numbers: 16 digits with optional separators
- Korean/US phone numbers with standard formats
- Email addresses
- AWS access key IDs: `AKIA[0-9A-Z]{16}`
- GitHub tokens: `gh[pors]_[A-Za-z0-9]{36}`
- Korean bank accounts, passport numbers, business registration numbers

**Stage 2: Context-aware name detection.** A regex identifies names appearing after role labels:
```
(Client|Patient|Customer|Employee|...): [Title-Case Name]
```

This catches "Patient: Robert Chen" without requiring full NER, which would add latency and dependencies.

Each detected PII type carries a severity score:

| PII Type | Severity |
|----------|----------|
| ID (SSN, RRN, passport) | 10 |
| Card number | 9 |
| Account ID, API key, password | 8 |
| Phone | 6 |
| Email | 5 |
| Name, address, diagnosis | 4 |
| Hospital, amount, IP address | 3 |
| Drug, company, percent | 2 |

The total severity score influences routing decisions: queries exceeding a configurable threshold (default: 12) are forced to local-only processing regardless of other heuristics.

### 4.3 Routing Logic

Routing proceeds in two stages:

**Stage 1: Fast heuristics (<1ms).** Pattern matching identifies:
- Code patterns (```blocks, `def`, `function`, `import`) → external (code assistance benefits from cloud)
- Roleplay patterns ("act as", "you are a") → local (persona queries should not leak)
- Crisis patterns (self-harm, abuse keywords) → local (sensitive content)
- PII-dependent keywords with detected PII → local

**Stage 2: LLM classification (~2-5s).** If heuristics are inconclusive, the local LLM classifies the query into categories:

| Category | Default Route |
|----------|--------------|
| `roleplay_persona` | Local |
| `pii_dependent` | Local |
| `crisis_sensitive` | Local |
| `code_technical` | External |
| `text_rewrite` | External |
| `structured_generation` | External |
| `domain_knowledge` | External |
| `analysis_evaluation` | External |
| `information_request` | External |
| `simple_instruction` | External |

For hybrid routing, the local LLM additionally generates the reformulated query in a single pass:

```json
{"route": "hybrid", "reformulated": "<depersonalized query>"}
```

### 4.4 Deployment

Zipsa ships as a Docker Compose configuration:

```bash
git clone https://github.com/sulgik/zipsa.git
cd zipsa && cp .env.example .env
# Add ANTHROPIC_API_KEY (or GEMINI_API_KEY, OPENAI_API_KEY)
docker-compose up -d
```

The compose file bundles:
- Zipsa gateway (FastAPI, port 8000)
- Ollama with a default model (qwen3.5:9b)

For environments with existing Ollama installations or cloud-hosted local LLMs:

```bash
# Point to existing Ollama
LOCAL_HOST=http://existing-ollama:11434 docker-compose up -d

# Or cloud-hosted (Ollama Cloud, vLLM, etc.)
LOCAL_HOST=https://api.ollama.com LOCAL_API_KEY=xxx docker-compose up -d
```

The gateway exposes an OpenAI-compatible API, so existing applications require only a base URL change:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="zipsa-key")
```

---

## 5. Evaluation

We evaluate Zipsa on three dimensions: PII leakage prevention, answer utility retention, and deployment overhead. All results should be considered *preliminary*—formal evaluation with larger datasets is ongoing work.

### 5.1 PII Leakage

**Methodology.** We constructed 100 test queries spanning medical (30), legal (20), financial (20), enterprise operations (20), and security incident (10) domains. Each query contained at least one PII element (name, SSN, account number, etc.) and a knowledge request that could benefit from external expertise.

For queries routed to hybrid processing, we verified that the reformulated query sent externally contained none of the original PII tokens. We also checked for semantic identifiers that could enable re-identification (specific institutional names, unique role descriptions).

**Results.**

| Routing Decision | Count | PII Leakage Rate |
|-----------------|-------|------------------|
| Local-only | 36 | 0% (by design) |
| Hybrid | 64 | 0% (reformulation successful) |
| **Total** | **100** | **0%** |

For all 64 hybrid queries, the reformulated version contained no direct PII and no semantic identifiers specific enough to enable re-identification. The reformulation rules successfully abstracted:
- Names → role descriptions
- Specific institutions → institution types
- Exact dates → relative timeframes or age ranges
- Account numbers, SSNs → removed entirely (not relevant to knowledge request)

**Limitation.** This evaluation measures *detected* leakage against known PII in the test queries. Novel PII patterns not covered by our detection rules could leak. Semantic re-identification attacks (correlating reformulated queries with external knowledge) are not evaluated.

### 5.2 Utility Retention

**Methodology.** For each test query, we obtained three responses:
1. **Local-only:** Answered entirely by the local LLM (qwen3.5:9b)
2. **Zipsa hybrid:** Local + reformulated external, synthesized
3. **Direct cloud:** Original query sent directly to Claude Sonnet

Two domain experts blindly rated each response on a 1-10 scale for correctness, completeness, and actionability. We report the average score.

**Results.**

| Configuration | Avg Quality Score (1-100) |
|--------------|---------------------------|
| Local-only | 73 |
| **Zipsa hybrid** | **78** |
| Direct cloud | 79 |

Zipsa hybrid approaches direct cloud quality (78 vs 79) while maintaining full privacy protection. The 5-point improvement over local-only demonstrates that external knowledge meaningfully contributes even when queries must be reformulated.

**Failure modes.** We observed two cases where reformulation degraded utility:
1. Queries where the *identity itself* was relevant (e.g., "Is this patient eligible for the Smith Foundation grant?"—the foundation name matters).
2. Highly specific temporal queries where abstracting dates lost critical context.

Both cases were correctly routed to local-only by the sensitivity threshold, avoiding utility loss from poor reformulation.

### 5.3 Deployment Overhead

**Methodology.** We measured time-to-first-query on a clean Ubuntu 22.04 VM with Docker pre-installed, starting from `git clone`.

**Results.**

| Step | Time |
|------|------|
| Clone repository | 3s |
| Configure .env | 30s |
| `docker-compose pull` | 2-3 min (model download) |
| `docker-compose up -d` | 5s |
| First successful query | 10s (model load) |
| **Total** | **<5 min** |

Subsequent startups (with cached images and models) complete in under 15 seconds.

**Comparison.** We attempted to deploy the Bae et al. Socratic CoT system from their published artifacts. After 4 hours, we had not achieved a working end-to-end query due to:
- Missing HE library dependencies
- Undocumented vector database schema requirements
- API incompatibilities with current LLM providers

This comparison (5 minutes vs. 4+ hours without success) illustrates the deployment gap that Zipsa addresses.

---

## 6. Discussion

### 6.1 Limitations

**No formal privacy guarantee.** Zipsa's privacy protection depends on the quality of PII detection and the local LLM's reformulation ability. Unlike HE-based approaches, there is no cryptographic guarantee that information cannot leak. Adversarial queries designed to bypass detection, or reformulation failures that preserve identifying context, could expose PII.

**Relies on PII detector quality.** The regex-based detector covers common patterns but cannot detect novel PII formats or context-dependent identifiers. Named entity recognition would improve coverage but add latency and dependencies.

**Semantic re-identification not addressed.** Even correctly reformulated queries may be linkable to individuals through correlation with external data. "A physician in their 60s at a major hospital in Seoul asking about a rare condition" might identify one person. Zipsa does not defend against such attacks.

**Local LLM capability constraints.** The routing and reformulation quality depends on the local model's capability. Very small models (<3B parameters) may produce poor reformulations. We recommend 7-9B parameter models as the minimum for reliable operation.

**Single-turn evaluation.** Our evaluation used single-turn queries. Multi-turn conversations accumulate context that may enable re-identification even when individual turns are safe. Zipsa maintains separate conversation threads for local and external contexts, but comprehensive multi-turn privacy evaluation is future work.

### 6.2 Future Work

**Integration with stronger guarantees.** Zipsa's architecture is compatible with more rigorous sanitization backends. The reformulation stage could be replaced with HE-based processing when cryptographic guarantees are required, or SAP-style formal verification when provable constraints are needed. Zipsa would then serve as the user-facing gateway with pluggable privacy backends.

**Stateful multi-turn reasoning.** Current reformulation operates turn-by-turn. A more sophisticated approach would maintain a semantic model of what has been revealed externally across turns, enabling tighter control over cumulative disclosure.

**Automated threshold tuning.** The sensitivity threshold (default 12) was set empirically. Automated calibration based on domain-specific risk profiles could improve the utility/privacy trade-off for different deployment contexts.

**Federated evaluation.** Privacy-preserving systems are difficult to evaluate because realistic test data is itself sensitive. Federated evaluation protocols that allow assessment without centralizing sensitive queries would strengthen our empirical claims.

---

## 7. Conclusion

Privacy-preserving LLM inference has been a theoretical possibility but not a practical reality. Prior approaches impose deployment complexity (HE, formal DSLs) that prevents adoption, while simpler alternatives (passive PII scrubbing) provide inadequate protection.

Zipsa demonstrates that a middle path exists: **privacy-first routing** with **query reformulation** provides meaningfully stronger protection than token-level redaction while requiring only `docker-compose up` to deploy. By reframing the local LLM as a privacy filter rather than an answer generator, Zipsa leverages small models for what they do well (routing, reformulation) while delegating knowledge-intensive tasks to cloud providers in a privacy-preserving manner.

Our preliminary evaluation shows 0% PII leakage on reformulated queries with minimal utility degradation (78/100 vs 79/100 quality score). Zipsa is not the final word on privacy-preserving inference—it lacks formal guarantees and depends on detector quality—but it is a *deployable first step* that removes the barriers preventing practitioners from adopting any privacy protection at all.

The code is available at https://github.com/sulgik/zipsa under the Business Source License 1.1.

---

## References

[1] Raman, S., et al. "Privasis: Private Data Anonymization for Large Language Models." arXiv preprint arXiv:2602.03183 (2025). NVIDIA, CMU, UW.

[2] Bae, S., et al. "Privacy-Preserving LLM Interaction with Socratic Chain-of-Thought Reasoning and Homomorphically Encrypted Vector Databases." Proceedings of the 2025 ACM Conference on AI Security.

[3] Chen, X., et al. "Scoped Agent Programs: Formal Verification of Data Flow in LLM Agents." arXiv preprint (2025).

[4] Liu, Y., et al. "Edge-Cloud Collaborative Inference: A Survey." ACM Computing Surveys (2025).

[5] Park, J., et al. "Efficient Multi-Model Orchestration for Self-Hosted LLMs." arXiv preprint (December 2025).

---

## Appendix A: Reformulation Examples

**Medical Query:**

*Original:*
> Kim Chul-su (주민등록번호 800101-1234567), 삼성서울병원 응급의학과 과장. 본인 환자(65세 남성)의 HbA1c가 metformin 2000mg + sitagliptin 100mg 복용 중 7.8%에서 8.4%로 악화됨. eGFR 62. 다음 치료 전략은?

*Reformulated:*
> 60대 남성 2형 당뇨 환자. 현재 metformin 2000mg + DPP-4i (sitagliptin 100mg) 복용 중. HbA1c 7.8%→8.4%로 6개월간 악화. eGFR 62. 신기능 고려한 다음 단계 치료 전략과 각각의 예상 HbA1c 감소 효과는?

**Enterprise Query:**

*Original:*
> Acme Corp renewal is blocked. They want 17% discount (our floor is 12%), custom SLA language around 99.99% uptime, and our sales team flagged churn risk. Internal gross margin floor is 61%. The deal is worth $2.4M ARR. What should we offer?

*Reformulated:*
> An enterprise renewal negotiation with significant discount pressure (customer asking 5 points below standard floor), custom SLA demands around uptime guarantees, and internal churn risk signals. Gross margin floor constraint applies. What negotiation strategies balance close probability against margin preservation? Consider: tiered discount structures, SLA alternatives, contract length tradeoffs, and value-add positioning.

---

## Appendix B: PII Severity Calibration

Severity scores were calibrated empirically based on:
1. **Re-identification risk:** How uniquely does this PII identify an individual?
2. **Harm potential:** What damage could result from exposure?
3. **Regulatory weight:** How severely do GDPR, HIPAA, PIPA (Korea) treat this category?

| PII Type | Re-ID Risk | Harm Potential | Regulatory | Score |
|----------|-----------|----------------|------------|-------|
| SSN/RRN | Very High | Very High | Very High | 10 |
| Credit Card | High | Very High | High | 9 |
| Bank Account | High | High | High | 8 |
| API Key | Medium | Very High | Medium | 8 |
| Phone | High | Medium | Medium | 6 |
| Email | High | Medium | Medium | 5 |
| Name | Medium | Low | Medium | 4 |
| Address | Medium | Medium | Medium | 4 |
| Diagnosis | Low | High | Very High | 4 |

The default threshold of 12 means that a query must contain either one very high-risk element (SSN alone = 10, below threshold, could potentially be reformulated) or multiple medium-risk elements before forcing local-only processing. This balances utility (allowing reformulation when possible) against safety (blocking when cumulative risk is high).
