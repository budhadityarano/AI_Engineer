# AI Industry Trends 2025 — Interview Briefing

## 1. The Rise of AI Engineering

**What's happening:** A new role category between software engineering and ML research — AI engineers build applications using foundation models without necessarily training models from scratch.

**Key skills in demand:**
- Prompt engineering and evaluation
- RAG pipeline design
- Agent orchestration (LangChain, LangGraph, Claude tool use)
- LLM ops: cost management, latency optimization, caching
- Evaluation frameworks (LLM-as-judge, RAGAS, custom evals)

**Interview angle:** "I understand that most AI product value now comes from how you wire together pre-trained models, not from training your own."

---

## 2. Agents and Autonomous AI Systems

**What's happening:** LLMs used not just for generation but as reasoning engines in agentic loops — planning, tool use, memory, multi-agent collaboration.

**Key patterns:**
- **ReAct**: Reason → Act (tool) → Observe → repeat
- **Supervisor agents**: orchestrator delegates to specialists
- **Long-running agents**: hours-long tasks with checkpointing
- **Computer use**: Claude and GPT-4o can control browser/desktop

**Production challenges:**
- Cost control (100 agent steps = 100× API calls)
- Reliability (agents drift, loop, or fail silently)
- Observability (trace every step like a distributed system)

**Tools:** LangGraph, Anthropic Claude Agents, AutoGen, CrewAI

---

## 3. Long Context and Context Engineering

**What's happening:** Frontier models now support 200K-1M token contexts. "In-context learning" is replacing fine-tuning for many use cases.

**Context engineering:** Deliberately constructing the context window to maximize model performance — which documents to include, in what order, how to format them.

**Key findings:**
- Lost-in-the-middle: models perform worse on content in the middle of long contexts
- Prompt caching: cache repeated system prompts and documents for cost reduction
- Needle-in-haystack: evaluation of recall across long contexts

**When long context beats RAG:** When you want the model to synthesize across many documents, not just retrieve from them.

---

## 4. Open Source vs Closed Models

**The gap is narrowing:**
- Llama 3.1 405B competitive with GPT-4
- DeepSeek V3 / R1: China open-weights, top-tier performance at low cost
- Mistral, Qwen, Gemma all strong mid-size options

**When to choose open source:**
- Data privacy requirements (PII, HIPAA)
- Custom fine-tuning needed
- Cost at scale (self-hosted inference cheaper at high volume)
- Specific architecture control

**When to choose closed API:**
- Latest capabilities needed immediately
- No MLOps team to manage infrastructure
- Multimodal, function calling, safety guardrails

---

## 5. Efficient Training: LoRA, QLoRA, and Fine-Tuning at Scale

**What's happening:** Full fine-tuning of LLMs is giving way to PEFT methods that require 10-100x less compute and memory.

**State of practice (2025):**
- **LoRA/QLoRA**: standard for supervised fine-tuning of LLMs
- **DPO (Direct Preference Optimization)**: simpler than RLHF, no reward model needed
- **GRPO**: group relative policy optimization used in DeepSeek R1

**Fine-tuning decision framework:**
```
Is zero-shot good enough?         → Use zero-shot
Can few-shot prompting work?      → Use few-shot
Need consistent style/format?    → Fine-tune with LoRA (< 1K examples)
Domain shift too large?          → Full fine-tune or domain-adapted base model
```

---

## 6. Multimodal AI Goes Mainstream

**What's happening:** All frontier models now process images. Video understanding emerging. Audio becoming standard.

**Practical applications:**
- Document AI: extract data from PDFs, invoices, forms
- Visual QA: analyze charts, diagrams, screenshots
- Code from screenshots: generate code from UI mockups
- Medical imaging: assist radiologists (with disclaimers)

**Architecture:** Vision encoder (ViT/SigLIP) → Projection layer → LLM. Images tokenized as ~1500-4000 tokens.

---

## 7. Reasoning Models and "Slow Thinking"

**What's happening:** Models that generate extended internal reasoning ("thinking") before answering hard problems. OpenAI o1/o3, Claude extended thinking, DeepSeek R1.

**Key insight:** Scaling inference compute (more tokens of reasoning) improves hard reasoning tasks similar to how scaling training compute improves language tasks.

**Trade-off:** 10-100× more tokens generated → 10-100× more cost and latency. Use only for hard reasoning tasks.

**When to use:** Math competition problems, complex code debugging, multi-step planning, logical deduction.

---

## 8. AI Safety and Alignment in Practice

**What's happening:** Safety and alignment going from research to production — companies now have Trust & Safety teams for AI products.

**Key concepts for interviews:**
- **Hallucination**: model confidently generates false information. Mitigation: RAG, grounding, citations.
- **Prompt injection**: malicious input overrides system prompt. Mitigation: input sanitization, privilege separation.
- **Jailbreaking**: bypass safety guardrails. Defense: RLHF, Constitutional AI, red-teaming.
- **Alignment**: model does what humans intend, not just what was literally specified.

**Evaluation:** Red-teaming, adversarial testing, automated safety eval suites.

---

## 9. Vector Databases and the Data Infrastructure Stack

**What's happening:** Vector databases are now standard infrastructure for AI applications, similar to how relational DBs are standard for CRUD apps.

**Production stack (2025):**
```
Data → Chunking → Embedding (voyage-3, text-embedding-3) → Vector DB (Pinecone/Weaviate)
Query → Embed → ANN search → Re-rank → LLM context → Response
```

**Top production vector DBs:**
- Pinecone: managed, serverless, easiest to use
- Weaviate: open-source, hybrid search built-in
- Qdrant: open-source, Rust-based, fast
- pgvector: PostgreSQL extension — simpler stack

---

## 10. Cost Optimization in AI Applications

**Why it matters:** LLM costs can dominate AI product P&L at scale. Engineering for cost efficiency is now a core skill.

**Optimization hierarchy (highest to lowest impact):**
1. **Model selection**: Haiku vs Sonnet vs Opus — 10-100x cost difference
2. **Prompt caching**: cache static system prompts — 90% cost reduction on repeated calls
3. **Output length control**: set max_tokens, instruct model to be concise
4. **Batching**: batch async requests for higher throughput
5. **Caching responses**: exact-match cache for common queries (Redis)
6. **Quantization**: INT8/INT4 for self-hosted models
7. **Speculative decoding**: 2-3x faster generation for self-hosted

**Typical cost targets:**
- Simple classification: < $0.001/call
- RAG QA: < $0.01/call
- Complex agent: < $0.10/session

---

## Key Numbers to Know

| Metric | Value |
|--------|-------|
| GPT-4 training compute (est.) | ~1×10²⁵ FLOPs |
| Llama 3.1 70B parameters | 70 billion |
| Claude 200K context | ~150,000 words |
| Transformer attention complexity | O(n²d) per layer |
| Typical embedding dimension | 768-4096 |
| BERT pretraining tokens | 3.3 billion |
| GPT-3 pretraining tokens | 300 billion |
| Llama 3 pretraining tokens | 15+ trillion |
| Approx chars per token (English) | ~4 characters |
| Approx words per token (English) | ~0.75 words |
