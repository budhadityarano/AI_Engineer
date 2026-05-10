# Model Comparison — 2024-2025

## Frontier LLM Comparison (as of mid-2025)

| Model | Provider | Context | Key Strengths | Pricing (est.) |
|-------|----------|---------|---------------|----------------|
| Claude Opus 4.7 | Anthropic | 200K | Reasoning, coding, safety, long context | ~$15/1M in, $75/1M out |
| Claude Sonnet 4.6 | Anthropic | 200K | Balance of capability and cost | ~$3/1M in, $15/1M out |
| Claude Haiku 4.5 | Anthropic | 200K | Speed, low cost, tool use | ~$0.80/1M in, $4/1M out |
| GPT-4o | OpenAI | 128K | Multimodal, function calling, speed | ~$5/1M in, $15/1M out |
| GPT-4o mini | OpenAI | 128K | Cost-effective | ~$0.15/1M in, $0.60/1M out |
| Gemini 1.5 Pro | Google | 1M | Longest context, multimodal | ~$3.5/1M in, $10.5/1M out |
| Gemini 1.5 Flash | Google | 1M | Fast, cheap, 1M context | ~$0.075/1M in, $0.30/1M out |
| Llama 3.1 405B | Meta | 128K | Open weights, self-hostable | Free (compute only) |
| Llama 3.1 70B | Meta | 128K | Open weights, competitive quality | Free (compute only) |
| Mistral Large 2 | Mistral | 128K | European, multilingual | ~$3/1M in, $9/1M out |
| DeepSeek V3 | DeepSeek | 128K | Open weights, strong coding | Free / API available |
| Qwen2.5 72B | Alibaba | 128K | Open weights, multilingual | Free (compute only) |

---

## Benchmark Performance (approximate, mid-2025)

### MMLU (General Knowledge, 5-shot)
| Model | Score |
|-------|-------|
| GPT-4o | 88.7% |
| Claude Opus 4 | ~90%+ |
| Gemini 1.5 Pro | 85.9% |
| Llama 3.1 405B | 88.6% |
| Mistral Large 2 | 84.0% |

### HumanEval (Python Coding, Pass@1)
| Model | Score |
|-------|-------|
| Claude Sonnet 4 | ~90%+ |
| GPT-4o | 90.2% |
| Gemini 1.5 Pro | 84.1% |
| Llama 3.1 405B | 89.0% |
| DeepSeek V3 | 91.6% |

### Context Window Comparison
| Model | Context Window | Practical Use |
|-------|---------------|---------------|
| Gemini 1.5 Pro | 1,000,000 tokens | Entire codebases, books |
| Claude 3/4 | 200,000 tokens | Long documents, large codebases |
| GPT-4o | 128,000 tokens | Long documents |
| Llama 3.1 | 128,000 tokens | Long documents |
| GPT-3.5 | 16,000 tokens | Standard tasks |

---

## Open-Source LLM Landscape

### Small Models (1-8B) — Run on CPU or consumer GPU
| Model | Params | Strengths |
|-------|--------|-----------|
| Llama 3.2 3B | 3B | Fast, multilingual, vision variant |
| Phi-3.5 Mini | 3.8B | Strong at reasoning for size |
| Gemma 2 2B | 2B | Google, strong for size |
| Qwen2.5 3B | 3B | Multilingual, code |
| SmolLM2 1.7B | 1.7B | Edge deployment |

### Medium Models (7-13B) — Single GPU (24GB)
| Model | Params | Strengths |
|-------|--------|-----------|
| Llama 3.1 8B | 8B | Best open 8B, 128K context |
| Mistral 7B v0.3 | 7B | Fast, good quality |
| Qwen2.5 7B | 7B | Strong coding, multilingual |
| Gemma 2 9B | 9B | Strong for size |
| DeepSeek-Coder-V2 Lite | 16B (MoE) | Coding specialist |

### Large Models (30-70B) — Multi-GPU
| Model | Params | Strengths |
|-------|--------|-----------|
| Llama 3.1 70B | 70B | Near-frontier open model |
| Qwen2.5 72B | 72B | Competitive with GPT-4 |
| DeepSeek V3 | 685B (MoE) | Open weights, top tier |
| Llama 3.3 70B | 70B | Strong instruction following |

---

## Embedding Models

| Model | Provider | Dim | Context | Notes |
|-------|----------|-----|---------|-------|
| text-embedding-3-large | OpenAI | 3072 | 8191 | Best quality |
| text-embedding-3-small | OpenAI | 1536 | 8191 | Good balance |
| voyage-3 | Anthropic | 1024 | 32K | Strong for RAG |
| voyage-3-lite | Anthropic | 512 | 32K | Fast, cheap |
| all-MiniLM-L6-v2 | SBERT | 384 | 512 | Free, fast |
| nomic-embed-text | Nomic | 768 | 8192 | Free, long context |
| bge-m3 | BAAI | 1024 | 8192 | Multilingual, free |

---

## Multimodal Models

| Model | Vision | Audio | Video | Notes |
|-------|--------|-------|-------|-------|
| GPT-4o | Yes | Yes | No | Real-time audio/vision |
| Claude 3/4 | Yes | No | No | Strong document analysis |
| Gemini 1.5 Pro | Yes | Yes | Yes | All modalities, 1M ctx |
| Llava-v1.6 | Yes | No | No | Open source, CLIP+LLM |
| LLaMA 3.2 Vision | Yes | No | No | Open source multimodal |
| Phi-3.5 Vision | Yes | No | No | Small vision model |
| Qwen2.5-VL | Yes | No | Yes | Strong OCR, video |

---

## Model Selection Decision Tree

```
Need open weights (self-host)?
├── Yes → LLaMA 3.1 70B, Qwen2.5 72B, DeepSeek V3
└── No → Continue

Latency-critical (< 500ms)?
├── Yes → Claude Haiku, GPT-4o mini, Gemini Flash
└── No → Continue

Long document (> 100K tokens)?
├── Yes → Gemini 1.5 Pro (1M ctx), Claude (200K ctx)
└── No → Continue

Cost-sensitive?
├── Yes → Claude Haiku, GPT-4o mini, Gemini Flash
└── No → Continue

Best quality needed?
└── Claude Opus 4.7, GPT-4o, Gemini 1.5 Pro (benchmark-specific)
```

---

## Key Trends to Know (2025)

1. **Reasoning models**: o1/o3 (OpenAI), Claude extended thinking — slow thinking for harder problems
2. **Long context**: 1M+ tokens enabling entire codebase in context
3. **Multimodal by default**: all frontier models process images/video
4. **Open models closing gap**: Llama 3.1, DeepSeek V3 competitive with GPT-4
5. **Efficient inference**: speculative decoding, quantization standard in production
6. **Agents and tool use**: function calling/tool use core API feature
7. **On-device models**: Phi-3, Llama 3.2 1B-3B for edge/mobile
