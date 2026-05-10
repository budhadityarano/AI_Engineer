# Key AI/ML Papers — TL;DR Summaries

## Foundation Models & LLMs

### 1. Attention Is All You Need (Vaswani et al., 2017)
**TL;DR:** Introduced the Transformer — self-attention replaces RNNs entirely. Encoder-decoder with multi-head attention, positional encoding, and feed-forward sublayers. Enabled parallelization of sequence processing.
**Key contribution:** Multi-head self-attention + "Attention(Q,K,V) = softmax(QK^T/√d_k)·V"
**Impact:** Foundation of all modern LLMs, vision transformers, and multimodal models.

### 2. BERT (Devlin et al., 2018)
**TL;DR:** Bidirectional encoder pretrained with masked language modeling (MLM) and next sentence prediction (NSP). Fine-tuned on downstream tasks with minimal architecture changes.
**Key contribution:** Bidirectional context — unlike GPT which only sees left context.
**Impact:** Established pretraining+fine-tuning paradigm for NLP.

### 3. GPT-3 (Brown et al., 2020)
**TL;DR:** 175B parameter autoregressive model. Demonstrated in-context learning (few-shot without gradient updates) and emergent capabilities at scale.
**Key contribution:** Few-shot learning from demonstrations in the prompt alone.
**Impact:** Shifted focus to scale and in-context learning. Enabled ChatGPT.

### 4. InstructGPT / RLHF (Ouyang et al., 2022)
**TL;DR:** Fine-tuned GPT-3 with human feedback using PPO. Reward model trained on human preference rankings. Results: much more helpful and less harmful than vanilla GPT-3 despite smaller size.
**Key contribution:** RLHF pipeline: supervised fine-tuning → reward model → RL fine-tuning.
**Impact:** Basis of all modern assistant models (Claude, ChatGPT, Gemini).

### 5. Constitutional AI (Bai et al., 2022 — Anthropic)
**TL;DR:** Replaces human feedback for harmlessness with AI self-critique using a constitution (set of principles). Two phases: supervised learning from revised outputs, then RL from AI feedback (RLAIF).
**Key contribution:** RLAIF reduces reliance on human feedback for harmlessness.
**Impact:** Claude's training methodology.

### 6. LLaMA (Touvron et al., 2023)
**TL;DR:** Open-source 7B-65B models that match GPT-3 quality with less compute. Key improvements: RMSNorm instead of LayerNorm, SwiGLU activation, rotary positional embeddings.
**Key contribution:** Competitive open-weight LLMs enabling research and fine-tuning.
**Impact:** Sparked open-source LLM ecosystem (Alpaca, Vicuna, Mistral).

### 7. Mistral 7B (Jiang et al., 2023)
**TL;DR:** 7B model outperforming LLaMA 2 13B on most benchmarks. Key: grouped-query attention (GQA) for faster inference, sliding window attention for efficiency.
**Key contribution:** GQA reduces KV cache memory by sharing keys/values across heads.
**Impact:** Demonstrated that architecture and training quality beats raw scale.

---

## Efficient Training & Inference

### 8. LoRA (Hu et al., 2021)
**TL;DR:** Low-Rank Adaptation: freeze pretrained weights W, inject trainable B·A where r << min(d,k). Update: W' = W + (α/r)·B·A. < 1% parameters trained.
**Key insight:** Weight updates during fine-tuning have low intrinsic rank.
**Impact:** Standard method for fine-tuning large models efficiently.

### 9. QLoRA (Dettmers et al., 2023)
**TL;DR:** Combines 4-bit NF4 quantization (NormalFloat) with LoRA. Paged optimizers handle memory spikes. Enables 65B model fine-tuning on 1× A100 48GB GPU.
**Key contribution:** NF4 data type is information-theoretically optimal for normally distributed weights.
**Impact:** Democratized LLM fine-tuning on consumer hardware.

### 10. FlashAttention (Dao et al., 2022)
**TL;DR:** Exact attention with O(N) memory (vs O(N²)) by tiling computation. Never materializes full N×N matrix. IO-aware algorithm exploiting GPU memory hierarchy.
**Key contribution:** 2-4x attention speedup, 5-20x memory reduction.
**Impact:** Enabled training on longer sequences. Standard in all modern training stacks.

### 11. Mixture of Experts (Shazeer et al., 2017 + Jiang et al., 2024 Mixtral)
**TL;DR:** Replace FFN with N expert FFNs, route each token to top-k experts. Mixtral 8x7B: only 2 experts active per token → 13B active, 47B total params.
**Key contribution:** Scale parameters without proportional compute increase.
**Impact:** Presumed architecture of GPT-4. Open-source: Mixtral, DeepSeek-MoE.

---

## Vision & Multimodal

### 12. CLIP (Radford et al., 2021 — OpenAI)
**TL;DR:** Jointly trains image encoder (ViT) and text encoder (Transformer) with contrastive loss on 400M (image, caption) pairs. Zero-shot classification by comparing image embedding to class text embeddings.
**Key contribution:** Vision-language alignment via contrastive pretraining.
**Impact:** Backbone of most VLMs and text-to-image models.

### 13. Vision Transformer / ViT (Dosovitskiy et al., 2020)
**TL;DR:** Split image into 16×16 patches, treat each as a token, apply standard Transformer encoder. Beats CNNs on ImageNet at scale.
**Key contribution:** Pure self-attention for vision — no convolutions needed.
**Impact:** Unified architecture for vision and language.

### 14. LLaVA (Liu et al., 2023)
**TL;DR:** Connect CLIP image encoder to LLaMA with a linear projection layer. Fine-tune on instruction-following vision-language data (GPT-4 generated). Competitive with GPT-4V on some benchmarks.
**Key contribution:** Simple, efficient visual instruction tuning.
**Impact:** Open-source foundation for multimodal assistants.

### 15. Stable Diffusion (Rombach et al., 2022)
**TL;DR:** Latent diffusion model: run diffusion in compressed latent space (via VAE), not pixel space. 4-8× cheaper than pixel-space diffusion. CLIP text conditioning for text-to-image.
**Key contribution:** Latent space diffusion for computational efficiency.
**Impact:** Open-source text-to-image generation democratized.

---

## RAG & Agents

### 16. RAG (Lewis et al., 2020)
**TL;DR:** Combines parametric (model weights) and non-parametric (retrieved documents) memory. Retriever finds relevant docs, generator conditions on them. Outperforms pure parametric models on knowledge-intensive tasks.
**Key contribution:** Retrieval augmentation for open-domain QA.
**Impact:** Foundation of modern RAG systems.

### 17. REALM (Guu et al., 2020)
**TL;DR:** Jointly trains retriever and language model end-to-end. Retriever is differentiable using MIPS. Knowledge embedded in retrieved documents, not model parameters.
**Key contribution:** End-to-end trainable retrieval-language model.
**Impact:** Theoretical foundation for learned retrieval in RAG.

### 18. ReAct (Yao et al., 2022)
**TL;DR:** Interleave reasoning (Thought) and acting (tool calls) in LLM generation. Model synergizes thinking with tool use. Outperforms pure chain-of-thought and pure acting.
**Key contribution:** Thought-Action-Observation loop for agents.
**Impact:** Standard agent pattern in LangChain, LangGraph, and production agents.

---

## Evaluation

### 19. MMLU (Hendrycks et al., 2020)
**TL;DR:** Massive Multitask Language Understanding: 57 subjects from STEM to humanities. Multiple-choice questions requiring broad knowledge. Now standard LLM benchmark.
**Key contribution:** Comprehensive knowledge benchmark across domains.
**Impact:** Primary benchmark for comparing GPT-4, Claude, Gemini capabilities.

### 20. HumanEval (Chen et al., 2021 — OpenAI)
**TL;DR:** 164 handwritten Python programming problems with unit tests. Pass@k measures probability of generating at least one correct solution in k attempts.
**Key contribution:** Execution-based code generation benchmark.
**Impact:** Standard for measuring LLM coding ability (Copilot, CodeLlama).
