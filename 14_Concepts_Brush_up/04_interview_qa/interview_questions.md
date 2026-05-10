# 100 AI/ML Engineering Interview Q&A

## Machine Learning Fundamentals (1–25)

**1. What is the bias-variance tradeoff?**
Bias = error from oversimplified model (underfitting). Variance = error from model too sensitive to training data (overfitting). Total error = Bias² + Variance + Noise. Increasing model complexity reduces bias but increases variance.

**2. What is regularization and why do we use it?**
Adds a penalty term to the loss to discourage large weights, reducing overfitting. L1 (Lasso) promotes sparsity; L2 (Ridge) shrinks all weights. Prevents the model from memorizing training data.

**3. Explain precision, recall, and F1.**
Precision = TP/(TP+FP) — accuracy of positive predictions. Recall = TP/(TP+FN) — coverage of actual positives. F1 = harmonic mean of both. Use F1 when false positives and negatives are equally costly.

**4. When would you use AUC-ROC vs PR-AUC?**
AUC-ROC is better when classes are balanced. PR-AUC is better for imbalanced datasets because ROC can be misleadingly optimistic when negatives dominate.

**5. What is cross-validation and why use k-fold?**
Evaluates model performance on multiple train/test splits. k-fold: split into k parts, train on k-1, test on 1, rotate. Reduces evaluation variance vs a single holdout split.

**6. Explain gradient descent and its variants.**
Iteratively updates parameters in direction of negative gradient. Batch GD: all data per step. SGD: one sample. Mini-batch: subset. Adam: adaptive learning rates using first and second moments.

**7. What is the vanishing gradient problem?**
In deep networks, gradients shrink exponentially during backpropagation, making early layers learn very slowly. Fixed by: ReLU activations, skip connections (ResNet), LSTM gating, batch normalization.

**8. How does Random Forest differ from a single decision tree?**
Random Forest builds N trees on bootstrap samples, each using a random subset of features. Averages predictions. Reduces variance (overfitting) compared to a single deep tree. Loses interpretability.

**9. What is gradient boosting? How does XGBoost improve it?**
Sequential ensemble: each tree corrects residuals of the previous. XGBoost adds: regularization terms in objective, second-order gradient optimization, column subsampling, sparse awareness, approximate tree learning.

**10. Explain the difference between bagging and boosting.**
Bagging: parallel independent models on bootstrap samples, reduces variance (e.g., Random Forest). Boosting: sequential dependent models weighted by error, reduces bias (e.g., XGBoost, AdaBoost).

**11. What is k-means clustering? What are its limitations?**
Partitions data into k clusters by minimizing within-cluster sum of squared distances. Limitations: requires k to be specified, assumes spherical clusters, sensitive to outliers and initialization.

**12. How does PCA work?**
Finds orthogonal directions (principal components) of maximum variance using SVD of the covariance matrix. Projects data onto the top components. Used for dimensionality reduction and visualization.

**13. What is the curse of dimensionality?**
In high dimensions: data becomes sparse, distances become meaningless, volume of space grows exponentially. More data needed to cover the space. Mitigate with: PCA, feature selection, regularization.

**14. What is overfitting? How do you detect and prevent it?**
Detect: large gap between train and validation accuracy. Prevent: regularization (L1/L2/dropout), more data, simpler model, early stopping, cross-validation, data augmentation.

**15. Explain the ROC curve.**
Plots TPR (recall) vs FPR as the classification threshold varies. AUC = 0.5 means random, 1.0 means perfect. Threshold choice depends on business cost of FP vs FN.

**16. What is class imbalance? How do you handle it?**
When one class has many more samples. Strategies: resample (SMOTE oversampling, undersampling), class weights in loss function, threshold tuning, use PR-AUC instead of accuracy.

**17. What is feature importance? How is it computed in Random Forest?**
Measures each feature's contribution to predictions. RF: average decrease in impurity (Gini) across trees when splitting on that feature. Also: permutation importance (shuffle feature, measure performance drop).

**18. Explain the difference between generative and discriminative models.**
Generative: models P(X,Y) or P(X|Y) (e.g., Naive Bayes, VAE, GPT). Can generate new samples. Discriminative: models P(Y|X) directly (e.g., Logistic Regression, SVM). Usually better classification accuracy.

**19. What is the EM algorithm?**
Expectation-Maximization: alternates between E-step (compute expected log-likelihood given current parameters) and M-step (maximize expected log-likelihood). Used for Gaussian Mixture Models, HMMs.

**20. What is a support vector machine?**
Finds the hyperplane that maximizes the margin between classes. Kernel trick maps data to higher dimensions for non-linear separation. C parameter controls regularization.

**21. How does Naive Bayes work? When is it appropriate?**
Applies Bayes theorem assuming feature independence: P(Y|X) ∝ P(Y)·ΠP(X_i|Y). Fast, works well with small data and high-dimensional features (text). Inappropriate when features are correlated.

**22. What is the difference between parametric and non-parametric models?**
Parametric: fixed number of parameters regardless of data size (Linear Regression, LDA). Non-parametric: parameters grow with data (KNN, kernel methods). Non-parametric more flexible, requires more data.

**23. What is multicollinearity and how do you handle it?**
When features are highly correlated, making coefficient estimates unstable in linear models. Detect: VIF > 10. Handle: remove correlated features, PCA, ridge regression, dimensionality reduction.

**24. What is model calibration?**
A well-calibrated model's predicted probability matches actual frequency. E.g., for 100 samples with P=0.7, ~70 should be positive. Calibrate with Platt scaling (logistic regression on outputs) or isotonic regression.

**25. When would you use Logistic Regression over a tree-based model?**
Logistic Regression: need interpretability, sparse/high-dim features (text), fast training, want probability outputs. Tree-based: non-linear relationships, mixed types, feature interactions, tabular data.

---

## Deep Learning (26–50)

**26. Explain backpropagation.**
Computes gradients of loss with respect to each parameter using chain rule. Forward pass computes activations; backward pass propagates gradients from output to input. Enabled by automatic differentiation.

**27. What is batch normalization? Why does it help?**
Normalizes activations to zero mean/unit variance within a mini-batch, then scales/shifts with learnable γ and β. Reduces internal covariate shift, allows higher learning rates, acts as regularizer.

**28. What is dropout? How does it work during inference?**
Randomly zeros activations with probability p during training. Forces redundant representations. At inference, multiply by (1-p) to keep expected value same (or divide during training — inverted dropout).

**29. What is the difference between LSTM and GRU?**
LSTM has 3 gates (forget, input, output) + separate cell state. GRU has 2 gates (reset, update), merges cell/hidden state. GRU is faster and simpler; LSTM has slightly more expressive power.

**30. Explain the attention mechanism.**
Computes: Attention(Q,K,V) = softmax(QK^T/√d_k)·V. Allows model to focus on relevant positions. Queries look up Keys to produce attention weights, then weighted sum of Values.

**31. What is the difference between self-attention and cross-attention?**
Self-attention: Q, K, V all come from same sequence (e.g., encoder attending to itself). Cross-attention: Q from one sequence, K/V from another (e.g., decoder attending to encoder output).

**32. Why is positional encoding needed in Transformers?**
Attention is permutation-invariant — it doesn't know token order. Positional encoding injects position information using sinusoids: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(...).

**33. What is the difference between BERT and GPT?**
BERT: encoder-only, bidirectional, pre-trained with masked LM. Good for understanding tasks (classification, NER). GPT: decoder-only, causal (left-to-right), pre-trained with next-token prediction. Good for generation.

**34. What is transfer learning?**
Reuse weights from a model trained on one task for a different task. Fine-tuning: adapt all or some pretrained weights. Feature extraction: freeze pretrained weights, train new head only.

**35. What is knowledge distillation?**
Train a small "student" model to mimic a large "teacher" model's output probabilities (soft labels), not just hard labels. Student learns teacher's dark knowledge about class relationships.

**36. What is LoRA? How does it reduce training parameters?**
Low-Rank Adaptation: freeze pretrained weights W, inject trainable matrices B·A where rank r << min(d,k). Update: W' = W + (α/r)·B·A. Reduces parameters from d·k to r·(d+k).

**37. Explain QLoRA.**
Combines 4-bit NF4 quantization of base model weights with LoRA adapters. Enables fine-tuning of 70B parameter models on a single 48GB GPU. Quantized base weights are dequantized during forward pass.

**38. What is gradient clipping?**
Cap gradient norm at a maximum value before parameter update: if ‖g‖ > max_norm, g ← g · max_norm/‖g‖. Prevents exploding gradients. Critical for RNNs, recommended for transformers.

**39. What is label smoothing?**
Replace one-hot targets with (1-ε) for true class, ε/(K-1) for others. Prevents overconfidence, improves calibration. Common ε = 0.1.

**40. What is weight decay? How does it differ from L2 regularization?**
In SGD: equivalent — L2 penalty gradient is λw, same as weight decay. In Adam: NOT equivalent. L2 adds λw to gradient (scaled by adaptive lr). Weight decay directly multiplies weights by (1-λ), independent of gradient magnitude. AdamW implements true weight decay.

**41. What is a residual connection (skip connection)?**
Adds input directly to output of a layer: y = F(x) + x. Allows gradients to flow directly through skip path. Enables training very deep networks. Key innovation in ResNet.

**42. What is layer normalization vs batch normalization?**
Batch norm: normalizes across batch dimension for each feature. Depends on batch size, not used during inference without running statistics. Layer norm: normalizes across feature dimension for each sample. Works with any batch size, preferred for transformers and NLP.

**43. What is beam search?**
At each decoding step, keeps top-k (beam width) candidate sequences instead of greedy 1-best. Trades compute for better generation quality. Larger beam = more diverse but slower.

**44. What is temperature in LLM sampling?**
Divides logits by T before softmax. T < 1: sharper distribution, more deterministic. T > 1: flatter distribution, more random. T = 0: argmax (greedy). T = 1: standard sampling.

**45. What is the difference between model parallelism and data parallelism?**
Data parallelism: replicate model on N GPUs, split batch. Gradient all-reduce to synchronize. Model parallelism: split model across GPUs (e.g., tensor parallelism, pipeline parallelism). Needed when model > 1 GPU memory.

**46. What is mixed precision training?**
Train with FP16 weights for speed/memory, keep FP32 master copy for precision. Loss scaling prevents FP16 underflow. 2x speedup on modern GPUs. PyTorch: torch.cuda.amp.

**47. What is the perplexity of a language model?**
Perplexity = exp(-1/N · Σ log P(w_t|w_{t-1},...,w_1)). Geometric mean of inverse probability. Lower = better. Perplexity 10 means model is as uncertain as uniform over 10 choices at each step.

**48. What are emergent abilities in LLMs?**
Capabilities that appear suddenly at certain scale thresholds (not gradual improvements). Examples: few-shot learning, chain-of-thought reasoning, instruction following. Not fully understood — debated whether truly emergent or just phase transitions.

**49. What is RLHF?**
Reinforcement Learning from Human Feedback: fine-tune LLM with PPO using a reward model trained on human preference rankings. Makes model helpful, harmless, honest. Constitutional AI (Anthropic) uses AI feedback instead.

**50. What is speculative decoding?**
Use small draft model to generate k tokens speculatively, then verify with large model in parallel. Accept matching tokens, regenerate from first mismatch. Achieves same quality at 2-3x speed.

---

## RAG & Agents (51–70)

**51. What is RAG and when should you use it?**
Retrieval-Augmented Generation: retrieve relevant documents at inference time and inject into LLM context. Use when: knowledge needs to be up-to-date, domain-specific, verifiable, or too large to fine-tune.

**52. What are the main failure modes in RAG?**
(1) Retrieval failure: relevant docs not retrieved (improve embeddings, chunking, hybrid search). (2) Context poisoning: wrong docs retrieved. (3) Faithfulness failure: LLM ignores retrieved context or hallucinates.

**53. What is chunking strategy? Why does it matter?**
How documents are split before embedding. Too small = loses context. Too large = dilutes relevant signal. Strategies: fixed-size, sentence splitter, semantic chunking, hierarchical (parent-child).

**54. What is hybrid retrieval?**
Combines sparse (BM25, keyword) and dense (embedding) retrieval. BM25 handles exact keyword matches; dense handles semantic similarity. Merge results with Reciprocal Rank Fusion (RRF).

**55. What is HyDE?**
Hypothetical Document Embeddings: instead of embedding the query, generate a hypothetical answer first (using LLM), then embed that. Bridges the query-document distribution gap.

**56. What is a cross-encoder? How does it differ from a bi-encoder?**
Bi-encoder: encodes query and doc separately → fast, precompute doc embeddings. Cross-encoder: encodes (query, doc) pair jointly → more accurate but slow (can't precompute). Use bi-encoder for retrieval, cross-encoder for re-ranking.

**57. What are RAGAS metrics?**
Faithfulness: are answer claims supported by context? Answer Relevancy: does answer address the question? Context Recall: is ground truth info in retrieved context? Context Precision: is retrieved context useful?

**58. What is tool calling in LLM agents?**
Model outputs structured function call (name + arguments) instead of text. Runtime executes function, returns result to model. Allows LLMs to interact with external systems (search, databases, APIs).

**59. What is the ReAct pattern?**
Interleaves Reasoning (Thought) and Acting (tool calls) in a loop. Thought: reasoning about what to do. Action: tool call. Observation: tool result. Repeat until final answer.

**60. What is the supervisor agent pattern?**
Orchestrator LLM decomposes task and delegates to specialized sub-agents. Collects results and synthesizes final answer. Enables parallel and sequential multi-agent workflows.

**61. What is prompt injection?**
Malicious input that overrides the system prompt or hijacks agent behavior. E.g., retrieved document containing "Ignore previous instructions and...". Mitigate: input sanitization, privilege separation, output validation.

**62. What is the difference between LangChain and LangGraph?**
LangChain: LCEL chains for linear pipelines (prompt → LLM → parser). LangGraph: stateful graph with nodes, edges, and conditional routing. LangGraph is better for agents with branching, loops, and state.

**63. How do you evaluate RAG quality without ground truth?**
LLM-as-judge for faithfulness and relevancy (no ground truth needed). Monitor retrieval quality via prediction distribution stability. Use online A/B test on user satisfaction signals.

**64. What is contextual compression?**
Extract only the relevant sentences from a retrieved document before passing to LLM. Reduces context window usage, lowers cost, reduces distraction from irrelevant content.

**65. What is multi-query retrieval?**
Generate N different phrasings of the query, retrieve for each, merge results. Increases recall at cost of N× retrieval calls. Each phrasing hits different relevant documents.

**66. What is a vector database? Name three.**
Database optimized for storing and querying high-dimensional embeddings using approximate nearest neighbor search. Examples: Pinecone, Weaviate, Qdrant, Chroma, FAISS (library), pgvector (PostgreSQL extension).

**67. What is the difference between fine-tuning and RAG?**
Fine-tuning: bakes knowledge into model weights. Needs data collection, training compute, updates require retraining. RAG: retrieves knowledge at inference time. Dynamic, no retraining, but adds latency and retrieval complexity.

**68. What is semantic chunking?**
Split documents at semantic boundaries rather than fixed character counts. Use sentence embeddings to detect topic shifts. Produces chunks with coherent meaning rather than arbitrary splits.

**69. What is HNSW?**
Hierarchical Navigable Small World graph. ANN algorithm with O(log N) search complexity. Builds multilayer proximity graph. Dominant algorithm in production vector DBs (Pinecone, Weaviate, FAISS).

**70. What is the lost-in-the-middle problem?**
LLMs perform worse when relevant information is in the middle of a long context vs at the beginning or end. Mitigation: reranking to put most relevant docs at edges, shorter contexts.

---

## MLOps & Production (71–85)

**71. What is model drift? What are the types?**
Data drift: P(X) changes. Concept drift: P(Y|X) changes. Label drift: P(Y) changes. Detect with: KS test, PSI, Jensen-Shannon divergence on feature distributions, monitoring prediction distribution.

**72. What is the difference between online and offline evaluation?**
Offline: evaluate on historical holdout set before deployment. Online: A/B test in production using business metrics. Offline metrics don't always correlate with online metrics.

**73. What is shadow mode deployment?**
Run new model in parallel without showing results to users. Compare predictions to ground truth and to champion model. Zero-risk, identifies issues before live traffic.

**74. What is feature store and why is it important?**
Centralized system for storing and serving ML features. Prevents training-serving skew by using same transformation logic in both. Enables feature reuse across teams. Examples: Feast, Tecton, SageMaker Feature Store.

**75. What is MLflow? What can it track?**
Open-source MLOps platform. Tracks: parameters, metrics, artifacts (models, plots). Manages experiment comparison, model registry, model serving. Central hub for ML lifecycle management.

**76. What is A/B testing for ML? What are common mistakes?**
Randomly split users into control/treatment, compare metrics. Mistakes: insufficient sample size, multiple testing without correction, novelty effect, not isolating variables, peeking at results early.

**77. What is the training-serving skew problem?**
When features computed differently during training vs serving, causing degraded model performance. Fix: use feature store, log serving features for training, integration tests comparing offline/online feature values.

**78. What is canary deployment?**
Route small % of traffic (1-5%) to new model version. Monitor closely. Gradually increase. Limit blast radius if new model has issues.

**79. What is containerization and why is it important for ML?**
Docker packages model + dependencies + code into portable image. Guarantees same environment in development and production. Enables easy scaling and deployment across clouds.

**80. What is Kubernetes relevant to ML?**
Container orchestration platform. Manages model serving deployments: auto-scaling, load balancing, rolling updates, resource allocation (CPU/GPU). Foundation for most production ML infrastructure.

**81. What is data versioning? Name a tool.**
Track dataset changes like code changes. Enables reproducibility (reproduce exact training dataset). DVC (Data Version Control) — Git for data, stores data in S3/GCS, tracks with Git.

**82. What is a model registry?**
Central repository for versioned models. Tracks: model artifacts, metadata, training run, metrics, lineage. Manages staging → production promotion. MLflow, Weights & Biases, Vertex AI all have registries.

**83. How do you handle model rollback?**
Keep previous model version in registry. Blue-green deployment: switch traffic between old (blue) and new (green). Canary rollback: reduce new version traffic to 0%. Always preserve champion model.

**84. What is concept drift and how do you detect it?**
Model accuracy degrades because the relationship between features and target changed. Detect by: monitoring model performance on recent labeled data, monitoring prediction distribution, monitoring feature distributions.

**85. What is the difference between ETL and ELT?**
ETL: Extract → Transform → Load. Transform before loading into warehouse. ELT: Extract → Load → Transform. Load raw data, transform inside warehouse (BigQuery, Redshift). Modern trend: ELT for flexibility.

---

## Recent AI Developments (86–100)

**86. What is in-context learning?**
LLM learns a task from examples in the prompt without gradient updates. Mechanism not fully understood. Few-shot examples provide task demonstrations. Performance scales with model size.

**87. What is chain-of-thought prompting?**
Prompting LLMs to "think step by step" before answering. Unlocks multi-step reasoning. Works with examples (few-shot CoT) or zero-shot ("Let's think step by step").

**88. What is Constitutional AI?**
Anthropic's approach to RLHF without human feedback for every response. Model uses a constitution (principles) to self-critique and revise responses. Claude trained with CAI.

**89. What is a mixture of experts (MoE)?**
Model has N expert sub-networks. Router selects top-k experts per token. Only those experts are activated. Increases model capacity without proportional compute increase. Used in GPT-4, Mixtral.

**90. What are long context models? What's the challenge?**
Models that support very long inputs (100K-1M tokens). Challenges: quadratic attention cost (mitigated by FlashAttention), memory, and the lost-in-the-middle problem. Examples: Claude (200K), Gemini (1M).

**91. What is FlashAttention?**
Exact attention algorithm that's memory-efficient by tiling computation to avoid materializing full N×N attention matrix. Trades compute for memory: O(N) memory vs O(N²). Key enabler for long contexts.

**92. What is multimodal AI?**
Models that process multiple input modalities (text + image + audio + video). Examples: GPT-4V, Claude 3 (image), Gemini (text+image+video). Architecture: vision encoder → projection → LLM.

**93. What is CLIP?**
Contrastive Language-Image Pre-training (OpenAI). Trains image encoder and text encoder to have similar embeddings for matching pairs. Enables zero-shot classification by comparing image embedding to class text embeddings.

**94. What is diffusion in image generation?**
Trains model to progressively denoise images. Forward process: add Gaussian noise step by step. Reverse process: learn to denoise. At generation: start from pure noise, denoise step by step. Examples: Stable Diffusion, DALL-E.

**95. What is the context window and how does it affect LLM behavior?**
Maximum sequence length model can process. Longer context: can process more information but costs more (quadratic in standard attention). Models can lose focus on information in the middle of long contexts.

**96. What is function calling vs tool use?**
Function calling (OpenAI): model outputs JSON specifying function name and arguments. Tool use (Anthropic): similar mechanism with `tool_use` content blocks. Both allow LLMs to interact with external systems.

**97. What is a system prompt?**
Instructions prepended to a conversation that set the assistant's behavior, persona, and constraints. Not visible in messages array. Used to customize LLM for specific applications.

**98. What is prompt caching?**
Cache reusable portions of the prompt (system prompt, documents) across requests. Reduces latency and cost for repeated contexts. Anthropic: up to 1hr TTL, charged at cache_read_input_tokens rate (much lower).

**99. What is the difference between zero-shot, few-shot, and fine-tuning?**
Zero-shot: task description only, no examples. Few-shot: 3-10 labeled examples in prompt. Fine-tuning: gradient updates on labeled dataset. Cost/effort: zero-shot < few-shot << fine-tuning. Quality generally reverses.

**100. What is the AI engineering vs ML engineering distinction?**
ML engineering: focus on training models, feature engineering, model evaluation. AI engineering: focus on building applications using pre-trained models (LLMs), prompt engineering, RAG pipelines, agent frameworks. Growing distinction as foundation models become the norm.
