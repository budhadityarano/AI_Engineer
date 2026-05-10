# ML System Design — Interview Cheatsheet

## Framework for ML System Design Interviews

**RADIO Framework:**
1. **R**equirements — functional + non-functional
2. **A**rchitecture — high-level components
3. **D**ata — collection, storage, pipelines
4. **I**nference — online vs offline, latency, throughput
5. **O**ptimization — model improvements, monitoring, A/B tests

---

## Common ML System Design Problems

### 1. Recommendation System (e.g., Netflix, Amazon)

**Requirements:**
- Personalized recommendations, real-time or near-real-time
- Handle cold start (new users/items)
- Scale to millions of users and items

**Architecture:**
```
User → API Gateway → Candidate Generation → Ranking → Re-ranking → Results
                           │                   │
                    Embedding Store      Scoring Model
                    (Approximate NN)     (Two-Tower / DLRM)
```

**Key Decisions:**
- Candidate generation: matrix factorization (ALS), two-tower model, ItemKNN
- Ranking: gradient boosting (XGBoost) or deep model (DLRM, DIN)
- Feature store for real-time user/item features (Redis/Feast)
- Approximate nearest neighbor for embedding retrieval (FAISS/Pinecone)

**Cold Start:** Popularity-based fallback → content-based → collaborative filtering

---

### 2. Fraud Detection System

**Requirements:**
- < 100ms latency for real-time transaction scoring
- High recall (miss no fraud) with acceptable precision
- Evolving fraud patterns (concept drift)

**Architecture:**
```
Transaction → Feature Extraction → Model Scoring → Rules Engine → Decision
     │                │                                │
  Raw Events    Feature Store              Fraud Analysts (FP review)
  (Kafka)       (Redis/Feast)
```

**Models:**
- Real-time: Logistic Regression / LightGBM (< 10ms)
- Batch: Graph Neural Network (catch fraud rings)
- Anomaly detection: Isolation Forest / Autoencoder

**Features:**
- Transaction velocity (last 1h/24h/7d)
- Merchant category, amount percentile
- Device fingerprint, geolocation distance
- Graph features: shared devices/emails/cards

**Monitoring:** Precision/Recall daily, feature drift, model score distribution

---

### 3. Search Ranking System

**Requirements:**
- Rank 1000+ candidates in < 50ms
- Relevance + personalization + freshness

**Architecture:**
```
Query → Query Understanding → Retrieval → Feature Computation → Ranking → Results
          │                    │                                  │
       NLU/NER             BM25 + Dense              LambdaMART / DNN
       Query Expansion      (Hybrid ANN)               (Pointwise/Pairwise)
```

**Retrieval (recall optimization):**
- BM25 for exact match (keyword)
- Dense retrieval (bi-encoder) for semantic
- Hybrid with RRF

**Ranking (precision optimization):**
- Learning-to-rank (LambdaMART, RankNet)
- NDCG loss, pairwise training
- Features: BM25 score, dense score, CTR, freshness, authority

---

### 4. Real-Time Ad Click-Through Rate (CTR) Prediction

**Requirements:**
- < 10ms inference (per impression)
- 10B+ impressions/day
- Sparse categorical features (user ID, ad ID, publisher ID)

**Architecture:**
```
Impression → Feature Lookup → CTR Model → Bid Adjustment
                │                 │
          Feature Store      DLRM / Wide&Deep
          (Redis O(1))       (Embedding + MLP)
```

**Model:** Wide & Deep, DeepFM, DLRM
- Wide part: memorization (feature crosses)
- Deep part: generalization (embeddings + MLP)

**Training:** Mini-batch SGD on streaming logs, daily refresh
**Serving:** Quantized model (INT8), batched inference

---

### 5. LLM-Powered Feature (e.g., Code Copilot)

**Requirements:**
- < 500ms latency for code completion
- Cost per request < $0.01

**Architecture:**
```
User → API → Cache → Router → [Small Model | Large Model] → Response
                        │
               Complexity Classifier
               (route simple → small, complex → large)
```

**Optimizations:**
- Prompt caching (Anthropic extended caching)
- Speculative decoding (draft model + verifier)
- KV cache for repeated system prompts
- RAG for codebase context injection

---

## Non-Functional Requirements Cheatsheet

| Requirement | Typical Target | How to Achieve |
|-------------|---------------|----------------|
| Latency (p99) | < 100ms | Model quantization, caching, ANN |
| Throughput | 10K+ QPS | Horizontal scaling, batching |
| Availability | 99.9% | Multi-region, circuit breakers |
| Freshness | < 1hr | Online learning, feature streaming |
| Consistency | Eventual OK | Async replication |

---

## Data Pipeline Design

```
Raw Data (S3/GCS)
    → ETL (Spark/DBT)
    → Feature Store (offline: Hive/BigQuery, online: Redis)
    → Training Pipeline (scheduled: Prefect/Airflow)
    → Model Registry (MLflow)
    → Shadow Deployment
    → A/B Test
    → Production
    → Monitoring (Evidently, Arize)
    → Drift Alert → Retrain Trigger
```

---

## Tradeoffs to Discuss

1. **Offline vs Online learning**: offline = stable, batch; online = fresh, complex
2. **Model complexity vs latency**: bigger model = better accuracy, higher latency
3. **Precision vs Recall**: tune threshold based on business cost of FP vs FN
4. **Single model vs ensemble**: ensemble = better accuracy, harder to debug
5. **Cloud-managed vs self-hosted**: managed = less ops, higher cost, less control

---

## Things Interviewers Want to Hear

- Start with requirements before architecture
- State your assumptions explicitly
- Discuss failure modes (what if model is down? data pipeline fails?)
- Mention monitoring/evaluation strategy
- Propose a phased rollout (shadow → A/B → full)
- Know the difference between model metrics (AUC) and business metrics (revenue)
