# AI System Design Cheatsheet

## Design Principles for ML Systems

### The Three Phases
1. **Offline**: Training, evaluation, experimentation
2. **Online**: Serving, real-time inference, A/B testing
3. **Monitoring**: Drift detection, retraining triggers, alerting

---

## Feature Stores

**Why:** Avoid feature computation duplication between training and serving (training-serving skew).

**Components:**
- **Offline store**: historical features for training (Hive, BigQuery, Parquet)
- **Online store**: low-latency features for serving (Redis, DynamoDB)
- **Feature registry**: metadata, definitions, lineage

**Key platforms:** Feast (open-source), Tecton (managed), AWS SageMaker Feature Store

**Common patterns:**
```
Raw Events → Streaming (Flink/Spark) → Online Store (Redis)
Raw Events → Batch (Spark) → Offline Store (Hive)
Training Job ← Offline Store
Serving ← Online Store (< 5ms lookup)
```

---

## Model Serving Architectures

### Online Inference
- **REST API** (FastAPI/Flask): simple, synchronous, good for < 1K QPS
- **gRPC**: binary protocol, lower latency, good for microservices
- **Model Server**: TorchServe, TF Serving, Triton Inference Server
  - Batching, multi-model, GPU management built-in

### Batch Inference
- Spark MLlib for distributed batch scoring
- SageMaker Batch Transform / Azure ML batch endpoints
- Use when: don't need real-time, score millions of records overnight

### Streaming Inference
- Kafka + Flink for continuous stream scoring
- Use when: event-driven, near-real-time (seconds not milliseconds)

### Serverless
- AWS Lambda, Cloud Run — pay-per-request, auto-scale to zero
- Good for: low traffic, spiky workloads, simple models

---

## Latency Optimization

| Technique | Latency Reduction | Tradeoff |
|-----------|------------------|----------|
| Model quantization (INT8) | 2-4x | Slight accuracy drop |
| Pruning | 2-5x | Accuracy drop, manual effort |
| Knowledge distillation | 5-10x | Training cost, accuracy gap |
| ONNX export | 1.5-2x | Portability, less flexibility |
| TensorRT | 2-4x | NVIDIA GPU only, complex setup |
| Batching | 5-10x throughput | Latency increase per request |
| Caching | 10-100x (cache hit) | Memory, staleness |
| Model parallelism | Enable large models | Engineering complexity |

---

## A/B Testing for ML Models

**Shadow mode:**
- New model runs in parallel, results not shown to users
- Compare predictions to ground truth later
- Zero risk, but no online feedback loop

**Canary deployment:**
- Route small % (1-5%) of traffic to new model
- Monitor business metrics closely
- Gradually increase if metrics hold

**A/B test:**
- Random split of users into control/treatment
- Statistical significance before concluding
- Minimum detectable effect, sample size calculation

**Multi-armed bandit:**
- Dynamically allocate traffic to better-performing variant
- Less total regret than fixed A/B split
- Use for continuous optimization (not one-time decision)

**Common mistakes:**
- Not waiting for statistical significance
- Multiple testing without correction (Bonferroni)
- Novelty effect (users behave differently when something changes)
- Interference between treatment/control (network effects)

---

## Monitoring & Observability

### What to Monitor

**Data Drift:**
- Input feature distribution shift
- Covariate shift: P(X) changes, P(Y|X) same
- Concept drift: P(Y|X) changes
- Detect with: KS test, PSI, Jensen-Shannon divergence

**Model Performance:**
- Prediction distribution (score histogram)
- Calibration (predicted prob vs actual frequency)
- Accuracy/AUC on labeled holdout (delayed labels)

**Infrastructure:**
- Latency percentiles (p50, p95, p99)
- Throughput (requests/second)
- Error rate
- Memory/CPU/GPU utilization

### Tools
- **Evidently AI**: open-source drift reports
- **Arize AI**: model observability platform
- **WhyLabs**: statistical monitoring
- **MLflow**: experiment tracking, model registry

---

## MLOps Maturity Levels

**Level 0:** Manual training, no versioning, notebook-to-production
- Risk: no reproducibility, slow iteration

**Level 1:** ML pipeline automation, feature store, model registry
- Automated training, but manual deployment triggers

**Level 2:** Full CI/CD for ML, automated retraining, shadow deployment
- Code change → test → train → evaluate → deploy automatically

---

## Data Architecture Patterns

### Lambda Architecture
- **Batch layer**: recomputes all features from raw data periodically
- **Speed layer**: processes real-time events, approximate
- **Serving layer**: merges batch and speed
- Con: two codepaths to maintain

### Kappa Architecture
- Single streaming pipeline for both real-time and historical
- Reprocess historical by replaying events (Kafka)
- Simpler but requires replayable event store

---

## Common Interview Questions

**Q: How would you handle model degradation in production?**
- Monitor prediction distribution and upstream data distributions
- Alert on PSI > 0.2 or KS p-value < 0.05
- Scheduled retraining with validation gate
- Champion/challenger setup for safe rollout

**Q: How do you prevent training-serving skew?**
- Use a feature store with same transformation code for training and serving
- Log features at serving time, use them for training
- Integration tests that compare offline/online feature values

**Q: How do you handle high-cardinality categoricals?**
- Target encoding with cross-validation (prevents leakage)
- Embedding layers (learned representations)
- Hashing trick for very high cardinality
- Frequency encoding or binning

**Q: When would you use a simpler model vs a complex one?**
- Simpler: interpretability required (healthcare, finance), small data, fast iteration needed
- Complex: lots of data, performance is critical, interpretability not required, unstructured data (images, text)
