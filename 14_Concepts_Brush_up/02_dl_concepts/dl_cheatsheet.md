# Deep Learning Concepts Cheatsheet

## Neural Network Fundamentals

### Activation Functions

| Function | Formula | Range | Use When | Issue |
|----------|---------|-------|----------|-------|
| Sigmoid | 1/(1+e^-x) | (0,1) | Binary output | Vanishing gradient |
| Tanh | (e^x-e^-x)/(e^x+e^-x) | (-1,1) | RNNs | Vanishing gradient |
| ReLU | max(0,x) | [0,∞) | Default hidden | Dying ReLU |
| Leaky ReLU | max(0.01x,x) | (-∞,∞) | When dying ReLU | Small negative slope |
| GELU | x·Φ(x) | (-∞,∞) | Transformers | Expensive |
| SiLU/Swish | x·sigmoid(x) | (-∞,∞) | Modern nets | — |
| Softmax | e^xi/Σe^xj | (0,1), sums to 1 | Multi-class output | Numerically unstable |

### Loss Functions

| Task | Loss | Formula |
|------|------|---------|
| Binary classification | BCE | -[y log ŷ + (1-y) log(1-ŷ)] |
| Multi-class classification | Cross-entropy | -Σy_c log ŷ_c |
| Regression | MSE | Σ(y-ŷ)²/n |
| Regression (robust) | Huber | MSE if |e|<δ else MAE |
| Object detection | Focal | -(1-ŷ)^γ log ŷ |

---

## Optimization

### Gradient Descent Variants

| Optimizer | Update Rule | Pros | Cons |
|-----------|------------|------|------|
| SGD | w -= η·∇L | Simple | Slow, oscillates |
| SGD+Momentum | v = βv + ∇L; w -= ηv | Faster, smoother | Tuning β |
| RMSProp | v = βv + (1-β)∇L²; w -= η·∇L/√v | Adaptive lr | No momentum |
| Adam | Combines momentum + RMSProp | Fast convergence | Can generalize worse |
| AdamW | Adam + weight decay decoupled | Better regularization | More hyperparams |
| Lion | Sign-based update | Memory efficient | New, less tested |

**Adam formulas:**
- m_t = β₁m_{t-1} + (1-β₁)g_t  (first moment)
- v_t = β₂v_{t-1} + (1-β₂)g_t²  (second moment)
- m̂_t = m_t/(1-β₁^t), v̂_t = v_t/(1-β₂^t)  (bias correction)
- w_t = w_{t-1} - η·m̂_t/(√v̂_t + ε)

**Typical values:** β₁=0.9, β₂=0.999, ε=1e-8, η=3e-4

### Learning Rate Scheduling

- **Step decay**: reduce lr by factor every N epochs
- **Cosine annealing**: lr follows cosine curve from max to min
- **Warmup + cosine**: start low, warm up linearly, then cosine decay (standard for transformers)
- **OneCycleLR**: one cycle of warmup → peak → decay (PyTorch)
- **Reduce on plateau**: reduce when validation metric stops improving

---

## Normalization Techniques

| Technique | Normalizes Over | When to Use |
|-----------|----------------|-------------|
| Batch Norm | Batch dimension | CNNs, MLPs. Needs batch size > 1. |
| Layer Norm | Feature dimension | Transformers, RNNs, small batches |
| Group Norm | Groups of channels | Detection, when batch size = 1 |
| Instance Norm | Per sample per channel | Style transfer |
| RMS Norm | Feature dimension | Llama, Mistral — simpler than LayerNorm |

---

## Architectures

### Convolutional Neural Networks (CNN)

**Key operations:**
- **Conv2d**: receptive field slides over image. Output size = (W-F+2P)/S + 1
- **Pooling**: reduces spatial dims. Max pooling = invariance. Average = smoothing.
- **Depthwise Separable**: factorize 3×3 conv into depthwise + 1×1. 8-9x cheaper.

**Architecture families:**
- **VGG**: deep 3×3 convs, simple. Good baseline.
- **ResNet**: skip connections (x + F(x)). Enables 100+ layers. He init.
- **EfficientNet**: compound scaling of depth/width/resolution.
- **ConvNeXt**: CNN redesigned to match ViT (LayerNorm, larger kernels).

### Recurrent Networks

**LSTM gates:**
- Forget gate: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
- Input gate: i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
- Cell update: g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)
- Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
- Output gate: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
- Hidden state: h_t = o_t ⊙ tanh(C_t)

**GRU** (simpler LSTM): 2 gates instead of 3. Faster, similar performance.

### Transformer Architecture

**Scaled Dot-Product Attention:**
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) · V
```
- Q, K, V: linear projections of input (d_model → d_k, d_v)
- Scaling by sqrt(d_k): prevents softmax saturation

**Multi-Head Attention:**
```
MHA(Q,K,V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)
```

**Transformer Block:**
```
x → MHA(LayerNorm(x)) → + residual → FFN(LayerNorm(x)) → + residual
```

**Positional Encoding (sinusoidal):**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Key Transformer Variants:**
- **BERT**: encoder-only, bidirectional, masked LM
- **GPT**: decoder-only, causal attention, next-token prediction
- **T5**: encoder-decoder, text-to-text framing
- **ViT**: patch embeddings for images → standard transformer

---

## Training Techniques

### Regularization
- **Dropout**: zero p fraction of activations. Applied during training only.
- **Weight Decay** (L2): penalize large weights. Standard in AdamW.
- **Label Smoothing**: soft targets instead of one-hot. Reduces overconfidence.
- **Gradient Clipping**: clip gradient norm to max value. Essential for RNNs.
- **Data Augmentation**: random flips, crops, color jitter, MixUp, CutMix.

### Initialization
- **Xavier/Glorot**: std = sqrt(2/(fan_in + fan_out)). Good for tanh/sigmoid.
- **He (Kaiming)**: std = sqrt(2/fan_in). Good for ReLU.
- **LeCun**: std = sqrt(1/fan_in). Good for SELU.

### Transfer Learning
1. **Feature Extraction**: freeze pretrained layers, train new head only.
2. **Fine-tuning**: unfreeze some/all layers, train with small lr.
3. **LoRA**: inject low-rank adapters into frozen model. Efficient fine-tuning.

---

## Common Failure Modes

| Problem | Symptom | Fix |
|---------|---------|-----|
| Vanishing gradient | Training stalls, deep layers don't learn | Skip connections, LSTM, ReLU, gradient clipping |
| Exploding gradient | NaN loss, huge gradients | Gradient clipping, smaller lr |
| Overfitting | Train acc high, val acc low | Dropout, weight decay, more data, early stopping |
| Underfitting | Both train/val acc low | Bigger model, more epochs, lower lr |
| Dying ReLU | Many neurons output 0 | Leaky ReLU, careful init, lower lr |
| Slow convergence | Loss barely decreasing | Higher lr, momentum, learning rate warmup |
| Mode collapse (GANs) | Generator produces same output | Unrolled GANs, WGAN, spectral norm |
