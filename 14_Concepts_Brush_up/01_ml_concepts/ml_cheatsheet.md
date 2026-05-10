# Machine Learning Concepts Cheatsheet

## Core ML Algorithms

### Supervised Learning

| Algorithm | Type | Bias | Variance | Interpretable | Best For |
|-----------|------|------|----------|---------------|----------|
| Linear Regression | Reg | High | Low | Yes | Linear relationships |
| Logistic Regression | Clf | High | Low | Yes | Binary classification baseline |
| Decision Tree | Both | Low | High | Yes | Non-linear, categorical features |
| Random Forest | Both | Medium | Low | Partial | General purpose, tabular |
| Gradient Boosting | Both | Low | Low | No | Competitions, tabular SOTA |
| SVM | Both | Medium | Low | No | High-dim, small datasets |
| KNN | Both | Low | High | Yes | Non-parametric, small N |
| Naive Bayes | Clf | High | Low | Yes | Text classification |

### Unsupervised Learning

| Algorithm | Type | Complexity | Requires k | Handles Noise |
|-----------|------|-----------|-----------|---------------|
| K-Means | Clustering | O(nkdi) | Yes | No |
| DBSCAN | Clustering | O(n log n) | No | Yes |
| Gaussian Mixture | Clustering | O(nkd²) | Yes | Soft |
| PCA | Dim reduction | O(nd²) | No (components) | No |
| t-SNE | Visualization | O(n²) | No | Partial |
| UMAP | Dim reduction | O(n^1.14) | No | Partial |
| Autoencoders | Dim reduction | Varies | No | Varies |

---

## Key Concepts

### Bias-Variance Tradeoff
- **Bias**: error from wrong assumptions (underfitting). High bias → model too simple.
- **Variance**: error from sensitivity to training data (overfitting). High variance → model too complex.
- **Total Error** = Bias² + Variance + Irreducible Noise
- **Sweet spot**: regularization, ensemble methods, cross-validation

### Regularization
- **L1 (Lasso)**: penalty = λ Σ|w_i|. Promotes sparsity (zeros out features). Feature selection.
- **L2 (Ridge)**: penalty = λ Σw_i². Shrinks all weights. Better when all features matter.
- **ElasticNet**: combination of L1 + L2.
- **Dropout**: neural net regularization — randomly zero activations during training.

### Cross-Validation
- **k-Fold**: split into k folds, train on k-1, test on 1, rotate. Average performance.
- **Stratified k-Fold**: preserves class proportions in each fold. Use for imbalanced data.
- **LOOCV**: k = n. Low bias, high variance. Expensive.
- **Time-Series CV**: never use future data to predict past. Walk-forward validation.

### Model Evaluation Metrics

**Classification:**
- **Accuracy**: (TP+TN)/(TP+TN+FP+FN). Misleading for imbalanced classes.
- **Precision**: TP/(TP+FP). "Of predicted positives, how many are right?"
- **Recall (Sensitivity)**: TP/(TP+FN). "Of actual positives, how many did we catch?"
- **F1**: 2·P·R/(P+R). Harmonic mean. Use when FP and FN equally costly.
- **AUC-ROC**: probability that model ranks random positive higher than random negative.
- **PR-AUC**: better than ROC for highly imbalanced datasets.

**Regression:**
- **MSE**: Mean Squared Error. Penalizes large errors. In units².
- **RMSE**: √MSE. Same units as target.
- **MAE**: Mean Absolute Error. Robust to outliers.
- **R²**: 1 - SS_res/SS_tot. Proportion of variance explained. Range: (-∞, 1].

### Feature Engineering
- **Normalization** (Min-Max): scale to [0,1]. Sensitive to outliers.
- **Standardization** (Z-score): zero mean, unit variance. Better for most algorithms.
- **Target Encoding**: replace category with mean of target. Risk of leakage.
- **One-Hot Encoding**: binary columns per category. Avoid for high cardinality.
- **Polynomial Features**: x → x, x², x³, x₁x₂. Captures non-linear interactions.
- **Log Transform**: for right-skewed distributions (e.g., income, count data).

---

## Ensemble Methods

### Bagging (Bootstrap Aggregating)
- Train N models on bootstrap samples of training data
- Average (regression) or majority vote (classification)
- **Reduces variance**. Works best with high-variance models (deep trees).
- Example: **Random Forest** (bagging + feature subsampling)

### Boosting
- Train models sequentially, each correcting errors of previous
- Weight misclassified samples more
- **Reduces bias**. Final model = weighted sum of weak learners.
- Examples: **AdaBoost**, **Gradient Boosting**, **XGBoost**, **LightGBM**

### Stacking
- Train base models on training data
- Meta-learner trained on out-of-fold predictions of base models
- More complex but can outperform both

---

## Gradient Boosting Details

**Algorithm:**
1. Initialize with prediction = mean(y)
2. For t = 1 to T:
   - Compute residuals: r_i = y_i - ŷ_i
   - Fit tree to residuals
   - Update: ŷ_i ← ŷ_i + η · tree(x_i)
3. Final = sum of all tree predictions

**Key Hyperparameters:**
- `n_estimators`: number of trees. More = less underfitting but more overfitting risk.
- `learning_rate`: η. Lower = more robust, needs more trees.
- `max_depth`: 3-6 typical for boosting. Deeper trees = more variance.
- `min_samples_leaf`: minimum samples at leaf. Higher = more regularization.
- `subsample`: fraction of samples per tree. < 1.0 = stochastic gradient boosting.

---

## Imbalanced Data Strategies

| Strategy | How | Pros | Cons |
|----------|-----|------|------|
| SMOTE | Synthetic minority oversampling | More data | May create noise |
| Undersampling | Remove majority examples | Fast | Loses information |
| Class weights | weight=class_weight='balanced' | No data modification | May underfit |
| Threshold tuning | Adjust decision threshold | Post-hoc | Requires calibration |
| Ensemble (BalancedBagging) | Combine under/oversampling | Robust | Complex |

---

## Hyperparameter Tuning

- **Grid Search**: exhaustive. Good for small search space.
- **Random Search**: sample randomly. Better for large spaces (Bergstra 2012).
- **Bayesian Optimization**: use prior results to guide search (Optuna, Hyperopt).
- **Early Stopping**: stop when validation metric plateaus. Essential for neural nets.
