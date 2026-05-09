# 05 — Machine Learning

End-to-end ML: from raw data to a tracked, evaluated model.

## Contents

| Folder | Topics |
|--------|--------|
| `01_supervised/` | Linear/Logistic Regression, Trees, Ensembles, SVMs, KNN |
| `02_unsupervised/` | K-Means, DBSCAN, PCA, t-SNE, Autoencoders |
| `03_feature_engineering/` | Encoding, scaling, imputation, feature selection, pipelines |
| `04_model_evaluation/` | Cross-val, metrics (AUC, F1, RMSE), calibration, bias-variance |
| `05_mlops_experiments/` | MLflow tracking, experiment management, model registry |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```