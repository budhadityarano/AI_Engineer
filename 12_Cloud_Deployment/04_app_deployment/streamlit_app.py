"""
Streamlit ML demo app.
Install: pip install streamlit scikit-learn numpy pandas matplotlib
Run:     streamlit run streamlit_app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Model Explorer",
    page_icon="",
    layout="wide",
)

st.title("ML Model Explorer")
st.markdown("Interactive demo for AI Engineering interview prep.")


# ── Sidebar Controls ──────────────────────────────────────────────
st.sidebar.header("Configuration")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    ["Breast Cancer", "Iris"],
    help="Select a built-in sklearn dataset"
)

model_name = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "Gradient Boosting", "Logistic Regression"]
)

test_size = st.sidebar.slider("Test Set Size", 0.10, 0.40, 0.20, 0.05)
random_state = st.sidebar.number_input("Random Seed", 0, 100, 42)

# Model hyperparameters
st.sidebar.subheader("Hyperparameters")
if model_name in ["Random Forest", "Gradient Boosting"]:
    n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100, 10)
    max_depth = st.sidebar.slider("max_depth (0=None)", 0, 20, 0)
    max_depth = None if max_depth == 0 else max_depth
else:
    n_estimators, max_depth = 100, None
    C = st.sidebar.slider("C (regularization)", 0.01, 10.0, 1.0)


# ── Data Loading ──────────────────────────────────────────────────
@st.cache_data
def load_dataset(name):
    if name == "Breast Cancer":
        ds = load_breast_cancer()
    else:
        ds = load_iris()
    return ds.data, ds.target, ds.feature_names.tolist(), ds.target_names.tolist()


X, y, feature_names, class_names = load_dataset(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# ── Model Training ─────────────────────────────────────────────────
@st.cache_resource
def train(model_name, n_estimators, max_depth, random_state, _X_train, _y_train):
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state, n_jobs=-1)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth or 3,
                                           random_state=random_state)
    else:
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    model.fit(_X_train, _y_train)
    return model


with st.spinner(f"Training {model_name}..."):
    model = train(model_name, n_estimators, max_depth, random_state, X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)


# ── Metrics Row ────────────────────────────────────────────────────
st.header("Performance Metrics")
acc = accuracy_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("Train Size", len(X_train))
col3.metric("Test Size", len(X_test))
col4.metric("Classes", len(class_names))


# ── Two-column layout ──────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with right:
    if len(class_names) == 2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        auc = roc_auc_score(y_test, y_proba[:, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        unique, counts = np.unique(y_test, return_counts=True)
        ax.bar([class_names[i] for i in unique], counts, color="steelblue")
        ax.set_ylabel("Count")
        ax.set_title("Test Set Class Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── Feature Importance ─────────────────────────────────────────────
if hasattr(model, "feature_importances_"):
    st.subheader("Top 10 Feature Importances")
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh([feature_names[i] for i in top_idx][::-1], importances[top_idx][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} Feature Importances")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Classification Report ──────────────────────────────────────────
with st.expander("Full Classification Report"):
    report = classification_report(y_test, y_pred, target_names=class_names)
    st.code(report)


# ── Live Prediction ────────────────────────────────────────────────
st.header("Live Prediction")
st.markdown("Adjust sliders to predict on a custom sample.")

feature_vals = []
cols = st.columns(5)
for i, fname in enumerate(feature_names[:10]):  # show first 10 features
    min_v, max_v = float(X[:, i].min()), float(X[:, i].max())
    val = cols[i % 5].slider(
        fname[:20], min_value=min_v, max_value=max_v,
        value=float(X[:, i].mean()), key=f"feat_{i}"
    )
    feature_vals.append(val)

# Fill remaining features with mean
full_features = list(feature_vals) + list(X[:, 10:].mean(axis=0))
sample = np.array(full_features).reshape(1, -1)
sample_s = scaler.transform(sample)

pred_class = model.predict(sample_s)[0]
pred_proba = model.predict_proba(sample_s)[0]

st.success(f"Predicted class: **{class_names[pred_class]}** (confidence: {pred_proba.max():.1%})")
