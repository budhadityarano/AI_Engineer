# AI Engineer Interview Prep

A structured, hands-on study repo covering every layer of the AI Engineer stack —
from Python fundamentals to production-grade agentic systems.

Each section is self-contained: its own `requirements.txt` (or `uv` config),
runnable Jupyter notebooks, and focused exercises.

---

## Repository Structure

```
AI_Engineer/
├── 01_Python_Foundations/        # Basics → Advanced Python
├── 02_Statistics_Math/           # Stats, Probability, Linear Algebra, Calculus
├── 03_SQL/                       # SQL from basics to query optimisation
├── 04_Data_Science/              # NumPy, Pandas, EDA
├── 05_Machine_Learning/          # Supervised, Unsupervised, Feature Eng, Eval
├── 06_Deep_Learning/             # NNs, CNNs, RNNs, Transformers, PyTorch/TF
├── 07_Generative_AI/             # LLMs, Fine-tuning, Embeddings, Multimodal
├── 08_Prompt_Engineering/        # Techniques, Chain-of-Thought, Evaluation
├── 09_RAG/                       # Vector DBs, Retrieval, Advanced RAG
├── 10_Agentic_AI/                # LangChain, LangGraph, Tools, Multi-Agent
├── 11_CI_CD/                     # GitHub Actions, Docker, Testing, MLOps
├── 12_Cloud_Deployment/          # AWS, GCP, Azure, App Hosting
├── 13_Coding_Problems/           # DSA & System Design interview problems
├── 14_Concepts_Brush_up/         # Quick-reference notes and interview Q&A
└── 15_Recent_AI_Developments/    # Paper summaries, new models, trends
```

---

## How to Use Each Section

### Option A — `pip` + virtual environment
```bash
cd 05_Machine_Learning
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

### Option B — `uv` (recommended, much faster)
```bash
cd 05_Machine_Learning
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
jupyter lab
```

---

## Study Roadmap

| Phase | Sections | Goal |
|-------|----------|------|
| **Foundation** | 01 → 03 | Python + Stats + SQL fluency |
| **Core ML** | 04 → 06 | End-to-end ML/DL pipelines |
| **GenAI Stack** | 07 → 10 | LLM applications & agents |
| **Engineering** | 11 → 12 | Ship & maintain AI products |
| **Interview Blitz** | 13 → 15 | DSA, concepts, current events |

---

## Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (optional but recommended)
- Jupyter Lab / Jupyter Notebook or Google Colab