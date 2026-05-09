# 11 — CI/CD for AI Systems

Automating test, build, and deployment pipelines for ML/AI projects.

## Contents

| Folder | Topics |
|--------|--------|
| `01_github_actions/` | Workflow YAML, matrix builds, secrets, caching, model CI |
| `02_docker/` | Dockerfiles for ML, multi-stage builds, docker-compose, GPU images |
| `03_testing/` | pytest, unit/integration tests for ML code, data validation |
| `04_mlops_pipelines/` | Prefect / Airflow DAGs, model retraining triggers, monitoring |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```