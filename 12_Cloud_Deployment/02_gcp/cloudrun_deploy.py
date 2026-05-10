"""
GCP Cloud Run ML deployment patterns.
Shows: containerize model, push to Artifact Registry, deploy to Cloud Run.
Install: pip install google-cloud-run google-cloud-storage scikit-learn
"""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "my-gcp-project")
REGION = os.environ.get("GCP_REGION", "us-central1")
REPO = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-models"
IMAGE_NAME = "ml-inference-api"
SERVICE_NAME = "ml-inference-service"


# ── Dockerfile content ─────────────────────────────────────────────
DOCKERFILE_CONTENT = """FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl scaler.pkl ./
COPY serve.py .

ENV PORT=8080
EXPOSE 8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 serve:app
"""

SERVE_PY_CONTENT = """
import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    X = np.array(data["instances"], dtype=np.float32)
    X_s = scaler.transform(X)
    preds = model.predict(X_s).tolist()
    probas = model.predict_proba(X_s).tolist()
    return jsonify({"predictions": preds, "probabilities": probas})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
"""

REQUIREMENTS_CONTENT = """flask==3.0.0
gunicorn==21.2.0
scikit-learn==1.4.0
numpy==1.26.0
joblib==1.3.2
"""


# ── Build and push container ───────────────────────────────────────
def build_and_push_image(model, scaler, image_tag: str = "latest") -> str:
    """Build Docker image with model and push to Artifact Registry."""
    image_uri = f"{REPO}/{IMAGE_NAME}:{image_tag}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model artifacts
        joblib.dump(model, f"{tmpdir}/model.pkl")
        joblib.dump(scaler, f"{tmpdir}/scaler.pkl")

        # Write Docker files
        Path(f"{tmpdir}/Dockerfile").write_text(DOCKERFILE_CONTENT)
        Path(f"{tmpdir}/serve.py").write_text(SERVE_PY_CONTENT)
        Path(f"{tmpdir}/requirements.txt").write_text(REQUIREMENTS_CONTENT)

        # Configure Docker auth
        subprocess.run(
            ["gcloud", "auth", "configure-docker", f"{REGION}-docker.pkg.dev", "--quiet"],
            check=True, capture_output=True
        )

        # Build and push
        subprocess.run(
            ["docker", "build", "-t", image_uri, tmpdir],
            check=True
        )
        subprocess.run(
            ["docker", "push", image_uri],
            check=True
        )

    logger.info(f"Image pushed: {image_uri}")
    return image_uri


# ── Deploy to Cloud Run ────────────────────────────────────────────
def deploy_cloud_run(
    image_uri: str,
    service_name: str = SERVICE_NAME,
    region: str = REGION,
    min_instances: int = 0,
    max_instances: int = 10,
    memory: str = "2Gi",
    cpu: str = "2",
    concurrency: int = 80,
    timeout: int = 30,
) -> str:
    """Deploy container to Cloud Run."""
    cmd = [
        "gcloud", "run", "deploy", service_name,
        "--image", image_uri,
        "--region", region,
        "--platform", "managed",
        "--allow-unauthenticated",
        f"--memory={memory}",
        f"--cpu={cpu}",
        f"--concurrency={concurrency}",
        f"--timeout={timeout}",
        f"--min-instances={min_instances}",
        f"--max-instances={max_instances}",
        "--format=value(status.url)",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    service_url = result.stdout.strip()
    logger.info(f"Service deployed: {service_url}")
    return service_url


# ── Cloud Run alternative: gcloud API ─────────────────────────────
def deploy_via_api(image_uri: str, service_name: str = SERVICE_NAME) -> dict:
    """Deploy using Cloud Run v2 API (alternative to gcloud CLI)."""
    from google.cloud import run_v2

    client = run_v2.ServicesClient()
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"

    service = run_v2.Service(
        template=run_v2.RevisionTemplate(
            containers=[
                run_v2.Container(
                    image=image_uri,
                    resources=run_v2.ResourceRequirements(
                        limits={"memory": "2Gi", "cpu": "2"}
                    ),
                )
            ],
            scaling=run_v2.RevisionScaling(min_instance_count=0, max_instance_count=10),
        )
    )

    operation = client.create_service(
        parent=parent,
        service=service,
        service_id=service_name,
    )
    result = operation.result(timeout=300)
    logger.info(f"Service URI: {result.uri}")
    return {"uri": result.uri, "name": result.name}


# ── Invoke Cloud Run service ────────────────────────────────────────
def invoke_service(service_url: str, instances: list) -> dict:
    """Make a prediction request to Cloud Run service."""
    import urllib.request

    payload = json.dumps({"instances": instances}).encode()
    req = urllib.request.Request(
        f"{service_url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ── Cloud Run job for batch inference ─────────────────────────────
BATCH_JOB_YAML = """
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: ml-batch-inference
spec:
  template:
    spec:
      template:
        spec:
          containers:
          - image: {image_uri}
            command: ["python", "batch_inference.py"]
            env:
            - name: INPUT_GCS_URI
              value: "gs://my-bucket/input/data.jsonl"
            - name: OUTPUT_GCS_URI
              value: "gs://my-bucket/output/"
            resources:
              limits:
                memory: "4Gi"
                cpu: "4"
          maxRetries: 3
          timeoutSeconds: 3600
"""


if __name__ == "__main__":
    # Local demo: train model
    ds = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(ds.data, ds.target, random_state=42)
    scaler = StandardScaler().fit(X_train)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    print(f"Model trained: {type(model).__name__}")
    print(f"\nTo deploy to Cloud Run:")
    print(f"  1. image_uri = build_and_push_image(model, scaler)")
    print(f"  2. url = deploy_cloud_run(image_uri)")
    print(f"  3. result = invoke_service(url, [[feature1, feature2, ...]])")
    print(f"\nCloud Run pricing: pay-per-request, scales to zero — ideal for low-traffic ML APIs.")
