"""
Azure ML deployment patterns.
Shows: workspace setup, environment, online endpoint, managed inference.
Install: pip install azure-ai-ml azure-identity scikit-learn
"""
import json
import logging
import os
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

SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "your-sub-id")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "ml-rg")
WORKSPACE_NAME = os.environ.get("AZURE_ML_WORKSPACE", "my-ml-workspace")
ENDPOINT_NAME = "ml-inference-endpoint"
DEPLOYMENT_NAME = "ml-deployment"


# ── Scoring script (runs inside Azure ML container) ────────────────
SCORE_PY = '''
import os
import json
import numpy as np
import joblib

def init():
    global model, scaler
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    model = joblib.load(os.path.join(model_dir, "model", "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "model", "scaler.pkl"))

def run(raw_data):
    try:
        data = json.loads(raw_data)
        X = np.array(data["instances"], dtype=np.float32)
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled).tolist()
        probabilities = model.predict_proba(X_scaled).tolist()
        return json.dumps({"predictions": predictions, "probabilities": probabilities})
    except Exception as e:
        return json.dumps({"error": str(e)})
'''

CONDA_ENV = '''
name: ml-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pip:
    - scikit-learn==1.4.0
    - numpy==1.26.0
    - joblib==1.3.2
    - azureml-defaults
'''


# ── Azure ML client setup ──────────────────────────────────────────
def get_ml_client():
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml import MLClient

    credential = DefaultAzureCredential()
    return MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)


# ── Register model ─────────────────────────────────────────────────
def register_model(ml_client, model, scaler) -> str:
    from azure.ai.ml.entities import Model
    from azure.ai.ml.constants import AssetTypes

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "model"
        model_dir.mkdir()
        joblib.dump(model, model_dir / "model.pkl")
        joblib.dump(scaler, model_dir / "scaler.pkl")

        registered = ml_client.models.create_or_update(
            Model(
                name="rf-classifier",
                version="1",
                path=str(model_dir),
                type=AssetTypes.CUSTOM_MODEL,
                description="RandomForest classifier for breast cancer detection",
                tags={"framework": "sklearn", "task": "classification"},
            )
        )

    logger.info(f"Model registered: {registered.name}:{registered.version}")
    return f"azureml:{registered.name}:{registered.version}"


# ── Create environment ─────────────────────────────────────────────
def create_environment(ml_client) -> str:
    from azure.ai.ml.entities import Environment, BuildContext

    env = ml_client.environments.create_or_update(
        Environment(
            name="ml-sklearn-env",
            version="1",
            conda_file=CONDA_ENV,
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            description="Scikit-learn inference environment",
        )
    )
    return f"azureml:{env.name}:{env.version}"


# ── Deploy managed online endpoint ────────────────────────────────
def deploy_online_endpoint(ml_client, model_id: str, env_id: str) -> str:
    from azure.ai.ml.entities import (
        ManagedOnlineEndpoint,
        ManagedOnlineDeployment,
        CodeConfiguration,
        ProbeSettings,
    )

    # Create endpoint
    endpoint = ml_client.online_endpoints.begin_create_or_update(
        ManagedOnlineEndpoint(
            name=ENDPOINT_NAME,
            description="ML model inference endpoint",
            auth_mode="key",
            tags={"env": "production"},
        )
    ).result()

    logger.info(f"Endpoint created: {endpoint.scoring_uri}")

    # Write scoring script
    with tempfile.TemporaryDirectory() as tmpdir:
        score_path = Path(tmpdir) / "score.py"
        score_path.write_text(SCORE_PY)

        # Create deployment
        deployment = ml_client.online_deployments.begin_create_or_update(
            ManagedOnlineDeployment(
                name=DEPLOYMENT_NAME,
                endpoint_name=ENDPOINT_NAME,
                model=model_id,
                environment=env_id,
                code_configuration=CodeConfiguration(code=tmpdir, scoring_script="score.py"),
                instance_type="Standard_DS3_v2",
                instance_count=1,
                liveness_probe=ProbeSettings(failure_threshold=30, success_threshold=1,
                                             timeout=2, period=10, initial_delay=10),
                request_settings={"request_timeout_ms": 90000, "max_concurrent_requests_per_instance": 1},
            )
        ).result()

    # Route 100% traffic
    ml_client.online_endpoints.begin_create_or_update(
        ManagedOnlineEndpoint(
            name=ENDPOINT_NAME,
            traffic={DEPLOYMENT_NAME: 100},
        )
    ).result()

    return endpoint.scoring_uri


# ── Invoke endpoint ────────────────────────────────────────────────
def invoke_endpoint(ml_client, instances: list) -> dict:
    import urllib.request

    # Get API key
    keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)
    api_key = keys.primary_key

    endpoint_info = ml_client.online_endpoints.get(ENDPOINT_NAME)
    scoring_uri = endpoint_info.scoring_uri

    payload = json.dumps({"instances": instances}).encode()
    req = urllib.request.Request(
        scoring_uri,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ── Azure ML Pipeline (training + evaluation) ───────────────────────
PIPELINE_EXAMPLE = '''
# Azure ML pipeline using component-based approach
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml.entities import PipelineJob

@dsl.pipeline(compute="cpu-cluster", description="ML training pipeline")
def training_pipeline(input_data: Input(type="uri_folder")):
    preprocess_step = preprocess_component(raw_data=input_data)
    train_step = train_component(
        training_data=preprocess_step.outputs.processed_data,
        n_estimators=100,
        max_depth=5,
    )
    evaluate_step = evaluate_component(
        model=train_step.outputs.model,
        test_data=preprocess_step.outputs.test_data,
        min_accuracy=0.85,
    )
    return {"trained_model": train_step.outputs.model}

pipeline_job = training_pipeline(
    input_data=Input(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/data/")
)
ml_client.jobs.create_or_update(pipeline_job, experiment_name="cancer-detection")
'''


if __name__ == "__main__":
    # Local demo
    ds = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(ds.data, ds.target, random_state=42)
    scaler = StandardScaler().fit(X_train)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    print("Azure ML Deployment Guide:")
    print("1. ml_client = get_ml_client()")
    print("2. model_id = register_model(ml_client, model, scaler)")
    print("3. env_id = create_environment(ml_client)")
    print("4. uri = deploy_online_endpoint(ml_client, model_id, env_id)")
    print("5. result = invoke_endpoint(ml_client, [[feature1, ...]])")
    print("\nAzure ML advantages:")
    print("  - MLflow integration built-in")
    print("  - Managed endpoints with auto-scaling")
    print("  - Pipeline SDK for reproducible training")
    print("  - Data versioning and lineage tracking")
