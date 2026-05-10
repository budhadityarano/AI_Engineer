"""
AWS SageMaker deployment patterns.
Shows: training job, model registration, endpoint deployment, batch transform.
Install: pip install boto3 sagemaker scikit-learn
"""
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path

import boto3
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── 1. SageMaker Script Mode (inference.py) ────────────────────────
# This code runs INSIDE the container on SageMaker
INFERENCE_SCRIPT = '''
import json
import os
import numpy as np
import joblib

def model_fn(model_dir):
    """Load model from SageMaker model directory."""
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return {"model": model, "scaler": scaler}

def input_fn(request_body, content_type="application/json"):
    """Deserialize and preprocess input."""
    if content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["instances"], dtype=np.float32)
    raise ValueError(f"Unsupported content_type: {content_type}")

def predict_fn(input_data, model_artifacts):
    """Run inference."""
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    X_scaled = scaler.transform(input_data)
    predictions = model.predict(X_scaled).tolist()
    probabilities = model.predict_proba(X_scaled).tolist()
    return {"predictions": predictions, "probabilities": probabilities}

def output_fn(prediction, accept="application/json"):
    """Serialize output."""
    if accept == "application/json":
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported accept: {accept}")
'''


# ── 2. Package model artifacts ──────────────────────────────────────
def train_and_package_model(output_path: str = "/tmp/model.tar.gz") -> str:
    """Train model locally and package for SageMaker."""
    logger.info("Training model locally...")
    ds = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(
        ds.data, ds.target, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        joblib.dump(model, f"{tmpdir}/model.pkl")
        joblib.dump(scaler, f"{tmpdir}/scaler.pkl")

        # Write inference script
        with open(f"{tmpdir}/inference.py", "w") as f:
            f.write(INFERENCE_SCRIPT)

        # Create tar.gz (SageMaker model format)
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(f"{tmpdir}/model.pkl", arcname="model.pkl")
            tar.add(f"{tmpdir}/scaler.pkl", arcname="scaler.pkl")
            tar.add(f"{tmpdir}/inference.py", arcname="code/inference.py")

    logger.info(f"Model packaged: {output_path}")
    return output_path


# ── 3. SageMaker deployment workflow ───────────────────────────────
def deploy_to_sagemaker(
    model_artifact_path: str,
    bucket: str,
    role_arn: str,
    instance_type: str = "ml.m5.large",
    endpoint_name: str = "ml-model-endpoint",
    region: str = "us-east-1",
):
    """
    Full SageMaker deployment:
    1. Upload model artifacts to S3
    2. Create SageMaker Model
    3. Create Endpoint Config
    4. Deploy Endpoint
    """
    import sagemaker
    from sagemaker.sklearn.model import SKLearnModel

    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    # Upload to S3
    logger.info(f"Uploading model to s3://{bucket}/models/")
    s3_model_uri = session.upload_data(model_artifact_path, bucket=bucket, key_prefix="models")
    logger.info(f"Model uploaded: {s3_model_uri}")

    # Create SageMaker SKLearn model
    sklearn_model = SKLearnModel(
        model_data=s3_model_uri,
        role=role_arn,
        entry_point="inference.py",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
    )

    # Deploy endpoint
    logger.info(f"Deploying to endpoint: {endpoint_name}")
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        wait=True,
    )
    logger.info(f"Endpoint deployed: {endpoint_name}")
    return predictor


# ── 4. Invoke endpoint ─────────────────────────────────────────────
def invoke_endpoint(endpoint_name: str, instances: list, region: str = "us-east-1") -> dict:
    """Call a deployed SageMaker endpoint."""
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    payload = json.dumps({"instances": instances})
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read().decode())
    return result


# ── 5. Batch Transform ─────────────────────────────────────────────
def run_batch_transform(
    model_name: str,
    input_s3_uri: str,
    output_s3_uri: str,
    instance_type: str = "ml.m5.xlarge",
    instance_count: int = 1,
    region: str = "us-east-1",
):
    """Run batch inference on large dataset in S3."""
    sm = boto3.client("sagemaker", region_name=region)

    job_name = f"batch-transform-{int(__import__('time').time())}"
    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_s3_uri}},
            "ContentType": "application/json",
            "SplitType": "Line",
        },
        TransformOutput={
            "S3OutputPath": output_s3_uri,
            "AssembleWith": "Line",
        },
        TransformResources={
            "InstanceType": instance_type,
            "InstanceCount": instance_count,
        },
        BatchStrategy="MultiRecord",
    )
    logger.info(f"Batch transform job started: {job_name}")
    return job_name


# ── 6. Endpoint Auto-scaling ───────────────────────────────────────
def configure_autoscaling(endpoint_name: str, variant_name: str = "AllTraffic",
                           min_capacity: int = 1, max_capacity: int = 10,
                           region: str = "us-east-1"):
    """Register auto-scaling for SageMaker endpoint."""
    aas = boto3.client("application-autoscaling", region_name=region)

    resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity,
    )

    # Scale on invocation count per instance
    aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scaling",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 1000.0,  # invocations/minute per instance
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
    logger.info(f"Auto-scaling configured for {endpoint_name}")


if __name__ == "__main__":
    # Local demo: package model only
    model_path = train_and_package_model("/tmp/model.tar.gz")
    print(f"Model packaged at: {model_path}")
    print("\nTo deploy to SageMaker:")
    print("  predictor = deploy_to_sagemaker(model_path, bucket='my-bucket', role_arn='arn:aws:iam::...')")
    print("\nTo invoke:")
    print("  result = invoke_endpoint('ml-model-endpoint', [[1.2, 3.4, ...]])")
