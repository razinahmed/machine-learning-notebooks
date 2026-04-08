# System Architecture -- Machine Learning Notebooks

## Overview
A serving platform for machine learning models developed in Jupyter notebooks. Models are trained in notebooks, exported as artifacts, and served through a REST API that handles loading, preprocessing, inference, and metrics collection.

## Components

### 1. API Server (Node.js / Express or FastAPI)
- Exposes REST endpoints for model management, prediction, and notebook execution.
- Routes prediction requests to the appropriate loaded model.
- Collects per-request latency and prediction metrics.

### 2. Model Registry
- Maintains a catalog of available models with metadata (name, framework, input shape, file path).
- Models are stored as serialized artifacts: `.pkl` for scikit-learn, `.pt` for PyTorch, `.h5` or SavedModel for TensorFlow.
- Supports loading and unloading models on demand to manage GPU/CPU memory.

### 3. Preprocessor
- Transforms raw feature dictionaries into numeric arrays suitable for model input.
- Supports min-max normalization and z-score standardization.
- Preprocessing parameters (min, max, mean, std) are stored alongside each model artifact.

### 4. Prediction Service
- Accepts preprocessed input and calls the model's predict function.
- Returns a label, confidence score, and per-class probability distribution.
- Supports batch inference for throughput-sensitive workloads.

### 5. Notebook Runner
- Executes Jupyter notebooks programmatically using `nbconvert` or `papermill`.
- Used for scheduled retraining pipelines: a cron job runs training notebooks and exports updated model artifacts.
- Execution results (cell outputs, errors) are captured and returned via the API.

### 6. Metrics Tracker
- Records every prediction with its predicted and actual label (when ground truth is available).
- Computes running accuracy, precision, recall, and F1 per class.
- Exposes metrics via the `/metrics` endpoint for monitoring dashboards.

## Data Flow
```
Client --> [API Server]
              |
         [Preprocessor] --> [Model Registry] --> [Loaded Model]
              |                                        |
         [Prediction Service] <--- inference result ---
              |
         [Metrics Tracker]
              |
         <-- Response { label, confidence, probabilities }
```

## Training Pipeline
```
[Jupyter Notebook] --> papermill --> [Trained Model Artifact]
                                           |
                                    [Model Registry] (hot-reload)
```

1. Data scientists develop and iterate in Jupyter notebooks.
2. A scheduled job runs the training notebook via `papermill` with parameterized inputs (dataset path, hyperparameters).
3. The notebook exports a model artifact to the shared model directory.
4. The API server detects the new artifact and hot-reloads it into the registry.

## Deployment
- The API server runs in a Docker container with NVIDIA CUDA base image for GPU inference.
- Kubernetes Deployment with GPU node affinity for production.
- CPU-only fallback for development and CI environments.
- Model artifacts are stored on a shared persistent volume (or S3 bucket with a sync sidecar).

## Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./models` | Directory containing model artifacts |
| `DEFAULT_MODEL` | `iris-classifier` | Model loaded at startup |
| `GPU_ENABLED` | `false` | Enable GPU inference |
| `MAX_BATCH_SIZE` | `32` | Maximum batch size for batch prediction |
| `METRICS_WINDOW` | `1000` | Number of recent predictions to track for metrics |
