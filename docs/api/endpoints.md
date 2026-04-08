# Machine Learning Notebooks API

## Model Management

### `GET /api/models`
List all registered models with their status and metadata.

**Response 200:**
```json
{
  "models": [
    { "name": "iris-classifier", "framework": "sklearn", "status": "loaded", "inputShape": [4] },
    { "name": "sentiment-bert", "framework": "pytorch", "status": "unloaded", "inputShape": [512] }
  ]
}
```

### `POST /api/models/load`
Load a model into memory by name.

**Request Body:** `{ "name": "sentiment-bert" }`
**Response 200:** `{ "model": { "name": "sentiment-bert", "framework": "pytorch", "status": "loaded", "inputShape": [512] } }`
**Response 404:** `{ "error": "Model 'nonexistent' not found in the model catalog" }`

### `POST /api/models/unload`
Unload a model from memory to free resources.

**Request Body:** `{ "name": "sentiment-bert" }`
**Response 200:** `{ "unloaded": true }`

## Prediction

### `POST /api/predict`
Run inference on a single input using a loaded model.

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | yes | Name of the loaded model |
| `input` | object | yes | Feature values matching the model's expected input shape |

**Response 200:**
```json
{
  "prediction": {
    "label": "setosa",
    "confidence": 0.95,
    "probabilities": { "setosa": 0.95, "versicolor": 0.03, "virginica": 0.02 }
  }
}
```

### `POST /api/predict/batch`
Run inference on multiple inputs in a single request.

**Request Body:** `{ "model": "iris-classifier", "inputs": [{ ... }, { ... }] }`
**Response 200:** `{ "predictions": [{ "label": "setosa", "confidence": 0.95 }, ...] }`

## Notebooks

### `GET /api/notebooks`
List available Jupyter notebooks.

### `POST /api/notebooks/:name/execute`
Execute a notebook end-to-end and return cell outputs.

**Response 200:** `{ "cells": [{ "type": "code", "source": "...", "output": "..." }], "duration": 12.5 }`

## Metrics

### `GET /api/metrics`
Return model performance metrics (accuracy, precision, recall) collected from recent predictions.

## Health

### `GET /api/health`
Returns `{ "status": "ok", "modelsLoaded": 2, "gpuAvailable": true }`.

## Authentication
All endpoints require an API key via the `X-API-Key` header.
