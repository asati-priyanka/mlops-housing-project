# 🏠 California Housing MLOps Pipeline

This project implements a **complete MLOps pipeline** for the California Housing dataset using **FastAPI, MLflow, DVC, Docker, GitHub Actions, Prometheus, and Grafana**.

It covers:
- Dataset versioning (DVC)
- Model training & experiment tracking (MLflow)
- Model registry & deployment (FastAPI in Docker)
- CI/CD automation (GitHub Actions)
- Monitoring & logging (SQLite + Prometheus)
- Auto-retraining trigger when data changes
- Streamlit for interactive UI

---

## 📜 Architecture

```mermaid
flowchart LR
    A["📦 Data Source"] --> B["🧭 DVC Tracking"]
    B --> C["🧠 Model Training<br/>📈 MLflow Tracking"]
    C --> D["🏛️ MLflow Model Registry (alias: best)"]
    D --> E["⚡ FastAPI Service (models:/...@best)"]
    E --> F["🐳 Docker Compose"]
    F --> G["🚀 Deploy (Local/Cloud)"]
    E --> H["🗄️ SQLite Logging"]
    E --> I["📥 Prometheus /metrics/prom"]
    I --> J["📊 Grafana Dashboard"]

    %% Streamlit UI calling API
    L["🖥️ Streamlit UI (streamlit_app.py)"] -->|POST /predict| E

    %% Compose orchestrates API + Streamlit (+ others)
    F --> L

    subgraph CICD["CI/CD"]
      K["🧪 Lint & Tests"] --> L1["🏗️ Build Image"]
      L1 --> M["⬆️ Push to Docker Hub"]
      M --> F
    end

