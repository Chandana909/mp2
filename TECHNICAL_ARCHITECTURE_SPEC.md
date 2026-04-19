# ML Reliability & Lifecycle Supervision Platform (MP2)
## System Design & Technical Architecture Specification

### 1. Executive Summary
The MP2 Platform is an industry-grade ecosystem designed to manage the "Day 2" operations of Machine Learning models. It addresses the critical gap between model development (Data Science) and model operations (MLE/DevOps) by providing a unified interface for model registration, autonomous monitoring, data drift detection, and closed-loop hyperparameter optimization.

### 2. Core Technology Stack
*   **Backend Engine:** FastAPI (Python 3.9+) with Uvicorn (standard workers).
*   **Concurrency Model:** Asynchronous I/O with background threading (`asyncio.to_thread`) for blocking CPU/IO tasks (Storage/Inference).
*   **Persistence Layer:** SQLite 3 with Write-Ahead Logging (WAL) mode enabled for simultaneous read/write operations.
*   **Machine Learning:** Scikit-Learn (Ensemble methods), NumPy, Joblib (binary serialization).
*   **Frontend Architecture:** Vanilla ES6 Javascript, HTML5 Semantic Tags, and Modern CSS (Glassmorphism, CSS Grid, Flexbox). Zero-framework dependency for maximum performance and compatibility.
*   **Monitoring Infrastructure:** Custom Drift Detection algorithms (KS-test based) and Prometheus-compatible metrics endpoint.

---

### 3. Integrated Module Directory Structure

```plaintext
C:/mp2/
|-- serving/
|   |-- api.py            <-- REST Entry, Concurrency Management, Daemons
|   |-- predictor.py      <-- Inference Engine (Scikit-learn wrapper)
|   |-- static/index.html <-- Premium Glassmorphic UI
|-- ml_platform/
|   |-- registry.py       <-- Central Orchestrator (Thread-safe)
|   |-- optimizer.py      <-- AutoML Engine (Randomized Grid Search)
|   |-- validator.py      <-- Input Schema Enforcement
|   |-- audit.py          <-- Persistence Ledger for Predictions
|   |-- models.py         <-- Pydantic Data Models (SSOT)
|-- storage/
|   |-- metadata_store.py <-- SQL Engine (WAL-optimized)
|   |-- model_store.py    <-- Binary Blob Manager (Joblib)
|   |-- audit_store.py    <-- JSONL Append-only Persistent Store
|-- monitoring/
|   |-- drift_simulator.py <-- Synthetic Signal Generation
|   |-- alerts.py         <-- Trigger Logic for Threshold Breaches
|-- lifecycle/
|   |-- rollback.py       <-- Version Failover Logic
|-- ml_platform_client.py   <-- Enterprise Integration SDK (httpx)
```

---

### 4. Detailed Module Analysis

#### 4.1 `ml_platform/optimizer.py` (The AutoML Engine)
**Purpose:** Implements the closed-loop parameter optimization.
**Technical Logic:**
*   **Search Space:** Defined as dictionaries (`RF_SEARCH_SPACE`) mapping hyperparameter names to lists of discrete values.
*   **Execution Strategy:** Uses `concurrent.futures.ThreadPoolExecutor` to execute `n_trials` in parallel. This is critical for scaling, as training multiple Random Forest variants is CPU-intensive.
*   **State Management:** An in-memory `OptimizationRun` registry tracks every trial's accuracy, duration, and parameter set.
*   **Registration Trigger:** If the best trial outperforms the current 'Active' model's accuracy, the optimizer programmatically calls `ModelRegistry.register_model()` to save the new version as 'Candidate'.

#### 4.2 `serving/api.py` (Concurrency & Daemons)
**Purpose:** API Routing and Lifecycle Daemons.
**Concurrency Design:**
*   Uses `asyncio.to_thread()` for all `registry`, `predictor`, and `audit` calls. This transforms the synchronous Scikit-learn/SQLite stack into a non-blocking asynchronous server capable of handling hundreds of concurrent users.
*   **Daemon 1 (Evaluation):** Polls every 3600s. It finds the candidate with the highest accuracy and promotes it to active.
*   **Daemon 2 (Optimization):** Polls every 21600s. It analyzes the `drift_history`. If a model's drift score > 0.35, it triggers the `HyperparameterOptimizer` to find a better parameter set for the drifted distribution.

#### 4.3 `storage/metadata_store.py` (Persistence)
**Purpose:** High-performance metadata indexing.
**Optimization Details:**
*   **WAL Mode:** `PRAGMA journal_mode=WAL` is enabled. In standard SQLite, writing locks the whole DB. In WAL mode, readers don't block writers, and writers don't block readers. This provides industry-grade performance for a file-based DB.
*   **Indexing:** SQL indexes are established on `model_name` and `status` columns, ensuring $O(k)$ lookup time even as the registry grows to thousands of entries.

#### 4.4 `monitoring/drift_simulator.py`
**Purpose:** Validation of the platform's drift-response capabilities.
**Technical Logic:**
*   Generates noise-modulated signals around a center severity float.
*   Uses `numpy.random.default_rng` to create per-feature scores (KS, JS, PSI) that mimic real-world distribution shifts.
*   The output is a `DriftReport` pydantic model transmitted to both the UI and the persistence ledger.

---

### 5. Systematic Integration Flow (Closed AutoML Loop)

1.  **Drift Detection:** `monitoring/drift_simulator` (or live monitor) detects a shift in input distribution.
2.  **Alerting:** The API `/drift/history` ledger is updated.
3.  **Auto-Optimization:** `auto_optimization_daemon` (in `api.py`) detects the drift and triggers `HyperparameterOptimizer.optimize()`.
4.  **Tuning:** Multiple models are trained in parallel across CPU cores.
5.  **Election:** The best model is registered as a new 'Candidate' version.
6.  **Promotion:** `auto_evaluation_daemon` (in `api.py`) detects the superior candidate and swaps the 'Active' status via `ModelRegistry.update_model_status()`.
7.  **Hot-Reloading:** The next call to `/predict` automatically loads the new artifact without server restart.

---

### 6. Security and Design Safety
*   **Data Integrity:** `threading.RLock` in the `ModelRegistry` prevents race conditions during status updates.
*   **Pre-flight Validation:** `validator.py` ensures input tensors match the feature shapes expected by the model binary before loading it into RAM.
*   **Auditability:** Every single prediction request/response is hashed and logged in `storage/audit_logs` for compliance.
