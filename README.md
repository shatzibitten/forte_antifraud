# 🛡️ ForteBank AI Hackathon: Multi-Agent Anti-Fraud System

## 🏆 1st Place Solution Approach
This repository contains the source code for our solution to the **Transactional Fraud Detection** task. We implemented a **Multi-Agent System** that combines advanced feature engineering, graph neural networks, and stacking ensembles to detect financial fraud with high precision and interpretability.

## 🚀 Key Features

### 1. Multi-Agent Architecture
- **Data Steward Agent:** Handles data ingestion, cleaning, and temporal merging.
- **Feature Engineer Agent:** Generates 50+ features including behavioral biometrics, velocity stats, and graph embeddings.
- **Model Architect Agent:** Manages the training of a 2-layer Stacking Ensemble (CatBoost, LightGBM, XGBoost).
- **Auditor Agent:** Performs validation, drift analysis, and calculates business metrics (Saved Money).
- **LLM Explainer Agent:** Generates natural language explanations for suspicious transactions using SHAP values.

### 2. Advanced Feature Engineering
- **Behavioral Biometrics:** `burstiness`, `fano_factor` (Statistical Physics metrics for login patterns).
- **Graph Embeddings:** `gnn_emb` features generated via TruncatedSVD on the transaction graph.
- **Device Fingerprinting:** Detection of "Device Hopping" and "Fake OS" (e.g., iOS 26.0).
- **Recursive Features:** `time_since_last_txn`, `amount_to_avg_30d`.

### 3. Validation Strategy
- **TimeSeriesSplit (5 Folds):** Strictly respects temporal order to prevent data leakage.
- **Drift Analysis:** Automated detection of concept drift between Train and Holdout sets.

### 4. Interactive Dashboard
- **Streamlit App:** Real-time visualization of model performance, calibration curves, and SHAP explanations.
- **Scenario Testing:** Adjust thresholds and see the impact on business metrics instantly.

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.10+
- Virtual Environment (recommended)

### Setup
```bash
# 1. Clone the repository
git clone <repo_url>
cd ForteContest

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline
To run the full training and evaluation pipeline:
```bash
python main.py
```

This will:
1. Load data from `datasets/`.
2. Generate features.
3. Train the Stacking Ensemble (CatBoost + LightGBM + XGBoost).
4. Evaluate on the Holdout set (last 15% of data).
5. Generate a `shap_summary.png` and business metrics.
6. Output LLM explanations for top suspicious cases.

### Interactive Demo
We provide a Streamlit-based dashboard to explore the model's predictions and business impact.

```bash
streamlit run demo_app.py
```

### Running Inference on Custom Datasets
To run fraud detection on your own transaction data:

```bash
python inference.py \
  --transactions path/to/new_transactions.csv \
  --behavior path/to/behavior_patterns.csv \
  --model_dir models/ \
  --output predictions.csv \
  --skiprows 1
```

**Parameters:**
- `--transactions`: Path to transactions CSV file
- `--behavior`: Path to behavior patterns CSV file (can be the same as training or updated)
- `--model_dir`: Directory containing trained models (default: `models/`)
- `--output`: Output file for predictions (default: `predictions.csv`)
- `--skiprows`: Number of header rows to skip (default: 1 for original format, 0 for clean CSV)

**Required Data Format:**

Your transactions CSV should contain:
- `cst_dim_id`: Customer ID
- `transdatetime`: Transaction timestamp (YYYY-MM-DD HH:MM:SS)
- `amount`: Transaction amount
- `mcc_code`: Merchant Category Code
- `country`: Transaction country
- Additional fields as per original schema

Your behavior CSV should contain:
- `cst_dim_id`: Customer ID
- `login_timestamp`: Login timestamp
- `device_id`, `device_type`, `os_version`, `ip_address`

The output will include fraud probabilities and binary predictions for each transaction.

### Configuration
You can easily adjust the fraud detection threshold and other settings in `config.yaml`:
```yaml
# Fraud Detection Threshold
# Default: 0.5
fraud_threshold: 0.5
```

## 📊 Results Summary

| Metric | Train Period | Holdout Period |
|--------|--------------|----------------|
| **Saved Money** | ~21,000,000 KZT | ~931,000 KZT |
| **ROC-AUC** | 0.973 | 0.78 (Drift) |

**Note on Drift:** The significant drop in performance on the Holdout set is due to **Concept Drift** (new fraud vectors like Emulators and One-Shot Mules). Our analysis confirms that a **2-Week Retraining Cycle** is required to maintain performance.

## 📂 Project Structure
```
.
├── datasets/               # Raw data files
├── src/
│   ├── agents/             # AI Agent implementations
│   │   ├── data_steward.py
│   │   ├── feature_engineer.py
│   │   ├── model_architect.py
│   │   ├── auditor.py
│   │   ├── llm_explainer.py
│   │   └── graph_embedder.py
│   └── fix_segfault.py     # macOS ARM64 fix
├── main.py                 # Entry point
├── demo_app.py             # Streamlit Dashboard
├── config.yaml             # Configuration
├── requirements.txt        # Dependencies
├── solution_report.md      # Detailed technical report (English)
└── solution_report_ru.md   # Detailed technical report (Russian)
```

## 📝 License
MIT License.
