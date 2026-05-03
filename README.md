# BlockFedZTA: Trust-Aware Federated Learning for Zero Trust Intrusion Detection

---

## Overview

**BlockFedZTA** is a research-driven framework that integrates **Federated Learning (FL)** with **Zero Trust Architecture (ZTA)** for secure, scalable, and privacy-preserving intrusion detection.

It introduces a **multi-factor trust-based aggregation mechanism** that improves robustness under data drift, adversarial conditions, and large-scale client environments.

---

## Key Features

- **Zero Trust Security Integration**
- **Federated Learning-based Intrusion Detection**
- **Multi-factor Trust Model:**
  - Accuracy-based trust
  - Confidence-aware trust
  - Historical trust (EMA smoothing)
- **Adversarial Robustness:**
  - Label-flipping attack simulation
- **Drift-Aware Evaluation:**
  - No Drift | Mild Drift | Severe Drift
- **Baseline Comparison:** FedAvg
- **Scalability Analysis:** 3–15 clients
- Reproducible and research-ready pipeline

---

## Installation

### Clone Repository

```bash
git clone https://github.com/shailinsta/BlockFedZTA.git
cd BlockFedZTA
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas scikit-learn xgboost scipy matplotlib seaborn
```

---

## Dataset

Dataset location:

```text
data/Final_5Class_IDS.zip
```

The dataset is constructed by combining three publicly available IDS benchmarks:

- **NSL-KDD**
- **CICIDS2017**
- **TON-IoT**

---

## Setup Instructions

### Step 1 — Extract Dataset

Unzip the archive:

```text
data/Final_5Class_IDS.zip
```

After extraction, the following file must be present:

```text
data/Final_5Class_IDS.csv
```

### Step 2 — Verify Path

Open `src/run_pipeline.py` and confirm the dataset path is set correctly:

```python
df = pd.read_csv("data/Final_5Class_IDS.csv")
```

### Step 3 — Run the Pipeline

```bash
python src/run_pipeline.py
```

---

## Project Structure

```text
BlockFedZTA/
│
├── data/
│   └── Final_5Class_IDS.zip        # Compressed unified IDS dataset
│
├── src/
│   └── run_pipeline.py             # Main pipeline script
│
├── notebooks/                      # Jupyter notebooks for exploration
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Pipeline Workflow

1. Data loading
2. Feature cleaning and preprocessing
3. Label encoding
4. Feature scaling
5. Federated client simulation
6. Drift injection:
   - Noise
   - Feature blackout
   - Label flipping
7. Local model training (XGBoost per client)
8. Trust score computation:
   - Accuracy
   - Confidence
   - Historical trust (EMA)
9. Trust-based global aggregation
10. FedAvg baseline comparison
11. Performance evaluation
12. Scalability analysis

---

## Trust Model

The trust score for each federated client is computed as:

```text
Trust = 0.7 × Accuracy + 0.2 × Confidence + 0.1 × Historical Trust
```

| Component        | Description                                      |
|------------------|--------------------------------------------------|
| Accuracy         | Validation accuracy on local held-out data       |
| Confidence       | Mean prediction probability strength             |
| Historical Trust | Exponential Moving Average of past trust scores  |

---

## Evaluation Metrics

- **Accuracy** — overall classification performance
- **F1 Score** — macro-averaged across all 5 classes
- **Robustness under Drift** — performance across no / mild / severe distribution shift
- **Trust vs. FedAvg Comparison** — side-by-side accuracy comparison
- **Scalability Performance** — accuracy trend from 3 to 15 clients

---

## Experimental Results

### Drift Evaluation

| Scenario     | Trust-Based | FedAvg |
|--------------|:-----------:|:------:|
| No Drift     | 0.9662      | 0.9672 |
| Mild Drift   | 0.9630      | 0.9541 |
| Severe Drift | 0.9654      | 0.9466 |

> Trust-based aggregation maintains stable accuracy under drift, while FedAvg degrades noticeably.

---

## Statistical Validation

Results validated using a two-sample t-test across 5 independent runs:

| Scenario     | Method | Accuracy (mean ± std) | T-Statistic | P-Value  | Significance                       |
|--------------|--------|-----------------------|:-----------:|:--------:|------------------------------------|
| No Drift     | Trust  | 0.9662 ± 0.0012       | —           | —        | —                                  |
|              | FedAvg | 0.9672 ± 0.0012       | −2.1983     | 0.092825 | Not Significant                    |
| Mild Drift   | Trust  | 0.9630 ± 0.0044       | —           | —        | —                                  |
|              | FedAvg | 0.9541 ± 0.0012       | 3.8451      | 0.018379 | **Significant (p < 0.05)**         |
| Severe Drift | Trust  | 0.9654 ± 0.0012       | —           | —        | —                                  |
|              | FedAvg | 0.9466 ± 0.0018       | 18.2601     | 0.000053 | **Highly Significant (p < 0.001)** |

> Under severe drift, Trust-based aggregation achieves a statistically highly significant improvement over FedAvg (p < 0.001).

---

## Scalability Results

| Clients | Trust-Based | FedAvg |
|:-------:|:-----------:|:------:|
| 3       | 0.9677      | 0.9615 |
| 5       | 0.9665      | 0.9567 |
| 8       | 0.9646      | 0.9478 |
| 10      | 0.9636      | 0.9459 |
| 15      | 0.9612      | 0.9412 |

> Trust-based aggregation consistently outperforms FedAvg across all client configurations, with the gap widening at scale.

---

## Key Insights

- Trust-based aggregation **matches FedAvg** under ideal (no-drift) conditions
- Trust-based aggregation **significantly outperforms FedAvg** under mild and severe drift
- The **performance gap widens** as the number of clients increases
- The trust model **effectively suppresses unreliable or adversarial clients**
- The system **remains stable** under label-flipping adversarial conditions

---

## Research Contributions

1. Trust-aware federated aggregation framework integrating Zero Trust Architecture
2. Confidence-enhanced trust scoring using prediction probability strength
3. Robustness against label-poisoning attacks via client trust suppression
4. Drift-resilient federated intrusion detection validated across three drift levels
5. Scalable evaluation across 3–15 federated client configurations
6. Integration of Zero Trust principles with federated learning for IDS

---

## Important Notes

> **Always extract the dataset before running the pipeline.**

- Use the **relative path** only:
  ```text
  data/Final_5Class_IDS.csv
  ```
- Do **not** use absolute paths such as `C:\Users\...` or `/home/user/...`
- Do **not** point the script to the `.zip` file — it must be the extracted `.csv`

---

## Common Errors

| Error               | Cause                          | Fix                                      |
|---------------------|--------------------------------|------------------------------------------|
| `FileNotFoundError` | Dataset archive not extracted  | Unzip `Final_5Class_IDS.zip` first       |
| `FileNotFoundError` | Absolute path used             | Replace with `data/Final_5Class_IDS.csv` |
| `ParserError`       | Script pointing to `.zip` file | Update path to use the `.csv` file       |

---

## Authors

| Name |
|------|
| R.A. |
| S.M. |
| M.R. |
| S.T. |

---

## License

> This project is intended for **academic and research use only.**  
