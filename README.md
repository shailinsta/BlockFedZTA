# BlockFedZTA: Trust-Aware Federated Learning for Zero Trust Intrusion Detection

## Overview

BlockFedZTA is a research-driven framework that integrates Federated Learning (FL) with Zero Trust Architecture (ZTA) for secure, scalable, and privacy-preserving intrusion detection.

This version introduces a multi-factor trust-based aggregation mechanism that improves robustness under data drift, adversarial conditions, and large-scale client environments.

---

## Key Features

- Zero Trust Security Integration  
- Federated Learning-based Intrusion Detection  
- Multi-factor Trust Model:
  - Accuracy-based trust  
  - Confidence-aware trust  
  - Historical trust (EMA smoothing)  
- Adversarial robustness:
  - Label-flipping attack simulation  
- Drift-aware evaluation:
  - No Drift  
  - Mild Drift  
  - Severe Drift  
- Baseline comparison:
  - FedAvg  
- Scalability analysis (3–15 clients)  
- Reproducible and research-ready pipeline  

---

## Dataset

The dataset is provided as:

```

data/Final_5Class_IDS.zip

```

It is a unified dataset created by combining:

- NSL-KDD  
- CICIDS2017  
- TON-IoT  

---

## Dataset Setup

### Step 1 — Extract Dataset

Unzip the dataset:

```

data/Final_5Class_IDS.zip

```

After extraction:

```

data/Final_5Class_IDS.csv

```

---

### Step 2 — Update Path

Open:

```

src/run_pipeline.py

````

Replace:

```python
df = pd.read_csv("YOUR_PATH_HERE")
````

With:

```python
df = pd.read_csv("data/Final_5Class_IDS.csv")
```

---

### Step 3 — Run the Project

```bash
python src/run_pipeline.py
```

---

## Project Structure

```
BlockFedZTA/
│
├── data/
│   └── Final_5Class_IDS.zip
│
├── src/
│   └── run_pipeline.py
│
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Pipeline Workflow

1. Load dataset
2. Feature cleaning and normalization
3. Label encoding
4. Feature scaling
5. Federated client simulation
6. Drift injection (noise, blackout, label flipping)
7. Model training (XGBoost per client)
8. Trust computation:

   * Accuracy
   * Confidence
   * Historical trust (EMA)
9. Trust-based aggregation
10. FedAvg baseline comparison
11. Performance evaluation
12. Scalability analysis

---

## Trust Model

The trust score for each client is computed as:

```
Trust = α × Accuracy + β × Confidence + γ × Historical Trust
```

Where:

* Accuracy = validation accuracy
* Confidence = prediction probability strength
* Historical Trust = exponential moving average

---

## Evaluation Metrics

* Accuracy
* F1 Score
* Robustness under drift
* Trust vs FedAvg comparison
* Scalability performance

---

## Experimental Results

### Drift Evaluation

| Scenario     | Trust-Based | FedAvg |
| ------------ | ----------- | ------ |
| No Drift     | ~0.963      | ~0.964 |
| Mild Drift   | ~0.962      | ~0.953 |
| Severe Drift | ~0.962      | ~0.949 |

---

### Scalability Results

| Clients | Trust  | FedAvg |
| ------- | ------ | ------ |
| 3       | ~0.964 | ~0.958 |
| 5       | ~0.963 | ~0.954 |
| 8       | ~0.962 | ~0.949 |
| 10      | ~0.960 | ~0.946 |
| 15      | ~0.959 | ~0.940 |

---

## Key Insights

* Trust-based aggregation matches FedAvg under ideal conditions
* Trust significantly outperforms FedAvg under drift
* Performance gap increases with number of clients
* Trust model effectively suppresses unreliable clients
* System remains stable under adversarial conditions

---

## Research Contributions

* Trust-aware federated aggregation framework
* Confidence-enhanced trust scoring mechanism
* Robustness against label poisoning attacks
* Drift-resilient federated intrusion detection
* Scalable evaluation across multiple client configurations
* Integration of Zero Trust principles with federated learning

---

## Important Notes

* Extract dataset before use
* Use relative path:

  ```
  data/Final_5Class_IDS.csv
  ```
* Do not use absolute paths (e.g., C:\Users...)

---

## Common Errors

| Error          | Cause                 | Fix            |
| -------------- | --------------------- | -------------- |
| File not found | Dataset not extracted | Unzip file     |
| Wrong path     | Using absolute path   | Use `data/...` |
| Read error     | Using `.zip` file     | Use `.csv`     |

---

## Future Work

* Real-time deployment on edge/IoT environments
* Blockchain-based trust validation
* Adaptive trust scoring
* Advanced adversarial defense mechanisms

---

## Authors

* S.M
* M.R
* S.T

---

## License

For academic and research use only.

