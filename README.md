#  BlockFedZTA: Federated Learning-Based Zero Trust Intrusion Detection

##  Overview

**BlockFedZTA** is a research-driven framework that integrates **Federated Learning (FL)** with **Zero Trust Architecture (ZTA)** for robust and privacy-preserving intrusion detection.

The system is designed for distributed environments such as IoT and edge networks, enabling secure and decentralized model training.

---

##  Key Features

*  Zero Trust Security Integration
*  Federated Learning Pipeline
*  Drift-aware evaluation:

  * No Drift
  * Mild Drift
  * Severe Drift
*  Trust-based client aggregation
*  Scalable and research-ready

---

##  Dataset

The dataset is included in this repository as a compressed file:

```text
data/Final_5Class_IDS.zip
```

This dataset is a processed and merged combination of:

* NSL-KDD
* CIC-IDS
* TON-IoT

---

##  Dataset Setup (Important)

###  Step 1 — Extract the Dataset

Go to the `data/` folder and unzip:

```text
Final_5Class_IDS.zip
```

After extraction, you should have:

```text
data/Final_5Class_IDS.csv
```

---

###  Step 2 — Update Path in Code

Open:

```text
src/run_pipeline.py
```

Find this line:

```python
df = pd.read_csv("YOUR_PATH_HERE")
```

Replace it with:

```python
df = pd.read_csv("data/Final_5Class_IDS.csv")
```

---

###  Step 3 — Run the Project

```bash
python src/run_pipeline.py
```

---

##  Project Structure

```text
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

##  Pipeline Workflow

1. Load dataset
2. Clean and preprocess features
3. Encode labels
4. Scale features
5. Simulate federated clients
6. Apply drift scenarios
7. Train models (XGBoost)
8. Trust-based aggregation
9. Evaluate performance

---

##  Evaluation Metrics

* Accuracy
* F1 Score
* Robustness under drift

---

##  Example Results

| Scenario     | Accuracy | F1 Score |
| ------------ | -------- | -------- |
| No Drift     | ~0.96    | ~0.96    |
| Mild Drift   | ~0.94    | ~0.94    |
| Severe Drift | ~0.92    | ~0.92    |

---

##  Research Contributions

* Federated Learning for IDS
* Trust-aware aggregation mechanism
* Drift-resilient evaluation
* Zero Trust integration in distributed ML

---

##  Important Notes

* Do NOT use the `.zip` file directly
* Always extract before running
* Use relative path:

  ```text
  data/Final_5Class_IDS.csv
  ```
* Do NOT use absolute paths like:

  ```text
  C:\Users\...
  ```

---

##  Common Errors

| Error          | Cause                 | Fix            |
| -------------- | --------------------- | -------------- |
| File not found | Dataset not extracted | Unzip file     |
| Wrong path     | Using system path     | Use `data/...` |
| Read error     | Using `.zip` file     | Use `.csv`     |

---

## Future Work

* Real-time deployment on edge devices
* Blockchain-based trust validation
* Adaptive trust scoring

---

##  Authors

* **Shailendra Mishra**
* **Megha Rathi**
* **Shams Tahzib**

---

## 📜 License

For academic and research use.
