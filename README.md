\# BlockFedZTA: Federated Learning-Based Intrusion Detection System



\## Overview

This project implements a \*\*Federated Learning-based Zero Trust Architecture (ZTA)\*\* for intrusion detection using multiple datasets (NSL-KDD, CIC-IDS, TON-IoT).



The pipeline includes:

\- Data preprocessing \& merging

\- Feature engineering and scaling

\- Model training (Random Forest / ML models)

\- Evaluation (accuracy, confusion matrix, ROC)



\---



\##  Project Structure



BlockFedZTA/

│

├── data/

│ ├── raw/ # Original datasets

│ └── processed/ # Cleaned \& transformed datasets

│ └── arrays/ # NumPy arrays (X, y, predictions)

│

├── models/ # Trained models \& preprocessors

├── results/

│ ├── figures/ # Plots (ROC, confusion matrix)

│ └── tables/ # Evaluation tables

│

├── notebooks/ # Jupyter notebooks (experiments)

├── src/ # Source code

│ └── run\_pipeline.py # Main pipeline script

│

├── docs/ # Documentation

└── README.md





\---



\##  Pipeline Workflow



1\. Load datasets (NSL, CIC, TON)

2\. Clean and preprocess data

3\. Merge datasets into unified format

4\. Scale and transform features

5\. Train model

6\. Evaluate performance

7\. Save outputs (model, predictions, metrics)



\---



\##  How to Run



\### Step 1: Install dependencies

```bash

pip install pandas numpy scikit-learn matplotlib seaborn joblib





python src/run\_pipeline.py



&#x20;Outputs



After running the pipeline:



data/processed/final\_df.csv → Final dataset

models/model.pkl → Trained model

models/scaler.pkl → Scaler

results/ → Predictions and evaluation results



&#x20;Datasets Used

NSL-KDD

CIC-IDS

TON-IoT



&#x20;Evaluation Metrics

Accuracy

F1-score

Confusion Matrix

ROC Curve



&#x20;Notes

This project focuses on federated and distributed IDS

Designed for research and reproducibility

Extendable to deep learning or blockchain integration



&#x20;Authors



Shailendra Mishra

Megha Rathi

Shams Tahzib



📜 License



For academic and research use.

