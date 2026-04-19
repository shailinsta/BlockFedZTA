# Data Directory

## Overview

This folder is used to store the dataset required to run the BlockFedZTA pipeline.

The dataset is not included in the repository due to size constraints and is hosted externally.

---

## Dataset Source

The dataset is available on Hugging Face:

https://huggingface.co/datasets/ShamsTahzib/BlockFedZTA-Dataset

---

## Required File

```text id="0e5q3p"
Final_5Class_IDS.csv
```

This file contains the fully processed dataset used for training and evaluation.

---

## Setup Instructions

### Step 1 — Download Dataset

Download the required CSV file from the dataset link above.

---

### Step 2 — Place the File

Place the file inside this directory:

```text id="qvzg2u"
data/Final_5Class_IDS.csv
```

---

## Usage

The pipeline expects the dataset to be available at:

```text id="y05k8p"
data/Final_5Class_IDS.csv
```

Make sure the path in `src/run_pipeline.py` matches this location.

---

## Notes

* Do not rename the file
* Ensure the dataset is fully downloaded before running
* The file must contain the `label` column used for classification

---

## Common Issues

| Problem        | Cause                        | Solution              |
| -------------- | ---------------------------- | --------------------- |
| File not found | Dataset not placed correctly | Move file to `data/`  |
| Incorrect path | Path mismatch in code        | Update path in script |
| Missing column | Wrong dataset file           | Use correct CSV       |

---

## Summary

| Step | Action                |
| ---- | --------------------- |
| 1    | Download dataset      |
| 2    | Place file in `data/` |
| 3    | Run pipeline          |

---

This setup ensures the dataset is correctly configured for running the project.
