# Models Directory

## Overview

This folder is intended to store trained models and preprocessing artifacts used in the BlockFedZTA pipeline.

Due to size constraints, pretrained models are not stored in this repository.

---

## Pretrained Model

The trained model is available on Hugging Face:

https://huggingface.co/ShamsTahzib/BlockFedZTA

---

## Usage

You can download and load the pretrained model directly from Hugging Face.

Example:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("ShamsTahzib/BlockFedZTA")
```

---

## Local Storage (Optional)

If you prefer to work offline:

1. Download the model from Hugging Face
2. Place it inside this folder:

```text
models/
```

3. Load using your preferred framework (e.g., PyTorch, scikit-learn)

---

## Notes

* Ensure compatibility between the model and dataset
* The model expects input features consistent with `Final_5Class_IDS.csv`
* Preprocessing steps (scaling, encoding) must match training pipeline

---

## Summary

| Option  | Description                     |
| ------- | ------------------------------- |
| Online  | Load model from Hugging Face    |
| Offline | Download and place in `models/` |

---

This setup keeps the repository lightweight while allowing access to pretrained models.
