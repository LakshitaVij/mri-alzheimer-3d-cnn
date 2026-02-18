

#  3D MRI Alzheimer Classification (OASIS-1)

A volumetric deep learning pipeline for Alzheimer’s disease classification using 3D MRI scans from the OASIS-1 dataset.

This project implements a 3D Convolutional Neural Network (CNN) for binary classification (CDR = 0 vs CDR > 0), with reproducible train/validation/test splits, early stopping, class weighting, and Streamlit deployment.

---

##  Model Overview

* 3D CNN with three convolutional blocks
* Batch Normalization + ReLU activations
* Progressive 3D max pooling
* Global Average Pooling
* Single logit output with `BCEWithLogitsLoss`

Evaluation metric: **AUC (Area Under ROC Curve)**
Test AUC ≈ **0.74**

---

##  Why AUC?

Given class imbalance and a small dataset (~200 scans), AUC was chosen over accuracy because it evaluates ranking quality across all thresholds rather than relying on a fixed cutoff.

---

##  Features

* MRI preprocessing (normalization + trilinear resizing to 128³)
* Class imbalance handling via weighted BCE loss
* Early stopping with checkpoint saving
* Final evaluation: Accuracy, AUC, Confusion Matrix
* Streamlit app for interactive inference

---

##  Run Streamlit App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `.mgz`, `.nii`, or `.nii.gz` MRI files to get predictions.

---

## ⚠️ Note

MRI data and trained weights are not included due to dataset licensing and size constraints.


