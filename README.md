# SPEC-1-UCI Heart Disease Logistic Regression

#README

# Heart Disease Prediction & Threshold Optimization (UCI Dataset)

This repository contains a complete end-to-end pipeline for analysing the **UCI Heart Disease dataset** using **logistic regression**, diagnostic evaluation techniques, and **health-economic threshold optimisation**. The code is written in Python and structured to be transparent, modular, and reproducible.

---

## 📁 Project Structure

### **1. Data Loading & Cleaning (`load_and_clean_data`)**

* Loads the UCI dataset from a CSV file
* Identifies missing values
* Fills numeric values with **medians**
* Fills categorical values with **modes**
* Saves a cleaned version of the dataset

**Key methods:** Handling missingness, Pandas preprocessing

---

### **2. Disease Burden Analysis (`analyze_disease_burden`)**

* Explores severity distribution
* Creates age–severity boxplots
* Generates sex × severity tables

---

### **3. Feature Engineering (`prepare_features`)**

* Boolean conversion
* Sex mapping
* One-hot encoding
* Binary or multiclass targets
* Ensures full numeric feature matrix

---

### **4. Logistic Regression (`run_logistic_regression`)**

* Train/test splitting
* Standard scaling
* Binary or multinomial logistic regression
* Outputs reports, confusion matrix, ROC curve, and AUC

**Math foundations:** logistic function, log-odds model, MLE, AUC ranking probability

---

### **5. Model Diagnostics (`plot_logistic_diagnostics`)**

* ROC curves
* Probability histograms
* Coefficient plots
* Probability vs age regression

---

### **6. Advanced Thresholding (`run_logistic_with_options`)**

* Class-weight balancing
* Oversampling
* Custom thresholds
* Recomputed confusion metrics

---

### **7. Health Economics (`threshold_sweep_nmb`)**

* Cost modelling of FP and FN based on WTP (£50k)
* Threshold sweep (0.0 → 1.0)
* Computes sensitivity, specificity, cost, and NMB

**Key lines:**

```python
best_idx = results['nmb'].idxmax()
best_row = results.loc[best_idx]
```

Meaning:

* `idxmax()` finds the threshold with highest NMB
* `loc[...]` retrieves its full confusion-metric row

---

### **8. Output Files**

* Cleaned dataset
* Confusion matrices
* ROC curves
* Coefficient plots
* Threshold sweep results

---

## Key Mathematical & Statistical Concepts

### Logistic Regression

* Linear log-odds
* Sigmoid probability mapping
* Maximum likelihood estimation

### Evaluation Metrics

* Confusion matrix
* Precision, Recall, F1
* ROC Curve, AUC

### Threshold Theory

* Lower threshold → ↑ sensitivity
* Higher threshold → ↑ specificity

### Youden’s J Statistic

`J = TPR − FPR`

### Health Economics

* FN cost dominates due to QALY loss
* NICE WTP benchmark (£50k)

### Decision Theory

Maximising Net Monetary Benefit (NMB) selects the most cost-efficient threshold.
*To be completed after clarification.*


