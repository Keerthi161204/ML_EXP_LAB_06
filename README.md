# ML_EXP_LAB_06


# 📘 Decision Tree vs Random Forest (Breast Cancer Classification)

## 🔍 Overview

This project compares two supervised machine learning models:

* **Decision Tree Classifier**
* **Random Forest Classifier**

The goal is to evaluate their performance on a real-world medical dataset and understand how ensemble methods improve accuracy and stability.

---

## 🎯 Objectives

* Implement Decision Tree and Random Forest models
* Apply **cross-validation (CV)** for reliable evaluation
* Compare model performance using accuracy and classification metrics
* Analyze overfitting and generalization

---

## 📂 Dataset

* Dataset: **Breast Cancer Wisconsin Dataset** (from Scikit-learn)
* Total samples: 569
* Features: 30 numerical features
* Classes:

  * `0` → Malignant
  * `1` → Benign

---

## ⚙️ Implementation Details

### 🔹 Data Preprocessing

* Train-test split: 80% training, 20% testing
* No scaling required (tree-based models don’t need it)

---

### 🌳 Decision Tree

* Criterion: `gini`
* Max depth: `5`
* Evaluated using **5-fold cross-validation**

---

### 🌲 Random Forest

* Number of trees: `100`
* Max depth: `5`
* Ensemble method (multiple decision trees)
* Evaluated using **5-fold cross-validation**

---

## 📊 Results (Typical Output)

### Decision Tree

* Cross-validation accuracy: ~0.92 – 0.95
* Average CV accuracy: ~0.93
* Test accuracy: ~0.93

### Random Forest

* Cross-validation accuracy: ~0.95 – 0.98
* Average CV accuracy: ~0.96
* Test accuracy: ~0.96 – 0.97

---

## 📈 Key Observations

* Random Forest consistently outperforms Decision Tree

* Decision Tree tends to **overfit** even with depth restriction

* Random Forest reduces overfitting by:

  * averaging multiple trees
  * introducing randomness

* Cross-validation gives a more reliable estimate than a single train-test split

---

## 🧠 Conclusion

Random Forest provides better accuracy and generalization compared to a single Decision Tree. It is more robust and suitable for real-world applications like medical diagnosis.

---

## 🚀 How to Run

```bash
pip install numpy scikit-learn
python your_script_name.py
```




