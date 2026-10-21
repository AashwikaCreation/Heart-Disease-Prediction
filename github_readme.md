# Heart Disease Prediction using Logistic Regression & SVM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Accuracy](https://img.shields.io/badge/Peak%20Accuracy-85.86%25-brightgreen.svg)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20Data%20Science-red.svg)

An end-to-end Machine Learning classification project designed to predict the presence of cardiovascular disease in patients based on clinical features using **Logistic Regression** and **Support Vector Machine (SVM)** algorithms.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death globally. Early diagnosis and prediction can significantly assist healthcare providers in implementing proactive medical interventions. 

This repository contains an end-to-end Data Science pipeline—from raw clinical data preprocessing and feature scaling to binary classification modeling and evaluation. The primary objective is to compare linear and non-linear classification models (**Logistic Regression** vs. **Support Vector Machines**) to accurately identify patients at risk of heart disease.

---

## 💡 Problem Statement & Business Impact

### Problem Statement
Given various health parameters (e.g., age, blood pressure, cholesterol, chest pain type, maximum heart rate), develop a predictive machine learning model to classify whether a patient has heart disease (`1`) or does not have heart disease (`0`).

### Business & Healthcare Impact
* **Early Medical Intervention:** Facilitates early detection for high-risk patients, allowing physicians to take preventative action before severe complications arise.
* **Resource Allocation:** Enables hospitals and clinics to prioritize diagnostic testing and care allocation efficiently.
* **Cost Reduction:** Reduces expensive emergency procedures by identifying risks early through routine clinical metrics.

---

## 📊 Dataset & Preprocessing Steps

The dataset used is the benchmark **Heart Disease UCI Dataset**.

### Preprocessing Pipeline:
1. **Identifier Cleanup:** Removed unnecessary identifier columns (e.g., `id`).
2. **Missing Value Imputation:**
   * **Numerical Columns:** Imputed using **Median Imputation** via `SimpleImputer` to minimize outlier sensitivity.
   * **Categorical Columns:** Imputed using **Most Frequent (Mode) Imputation**.
3. **Encoding & Column Alignment:**
   * Applied **One-Hot Encoding** (`pd.get_dummies`) for categorical variables.
   * Utilized `.align()` to ensure identical feature columns across train and test sets.
4. **Target Binarization:** Converted multi-class target labels (`0–4`) into a clean binary classification target (`0 = No Disease`, `1 = Has Heart Disease`).
5. **Feature Scaling:** Standardized all numeric features using `StandardScaler` to optimize distance-based optimization in Logistic Regression and SVM.
6. **Data Splitting:** Stratified 80/20 train-test split (`train_test_split`) with `random_state=42`.

---

## ⚙️ Model Training & Comparison

Two core machine learning models were trained and evaluated on unseen test data ($N = 184$):

### 1. Logistic Regression
* **Max Iterations:** 1000
* **Accuracy:** **83.15%**
* **Strengths:** High interpretability, fast inference time, and reliable probabilistic outputs.

### 2. Support Vector Machine (SVM)
* **Kernel:** Radial Basis Function (`rbf`)
* **Accuracy:** **85.86%** *(Top Performing Model)*
* **Strengths:** Superior performance in capturing complex, non-linear relationships in higher-dimensional feature spaces.

---

## 📈 Detailed Performance Metrics

| Metric | Logistic Regression | Support Vector Machine (SVM) |
| :--- | :---: | :---: |
| **Overall Accuracy** | **83.15%** | **85.86%** |
| **Precision (Class 1 - Disease)** | 0.87 | **0.90** |
| **Recall (Class 1 - Disease)** | 0.84 | **0.85** |
| **F1-Score (Class 1 - Disease)** | 0.86 | **0.88** |
| **Macro Average F1** | 0.83 | **0.86** |

---

## 🛠️ Tech Stack & Tools

* **Programming Language:** Python
* **Data Processing & Analysis:** Pandas, NumPy
* **Machine Learning Library:** Scikit-Learn (`SimpleImputer`, `StandardScaler`, `LogisticRegression`, `SVC`, `metrics`)
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebook

---

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AashwikaCreation/Heart-Disease-Prediction-ML.git
   cd Heart-Disease-Prediction-ML
   ```

2. **Install required dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Run the Notebook / Script:**
   * Open `Heart_Disease_Prediction.ipynb` in Google Colab or Jupyter Notebook.
   * Ensure `heart_disease_uci.csv` is uploaded in the root directory.
   * Run all cells sequentially to train the models and output metrics.

---

## 📝 Conclusion & Future Work

* **Key Takeaway:** The **SVM Classifier (RBF Kernel)** achieved the highest accuracy at **85.86%** with an exceptional **0.90 Precision** for identifying high-risk heart disease patients.
* **Future Improvements:**
  * Perform Hyperparameter Tuning (`GridSearchCV`) for SVM ($C$ and $\gamma$ values).
  * Experiment with Ensemble algorithms like **Random Forest** and **XGBoost**.
  * Deploy the best model as a web tool using **Streamlit** or **Flask**.

---

## 👤 Author

**Nikita Raut**  
*Aspiring Machine Learning Engineer & Data Scientist*  
* **GitHub:** [@AashwikaCreation](https://github.com/AashwikaCreation)  
* **LinkedIn:** [linkedin.com/aashwika](https://linkedin.com/aashwika)