Heart Disease Prediction using Logistic Regression & SVMAn end-to-end Machine Learning classification project designed to predict the presence of cardiovascular disease in patients based on clinical features using Logistic Regression and Support Vector Machine (SVM) algorithms.

📌 Project Overview
Heart disease is one of the leading causes of death globally. Early diagnosis and prediction can significantly assist healthcare providers in implementing proactive medical interventions.This repository contains an end-to-end Data Science pipeline—from raw clinical data preprocessing and feature scaling to binary classification modeling and evaluation. The primary objective is to compare linear and non-linear classification models (Logistic Regression vs. Support Vector Machines) to accurately identify patients at risk of heart disease.

💡 Problem Statement & Business ImpactProblem Statement
Given various health parameters (e.g., age, blood pressure, cholesterol, chest pain type, maximum heart rate), develop a predictive machine learning model to classify whether a patient has heart disease (1) or does not have heart disease (0).

Business & Healthcare ImpactEarly Medical Intervention: 
Facilitates early detection for high-risk patients, allowing physicians to take preventative action before severe complications arise.

Resource Allocation: Enables hospitals and clinics to prioritize diagnostic testing and care allocation efficiently.

Cost Reduction: Reduces expensive emergency procedures by identifying risks early through routine clinical metrics.

📊 Dataset & Preprocessing Steps
The dataset used is the benchmark Heart Disease UCI Dataset.
Preprocessing Pipeline:
Identifier Cleanup: Removed unnecessary identifier columns (e.g., id).
Missing Value Imputation:Numerical Columns: Imputed using Median Imputation via SimpleImputer to minimize outlier sensitivity.
Categorical Columns: Imputed using Most Frequent (Mode) Imputation.
Encoding & Column Alignment:Applied One-Hot Encoding (pd.get_dummies) for categorical variables.Utilized .align() to ensure identical feature columns across train and test sets.
Target Binarization: Converted multi-class target labels (0–4) into a clean binary classification target (0 = No Disease, 1 = Has Heart Disease).
Feature Scaling: Standardized all numeric features using StandardScaler to optimize distance-based optimization in Logistic Regression and SVM.
Data Splitting: Stratified 80/20 train-test split (train_test_split) with random_state=42.

⚙️ Model Training & Comparison
Two core machine learning models were trained and evaluated on unseen test data ($N = 184$):
1. Logistic Regression
2. Max Iterations: 1000Accuracy: 83.15%Strengths: High interpretability, fast inference time, and reliable probabilistic outputs.
3.
4. 2. Support Vector Machine (SVM)Kernel: Radial Basis Function (rbf)Accuracy: 85.86%
   3. (Top Performing Model)
   4. Strengths: Superior performance in capturing complex, non-linear relationships in higher-dimensional feature spaces.

      📈 Detailed Performance Metrics
      MetricLogistic Regression
      Support Vector Machine (SVM)
      Overall Accuracy83.15%85.86%
      Precision (Class 1 - Disease)0.870.90
      Recall (Class 1 - Disease)0.840.85
      F1-Score (Class 1 - Disease)0.860.88
      Macro Average F10.830.86

      🛠️ Tech Stack & ToolsProgramming Language:
      PythonData Processing & Analysis: Pandas, NumPy
      Machine Learning Library: Scikit-Learn (SimpleImputer, StandardScaler, LogisticRegression, SVC, metrics)
      Data Visualization: Matplotlib, Seaborn
      Environment: Google Colab / Jupyter Notebook🚀 Installation & Usage
      Clone the repository:git clone https://github.com/AashwikaCreation/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML

Install required dependencies:pip install pandas numpy scikit-learn matplotlib seaborn
Run the Notebook / Script:Open Heart_Disease_Prediction.ipynb in Google Colab or Jupyter Notebook.Ensure heart_disease_uci.csv is uploaded in the root directory.Run all cells sequentially to train the models and output metrics.

📝 Conclusion & Future WorkKey Takeaway:
The SVM Classifier (RBF Kernel) achieved the highest accuracy at 85.86% with an exceptional 0.90 Precision for identifying high-risk heart disease patients.
Future Improvements:
Perform Hyperparameter Tuning (GridSearchCV) for SVM ($C$ and $\gamma$ values).
Experiment with Ensemble algorithms like Random Forest and XGBoost.
Deploy the best model as a web tool using Streamlit or Flask.👤 AuthorNikita RautAspiring Machine Learning Engineer & Data Scientist

GitHub: @AashwikaCreation
