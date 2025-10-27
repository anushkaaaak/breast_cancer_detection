# 🩺 Breast Cancer Classification and Prediction

## 🎯 Executive Summary  

This project delivers an **end-to-end Machine Learning pipeline** for classifying breast tumors as **Benign (0)** or **Malignant (1)** using the **Wisconsin Diagnostic Breast Cancer dataset**.  

The objective was to **minimize False Negatives (maximize Recall)** for the malignant class — a critical clinical requirement.  
After model experimentation, the **Random Forest Classifier** achieved:  

- **Accuracy:** 97.3%  
- **ROC-AUC:** 0.99  
- **Recall (Malignant):** 0.93  

The final model and preprocessing pipeline are serialized using **Joblib**, integrated into an **interactive prediction widget**, and **ready for real-world deployment**.  

---

## 🧠 Technical Skills & Tools  

| Category | Concepts & Highlights | Tools / Libraries |
|-----------|----------------------|-------------------|
| **Model Development** | Ensemble Learning (Random Forest), Kernel Methods (SVM), Model Selection, Bias–Variance Analysis | Scikit-learn, NumPy |
| **Data Engineering** | Data Cleaning, Feature Standardization (`StandardScaler`), Correlation Analysis, API Integration (Kaggle) | Pandas, os, joblib |
| **Evaluation & Validation** | ROC-AUC, Confusion Matrix, Precision/Recall Optimization for clinical reliability | Matplotlib, Seaborn |
| **Deployment Readiness** | Model Serialization, Interactive Demo (`ipywidgets`), Full ML Lifecycle Management | Google Colab, GitHub |

---
<img width="921" height="682" alt="image" src="https://github.com/user-attachments/assets/190771ff-c0f8-4e74-8e33-181d67b96b37" />


## 📈 Performance Comparison  

| Model | Accuracy | Recall (Malignant) | ROC-AUC | Rationale |
|--------|-----------|--------------------|----------|------------|
| **Random Forest (Selected)** | **97.37%** | **0.93** | **0.99** | High generalization, interpretable, low inference overhead |
<img width="530" height="455" alt="image" src="https://github.com/user-attachments/assets/3b1e9242-2299-40ca-9fab-0b8b485e7b86" />

| **SVM (RBF Kernel)** | 97.37% | 0.93 | 0.99 | Strong kernel-based performance |
<img width="530" height="455" alt="image" src="https://github.com/user-attachments/assets/0b5968fb-974c-4230-9e67-db61b9dcb0ed" />

| **Logistic Regression** | 96.49% | 0.93 | 0.99 | Reliable linear baseline |
<img width="530" height="455" alt="image" src="https://github.com/user-attachments/assets/019579b8-fb61-4cfa-8e32-4e2c542cf2fe" />


✅ **Decision:** The **Random Forest Classifier** was selected for its interpretability (feature importances), robustness, and ease of deployment.

---

## 🔍 Feature Correlation Insights  

Correlation analysis revealed that geometric tumor features such as  
`radius_worst`, `perimeter_worst`, and `area_worst` showed the **strongest positive correlation (0.78 – 0.79)** with malignancy — aligning with medical expectations.

---

## ⚙️ Project Highlights  

- **Data Preprocessing Pipeline:** Removed redundant columns (`id`, nulls) and standardized all 30 features to prevent scale bias.  
- **Stratified Data Split:** Ensured balanced benign/malignant ratios in both training and test sets.  
- **Deployment Simulation:** Built an **interactive prediction dashboard** with `ipywidgets` allowing user input of tumor metrics for instant classification and probability output.  
- **Model Persistence:** Serialized both the trained model and the `StandardScaler` object (`model.pkl`, `scaler.pkl`) using **Joblib**, enabling reproducible inference independent of training.  

---

## 🚀 Future Enhancements  

- **Explainable AI (XAI):** Integrate **SHAP** for feature-level interpretability and clinical transparency.  
- **Web Deployment:** Package the model using **Streamlit** or **Flask** for cloud-based inference.  
- **Hyperparameter Optimization:** Employ **GridSearchCV** or **RandomizedSearchCV** for fine-tuned model performance.  

---

## 👩‍💻 Author  

**Anushka Kandwal**  
*Final Year BTech(CSE) student *  
📎 [LinkedIn Profile](https://www.linkedin.com/in/anushka-kandwal-a9b391257)  

---

## 💡 Repository Overview  

This repository demonstrates a **production-ready ML pipeline** — from data preprocessing to model deployment — following industry best practices.  
It highlights skills in **machine learning, model evaluation, data engineering, and deployment-readiness**, making it suitable for roles in **AI, Data Science, and Healthcare ML**.

**Keywords:** `Random Forest` · `SVM` · `Machine Learning` · `Model Deployment` · `Breast Cancer Detection` · `Explainable AI` · `Healthcare AI`  
