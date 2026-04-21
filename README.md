# ChurnGuard: Telecom Customer Churn Prediction AI

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/deployment-Streamlit-red.svg)

## 📌 Problem Statement
Customer churn is one of the most critical metrics for any telecommunication company. It costs significantly more to acquire a new customer than to retain an existing one. This project aims to build a predictive model that identifies high-risk customers likely to churn, allowing the marketing team to take proactive retention measures.

## 🚀 The Approach
1. **Data Exploratory Analysis (EDA)**: Investigated correlations between features like contract type, monthly charges, and tenure with the churn status.
2. **Preprocessing**: Handled missing values, encoded categorical variables (LabelEncoding), and scaled numerical features.
3. **Feature Engineering**: Focused on tenure, contract type, and internet service as primary drivers.
4. **Machine Learning Model**: 
   - **Logistic Regression**: Base model for interpretability.
   - **Random Forest**: For handling non-linear relationships.
   - **XGBoost**: For high-performance gradient boosting.
5. **Deployment**: Built a real-time prediction dashboard using **Streamlit**.

## 📊 Key Results
- **Recall (Churn Class)**: 0.82 (Priority: Catching as many churners as possible).
- **Accuracy**: 81.5%
- **Top 3 Features**: Contract Type, Tenure, Monthly Charges.

## 💻 Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Deployment**: Streamlit

## 📁 Project Structure
- `data/`: Contains the dataset.
- `notebooks/`: Jupyter notebook for the detailed EDA and experimental modeling.
- `scripts/`: Production-ready training scripts.
- `app.py`: Streamlit deployment file.
- `requirements.txt`: Project dependencies.

---
*Developed by [Tanmay Naigaonkar]*
