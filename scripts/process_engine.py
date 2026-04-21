import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
import joblib

# 1. Load Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. Preprocessing
def preprocess_data(df):
    # Handle TotalCharges (convert to numeric, fill NaNs)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Encode categorical variables
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

# 3. Train Models
def train_and_evaluate(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        results[name] = {"Accuracy": acc, "Recall": rec}
        
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
    # Save the best model (Random Forest in this mock logic)
    best_model = models["Random Forest"]
    joblib.dump(best_model, 'models/churn_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return results

if __name__ == "__main__":
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
        
    data_path = 'data/telecom_churn.csv'
    df = load_data(data_path)
    df_processed = preprocess_data(df)
    train_and_evaluate(df_processed)
