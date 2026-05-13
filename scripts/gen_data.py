import pandas as pd
import numpy as np

np.random.seed(42)
n = 300

contracts       = np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.55,0.25,0.20])
internet        = np.random.choice(['DSL','Fiber optic','No'], n, p=[0.35,0.45,0.20])
payment         = np.random.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'], n, p=[0.35,0.23,0.22,0.20])
tenure          = np.where(contracts=='Month-to-month', np.random.randint(1,36,n), np.random.randint(12,72,n))
monthly_charges = np.where(internet=='Fiber optic', np.random.uniform(70,110,n), np.where(internet=='DSL', np.random.uniform(25,65,n), np.random.uniform(18,30,n)))
total_charges   = tenure * monthly_charges + np.random.uniform(-20,20,n)
total_charges   = np.clip(total_charges, 0, None).round(2)
monthly_charges = monthly_charges.round(2)

online_sec  = np.random.choice(['Yes','No','No internet service'], n, p=[0.30,0.50,0.20])
tech_sup    = np.random.choice(['Yes','No','No internet service'], n, p=[0.28,0.52,0.20])
phone_svc   = np.random.choice(['Yes','No'], n, p=[0.90,0.10])
multiple_ln = np.where(phone_svc=='No', 'No phone service', np.random.choice(['Yes','No'], n, p=[0.45,0.55]))
gender      = np.random.choice(['Male','Female'], n)
senior      = np.random.choice([0,1], n, p=[0.84,0.16])
partner     = np.random.choice(['Yes','No'], n, p=[0.48,0.52])
dependents  = np.random.choice(['Yes','No'], n, p=[0.30,0.70])
paperless   = np.random.choice(['Yes','No'], n, p=[0.60,0.40])
online_bk   = np.random.choice(['Yes','No','No internet service'], n, p=[0.28,0.52,0.20])
dev_prot    = np.random.choice(['Yes','No','No internet service'], n, p=[0.28,0.52,0.20])
stream_tv   = np.random.choice(['Yes','No','No internet service'], n, p=[0.38,0.42,0.20])
stream_mv   = np.random.choice(['Yes','No','No internet service'], n, p=[0.38,0.42,0.20])

churn_score = (
    (contracts == 'Month-to-month').astype(float) * 0.40 +
    (internet  == 'Fiber optic').astype(float)    * 0.20 +
    (tenure    < 12).astype(float)                * 0.20 +
    (online_sec == 'No').astype(float)            * 0.10 +
    (payment   == 'Electronic check').astype(float) * 0.10 +
    np.random.uniform(-0.15, 0.15, n)
)
churn_score = np.clip(churn_score, 0, 1)
churn_label = (churn_score > 0.48).astype(int)

ids = ['CUST-' + str(i).zfill(4) for i in range(1, n+1)]

df = pd.DataFrame({
    'customerID': ids,
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_svc,
    'MultipleLines': multiple_ln,
    'InternetService': internet,
    'OnlineSecurity': online_sec,
    'OnlineBackup': online_bk,
    'DeviceProtection': dev_prot,
    'TechSupport': tech_sup,
    'StreamingTV': stream_tv,
    'StreamingMovies': stream_mv,
    'Contract': contracts,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': np.where(churn_label == 1, 'Yes', 'No')
})

df.to_csv('data/sample_dataset.csv', index=False)
churn_rate = (df['Churn'] == 'Yes').sum() / len(df)
print(f"Dataset created: {len(df)} rows, Churn rate: {churn_rate:.1%}")
