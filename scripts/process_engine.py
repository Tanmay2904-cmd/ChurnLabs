import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. AUTO-DETECT CHURN COLUMN
# ─────────────────────────────────────────────
def detect_churn_column(df: pd.DataFrame) -> str | None:
    """Try to auto-detect which column is the churn label."""
    candidates = [c for c in df.columns if "churn" in c.lower()]
    if candidates:
        return candidates[0]
    # Fallback: look for binary columns named 'label', 'target', 'churned'
    for kw in ["label", "target", "churned", "attrition", "left"]:
        for c in df.columns:
            if kw in c.lower():
                return c
    return None


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, churn_col: str, id_col: str = None):
    """
    Preprocess dataframe:
    - Drop ID column if given
    - Convert TotalCharges / numeric-looking object cols
    - Label encode categoricals
    - Return X, y, feature_names, encoders dict
    """
    df = df.copy()

    # Drop ID column
    if id_col and id_col in df.columns:
        df.drop(columns=[id_col], inplace=True)

    # Separate target
    y_raw = df[churn_col].copy()
    df.drop(columns=[churn_col], inplace=True)

    # Encode target → binary (always via string mapping to handle Yes/No, 1/0, True/False)
    y_str = y_raw.astype(str).str.strip().str.lower()
    y_raw = y_str.map(lambda v: 1 if v in ["yes", "1", "true", "churn", "1.0"] else 0)

    # Convert all object cols that look numeric
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() / len(df) > 0.8:
            df[col] = converted

    # Fill numeric NaNs with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Label-encode remaining categoricals
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_names = list(df.columns)
    return df, y_raw, feature_names, encoders


# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────
def train_models(X: pd.DataFrame, y: pd.Series):
    """
    Train Logistic Regression, Random Forest, XGBoost.
    Returns dict with model objects, scaler, metrics, best model name.
    """
    # Handle tiny datasets gracefully
    test_size = 0.2 if len(X) > 20 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() == 2 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":             xgb.XGBClassifier(
                                   n_estimators=100, random_state=42,
                                   eval_metric="logloss", verbosity=0
                               ),
    }

    results    = {}
    trained    = {}
    best_name  = None
    best_score = -1

    for name, model in models.items():
        use_scaled = name == "Logistic Regression"
        Xtr = X_train_scaled if use_scaled else X_train.values
        Xte = X_test_scaled  if use_scaled else X_test.values

        model.fit(Xtr, y_train)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics = {
            "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
            "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0),        4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5, 4),
            "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        results[name] = metrics
        trained[name] = model

        if metrics["F1 Score"] > best_score:
            best_score = metrics["F1 Score"]
            best_name  = name

    return trained, scaler, results, best_name


# ─────────────────────────────────────────────
# 4. BATCH PREDICT ON FULL DATASET
# ─────────────────────────────────────────────
def predict_batch(model, scaler, X: pd.DataFrame, use_scaled: bool = False):
    """
    Predict churn probability for all rows.
    Returns numpy array of probabilities.
    """
    Xv = scaler.transform(X) if use_scaled else X.values
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xv)[:, 1]
    else:
        proba = model.predict(Xv).astype(float)
    return proba


# ─────────────────────────────────────────────
# 5. RISK CLASSIFICATION
# ─────────────────────────────────────────────
def classify_risk(prob: float) -> str:
    if prob >= 0.70:
        return "🔴 High"
    elif prob >= 0.40:
        return "🟡 Medium"
    else:
        return "🟢 Low"


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from tree-based or linear models."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 7. RETENTION RECOMMENDATIONS
# ─────────────────────────────────────────────
RULE_BOOK = [
    # (column_keyword, value_condition_fn, recommendation)
    ("contract",         lambda v: "month" in str(v).lower(),         "Offer a discounted annual/2-year contract to reduce churn risk."),
    ("tenure",           lambda v: float(v) < 12,                      "New customer (<1 yr) — enroll in an onboarding loyalty program."),
    ("monthlycharges",   lambda v: float(v) > 80,                      "High bill detected — offer a personalised cost-reduction bundle."),
    ("internetservice",  lambda v: "fiber" in str(v).lower(),          "Fiber optic user — ensure SLA uptime; offer free speed upgrade."),
    ("onlinesecurity",   lambda v: str(v).lower() in ["no", "0"],      "No online security — offer free 3-month security add-on."),
    ("techsupport",      lambda v: str(v).lower() in ["no", "0"],      "No tech support — provide priority support for 6 months."),
    ("paymentmethod",    lambda v: "electronic check" in str(v).lower(),"Electronic check users churn more — incentivise auto-pay switch."),
    ("seniorcitizen",    lambda v: str(v) in ["1", "yes", "true"],     "Senior citizen — assign dedicated account manager."),
]

def generate_recommendations(row: pd.Series) -> list[str]:
    """Generate retention recommendations for a single customer row."""
    recs = []
    for col_kw, condition, advice in RULE_BOOK:
        for col in row.index:
            if col_kw in col.lower():
                try:
                    if condition(row[col]):
                        recs.append(advice)
                except Exception:
                    pass
                break
    if not recs:
        recs.append("Monitor customer satisfaction scores proactively.")
    return recs
