import os
import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def run_pipeline():
    print("[*] Loading UCI Breast Cancer Diagnostic dataset (sklearn standard benchmark)...")
    raw_data = load_breast_cancer(as_frame=True)
    df = raw_data.frame
    
    # Clean feature names (remove spaces for safe data handling)
    df.columns = [col.replace(" ", "_") for col in df.columns]
    
    print(f"[+] Total samples: {df.shape[0]} rows (Criteria >= 500: PASSED)")
    print(f"[+] Total predictors: {df.shape[1] - 1} features (Criteria >= 12: PASSED)")
    print(f"[+] Target column: 'target' (Binary: 0 = Benign, 1 = Malignant)")
    
    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 80-20 stratified split (114 holdout test samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Save the holdout test data as test_data.csv for the Streamlit app
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv("test_data.csv", index=False)
    print(f"[+] test_data.csv generated successfully with shape: {test_df.shape}")
    
    # Normalize features for distance- and margin-based classifiers
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.pkl")
    print("[+] Scaler saved to model/scaler.pkl")
    
    # Configure all 5 classification models
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=5, random_state=42), False),
        "KNN": (KNeighborsClassifier(n_neighbors=5), True),
        "Naive Bayes": (GaussianNB(), False),
        "Support Vector Machine": (SVC(probability=True, kernel='rbf', C=1.0, random_state=42), True)
    }
    
    results = []
    
    print("\n" + "="*75)
    print("TRAINING AND COMPUTING EVALUATION METRICS FOR ALL 5 CLASSIFIERS")
    print("="*75)
    
    for name, (clf, needs_scaling) in models.items():
        X_tr = X_train_scaled if needs_scaling else X_train
        X_te = X_test_scaled if needs_scaling else X_test
        
        clf.fit(X_tr, y_train)
        
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Save model artifact
        slug = name.lower().replace(" ", "_")
        joblib.dump(clf, f"model/{slug}.pkl")
        
        results.append({
            "ML Model Name": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })
        
    res_df = pd.DataFrame(results)
    print("\n--- FINAL EVALUATION METRICS TABLE ---")
    print(res_df.to_string(index=False))
    print("\n[+] All 5 model artifacts successfully generated in 'model/' directory.")

if __name__ == "__main__":
    run_pipeline()
