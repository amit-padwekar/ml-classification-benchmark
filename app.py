import io
import os
import csv
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
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
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Machine Learning Evaluation & Observability Suite",
    layout="wide"
)

st.title("Machine Learning Classification Evaluation & Observability Suite")
st.markdown(
    "A standardized platform for evaluating, stress-testing, and monitoring classification models. "
    "Designed for both technical teams and business stakeholders to evaluate model safety, "
    "class balance, and decision confidence."
)

st.sidebar.header("Platform Navigation")

model_names = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Naive Bayes",
    "Support Vector Machine"
]

selected_model_name = st.sidebar.selectbox(
    "Select Model to Inspect:",
    model_names
)

# Robust dynamic CSV parser
def parse_csv_safe(file_or_path):
    if hasattr(file_or_path, "read"):
        raw_bytes = file_or_path.read()
        file_or_path.seek(0)
        sample = raw_bytes[:8192].decode('utf-8', errors='ignore')
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
            sep = dialect.delimiter
        except Exception:
            sep = ';' if ';' in sample.splitlines()[0] else ','
        return pd.read_csv(io.StringIO(raw_bytes.decode('utf-8', errors='ignore')), sep=sep)
    else:
        with open(file_or_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
            sep = dialect.delimiter
        except Exception:
            sep = ';' if ';' in sample.splitlines()[0] else ','
        return pd.read_csv(file_or_path, sep=sep)

try:
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button(
            label="Download Benchmark Test CSV",
            data=f,
            file_name="benchmark_test_data.csv",
            mime="text/csv",
            help="Download the verified 30-feature holdout test CSV."
        )
except Exception:
    pass

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset for Evaluation (CSV)",
    type=["csv"],
    help="Supports standard tabular datasets or test sets with comma, semicolon, or tab separation."
)

if uploaded_file is not None:
    df_raw = parse_csv_safe(uploaded_file)
    data_label = f"Uploaded: {uploaded_file.name}"
else:
    try:
        df_raw = parse_csv_safe("test_data.csv")
        data_label = "Verified Benchmark Holdout Set (test_data.csv)"
    except Exception:
        df_raw = None
        st.warning("No evaluation dataset available. Please upload a CSV.")

if df_raw is not None:
    st.caption(f"Active Data: **{data_label}** | Total Rows: **{df_raw.shape[0]}** | Total Columns: **{df_raw.shape[1]}**")
    
    # Auto-detect target column
    potential_targets = ['target', 'diagnosis', 'class', 'label', 'price_range', 'output', 'y', 'G3', 'admit']
    target_candidate = None
    for c in potential_targets:
        if c in df_raw.columns:
            target_candidate = c
            break
            
    if target_candidate is None:
        target_candidate = st.sidebar.selectbox("Select Target Column to Predict:", list(df_raw.columns), index=len(df_raw.columns)-1)
    
    # Robust automated feature processing
    df_processed = df_raw.copy()
    
    # Drop pure ID/Index columns if present
    for col in df_processed.columns:
        if col.lower() in ['id', 'index', 'unnamed: 0']:
            df_processed.drop(columns=[col], inplace=True)
            
    # Process target
    y_raw = df_processed[target_candidate]
    if y_raw.dtype == 'object' or str(y_raw.dtype) == 'category':
        y_encoded = LabelEncoder().fit_transform(y_raw.astype(str))
        class_0_label = f"Class 0 ({str(np.unique(y_raw)[0])})"
        class_1_label = f"Class 1 ({str(np.unique(y_raw)[1])})" if len(np.unique(y_raw)) > 1 else "Class 1"
    else:
        if y_raw.nunique() > 2:
            median_val = y_raw.median()
            y_encoded = (y_raw >= median_val).astype(int)
            class_0_label = f"Below Median (< {median_val})"
            class_1_label = f"Above/Equal Median (>= {median_val})"
        else:
            y_encoded = y_raw.astype(int)
            class_0_label = "Negative / Healthy (0)"
            class_1_label = "Positive / Critical (1)"
            
    X_raw = df_processed.drop(columns=[target_candidate])
    
    # Process features: encode text/strings into numeric
    X_clean = pd.DataFrame(index=X_raw.index)
    for col in X_raw.columns:
        if X_raw[col].dtype in ['int64', 'float64']:
            X_clean[col] = X_raw[col].fillna(X_raw[col].median())
        else:
            # Check if column is high cardinality text (passages/NLP)
            if X_raw[col].astype(str).str.len().mean() > 40:
                tfidf = TfidfVectorizer(max_features=10)
                tfidf_mat = tfidf.fit_transform(X_raw[col].astype(str).fillna("")).toarray()
                for idx in range(tfidf_mat.shape[1]):
                    X_clean[f"{col}_tfidf_{idx}"] = tfidf_mat[:, idx]
            else:
                X_clean[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))
                
    # Detect pre-trained benchmark state
    is_benchmark = (X_clean.shape[1] == 30 and uploaded_file is None)
    
    def load_model_from_disk(name):
        slugs = {
            "Logistic Regression": ["logistic_regression.pkl"],
            "Decision Tree": ["decision_tree.pkl"],
            "K-Nearest Neighbors": ["knn.pkl", "k-nearest_neighbor.pkl"],
            "Naive Bayes": ["naive_bayes.pkl", "gaussian_nb.pkl"],
            "Support Vector Machine": ["support_vector_machine.pkl", "svm.pkl"]
        }
        for fn in slugs.get(name, []):
            path = os.path.join("model", fn)
            if os.path.exists(path):
                return joblib.load(path)
        return None

    needs_scaling = ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]

    if is_benchmark:
        scaler = joblib.load("model/scaler.pkl")
        X_test_scaled = scaler.transform(X_clean)
        X_eval_map = {m: (X_test_scaled if m in needs_scaling else X_clean.values) for m in model_names}
        models_dict = {m: load_model_from_disk(m) for m in model_names}
        y_test_final = y_encoded
    else:
        # Dynamic split and fit for custom uploads
        X_tr, X_te, y_tr, y_te = train_test_split(X_clean, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)
        dyn_scaler = StandardScaler()
        X_tr_s = dyn_scaler.fit_transform(X_tr)
        X_te_s = dyn_scaler.transform(X_te)
        
        models_dict = {}
        X_eval_map = {}
        for m in model_names:
            if m == "Logistic Regression":
                clf = LogisticRegression(max_iter=1000, random_state=42)
            elif m == "Decision Tree":
                clf = DecisionTreeClassifier(max_depth=5, random_state=42)
            elif m == "K-Nearest Neighbors":
                clf = KNeighborsClassifier(n_neighbors=5)
            elif m == "Naive Bayes":
                clf = GaussianNB()
            elif m == "Support Vector Machine":
                clf = SVC(probability=True, kernel='rbf', random_state=42)
                
            tr_inp = X_tr_s if m in needs_scaling else X_tr
            te_inp = X_te_s if m in needs_scaling else X_te
            clf.fit(tr_inp, y_tr)
            models_dict[m] = clf
            X_eval_map[m] = te_inp
            
        y_test_final = y_te

    # Evaluate active model
    current_model = models_dict[selected_model_name]
    X_active = X_eval_map[selected_model_name]
    
    y_pred = current_model.predict(X_active)
    y_prob = current_model.predict_proba(X_active)[:, 1] if hasattr(current_model, "predict_proba") else None
    
    # Calculate metrics
    acc = accuracy_score(y_test_final, y_pred)
    auc = roc_auc_score(y_test_final, y_prob) if y_prob is not None else 0.0
    prec = precision_score(y_test_final, y_pred, zero_division=0)
    rec = recall_score(y_test_final, y_pred, zero_division=0)
    f1 = f1_score(y_test_final, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test_final, y_pred)

    st.markdown(f"### Performance Overview: **{selected_model_name}**")
    
    # Clear metric cards showing percentages and decimal notation
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc*100:.2f}%", f"{acc:.4f} score", help="Proportion of total predictions correctly identified.")
    c2.metric("ROC-AUC", f"{auc*100:.2f}%", f"{auc:.4f} score", help="Ability of the model to distinguish between positive and negative classes across thresholds.")
    c3.metric("Precision", f"{prec*100:.2f}%", f"{prec:.4f} score", help="Out of all cases predicted as positive, how many were truly positive?")
    c4.metric("Recall", f"{rec*100:.2f}%", f"{rec:.4f} score", help="Out of all actual positive cases, how many did the model detect?")
    c5.metric("F1-Score", f"{f1*100:.2f}%", f"{f1:.4f} score", help="Harmonic balance between Precision and Recall.")
    c6.metric("MCC Score", f"{mcc:.4f}", "Correlation (-1 to +1)", help="Matthews Correlation Coefficient: balanced measure robust against class imbalance.")


    # c1, c2, c3, c4, c5, c6 = st.columns(6)
    # c1.metric("Accuracy", f"{acc*100:.2f}%", help="Proportion of total predictions correctly identified.")
    # c2.metric("ROC-AUC", f"{auc*100:.2f}%", help="Ability of the model to distinguish between positive and negative classes.")
    # c3.metric("Precision", f"{prec*100:.2f}%", help="Out of all cases predicted as positive, how many were truly positive?")
    # c4.metric("Recall", f"{rec*100:.2f}%", help="Out of all actual positive cases, how many did the model detect?")
    # c5.metric("F1-Score", f"{f1*100:.2f}%", help="Harmonic balance between Precision and Recall.")
    # c6.metric("MCC Score", f"{mcc:.4f}", help="Matthews Correlation Coefficient (-1 to +1).")

    st.markdown("---")

    tab_diag, tab_benchmark, tab_observability = st.tabs([
        "Diagnostics & Prediction Confidence",
        "Cross-Model Comparative Benchmarking",
        "Data Observability & Feature Inspection"
    ])

    with tab_diag:
        d1, d2 = st.columns([1, 1])

        with d1:
            st.subheader("Confusion Matrix")
            st.caption("Visual breakdown of correct predictions vs misclassifications.")
            cm = confusion_matrix(y_test_final, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.2))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[class_0_label, class_1_label],
                yticklabels=[class_0_label, class_1_label],
                cbar=False, ax=ax_cm
            )
            ax_cm.set_xlabel("Model Predicted Label")
            ax_cm.set_ylabel("Actual True Label")
            ax_cm.set_title(f"Confusion Matrix: {selected_model_name}")
            st.pyplot(fig_cm)

        with d2:
            st.subheader("Prediction Confidence Distribution")
            st.caption("How confident the model was in its decisions (0% = confident Class 0, 100% = confident Class 1).")
            if y_prob is not None:
                fig_prob, ax_prob = plt.subplots(figsize=(5, 3.2))
                sns.histplot(y_prob, bins=15, kde=True, color="#1f77b4", ax=ax_prob)
                ax_prob.axvline(0.5, color="red", linestyle="--", label="Decision Cutoff (50%)")
                ax_prob.set_xlabel("Confidence Probability for Positive Class")
                ax_prob.set_ylabel("Number of Samples")
                ax_prob.set_title("Certainty Profile & Boundary Separation")
                ax_prob.legend()
                st.pyplot(fig_prob)
            else:
                st.info("Probability distributions are not exposed for this model configuration.")

        st.subheader("Detailed Classification Report")
        st.caption("Performance breakdown per category.")
        cr_dict = classification_report(
            y_test_final, y_pred,
            target_names=[class_0_label, class_1_label],
            output_dict=True
        )
        report_df = pd.DataFrame(cr_dict).transpose()
        st.dataframe(report_df.style.format(precision=4), use_container_width=True)

    with tab_benchmark:
        st.subheader("Comparative Benchmark Across All 5 Classifiers")
        st.caption("Compare how each algorithm performs on the same test set.")
        
        comp_records = []
        for m in model_names:
            clf_obj = models_dict[m]
            x_test_subset = X_eval_map[m]
            
            p_preds = clf_obj.predict(x_test_subset)
            p_probs = clf_obj.predict_proba(x_test_subset)[:, 1] if hasattr(clf_obj, "predict_proba") else None
            
            comp_records.append({
                "Model Name": m,
                "Accuracy (%)": round(accuracy_score(y_test_final, p_preds) * 100, 2),
                "ROC-AUC (%)": round(roc_auc_score(y_test_final, p_probs) * 100 if p_probs is not None else 0.0, 2),
                "Precision (%)": round(precision_score(y_test_final, p_preds, zero_division=0) * 100, 2),
                "Recall (%)": round(recall_score(y_test_final, p_preds, zero_division=0) * 100, 2),
                "F1 Score (%)": round(f1_score(y_test_final, p_preds, zero_division=0) * 100, 2),
                "MCC Score": round(matthews_corrcoef(y_test_final, p_preds), 4)
            })

        comp_df = pd.DataFrame(comp_records).set_index("Model Name")
        st.dataframe(comp_df, use_container_width=True)

        st.subheader("Comparative Performance Visualization")
        st.bar_chart(comp_df[["Accuracy (%)", "ROC-AUC (%)", "F1 Score (%)"]])

    with tab_observability:
        st.subheader("Input Data Sample (First 10 Records)")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        oc1, oc2 = st.columns([1, 1])
        with oc1:
            st.subheader("Class Distribution Balance")
            counts = pd.Series(y_encoded).value_counts()
            fig_cls, ax_cls = plt.subplots(figsize=(4, 2.8))
            sns.barplot(x=[class_0_label, class_1_label], y=counts.values, ax=ax_cls, palette="Blues_d")
            ax_cls.set_ylabel("Count")
            ax_cls.set_title("Target Distribution")
            st.pyplot(fig_cls)

        with oc2:
            st.subheader("Numeric Features Summary")
            st.dataframe(X_clean.describe().transpose()[["mean", "std", "min", "max"]].head(10), use_container_width=True)
