import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
    page_title="Classification Model Benchmark",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Clinical Classification Benchmark Dashboard")
st.markdown(
    "Interactive evaluation platform for comparing 5 foundational machine learning models "
    "on the UCI Diagnostic benchmark dataset. Upload holdout test data to inspect predictions, metrics, and confusion matrices."
)

st.sidebar.header("Controls & Setup")

model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Select ML Classification Model:",
    list(model_options.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"],
    help="Upload test_data.csv containing all 30 predictor features and the 'target' column."
)

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("model/scaler.pkl")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

@st.cache_resource
def load_model(filename):
    try:
        return joblib.load(f"model/{filename}")
    except Exception as e:
        st.error(f"Error loading model {filename}: {e}")
        return None

scaler = load_scaler()
model = load_model(model_options[selected_model_name])

needs_scaling = selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
else:
    try:
        df_test = pd.read_csv("test_data.csv")
        st.info("Displaying default holdout 'test_data.csv' from repository. Upload another CSV via sidebar if needed.")
    except Exception:
        df_test = None
        st.warning("Please upload a CSV file to evaluate the classification models.")

if df_test is not None:
    st.subheader(f"Test Data Overview ({df_test.shape[0]} rows, {df_test.shape[1]} columns)")
    st.dataframe(df_test.head(5), use_container_width=True)

    if 'target' not in df_test.columns:
        st.error("Uploaded CSV must contain the ground-truth 'target' column for metric calculation.")
    else:
        X_eval = df_test.drop(columns=['target'])
        y_eval = df_test['target']

        if needs_scaling and scaler is not None:
            X_eval_input = scaler.transform(X_eval)
        else:
            X_eval_input = X_eval.values

        y_pred = model.predict(X_eval_input)
        y_proba = model.predict_proba(X_eval_input)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_eval, y_pred)
        auc = roc_auc_score(y_eval, y_proba) if y_proba is not None else 0.0
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_eval, y_pred)

        st.markdown(f"### Evaluation Metrics: **{selected_model_name}**")
        
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("AUC Score", f"{auc:.4f}")
        m3.metric("Precision", f"{prec:.4f}")
        m4.metric("Recall", f"{rec:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")

        tab1, tab2 = st.tabs(["📊 Confusion Matrix", "📋 Classification Report"])

        with tab1:
            cm = confusion_matrix(y_eval, y_pred)
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Benign (0)", "Malignant (1)"],
                yticklabels=["Benign (0)", "Malignant (1)"],
                cbar=False,
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix: {selected_model_name}")
            st.pyplot(fig)

        with tab2:
            report_dict = classification_report(y_eval, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)
