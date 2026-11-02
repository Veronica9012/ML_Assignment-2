import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Wine Quality ML Classifier", layout="wide")

st.title("🍷 Wine Quality Classification Evaluator")
st.write("BITS Pilani WILP - Machine Learning Assignment 2")

# Sidebar for user inputs
st.sidebar.header("1. Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

model_option = st.sidebar.selectbox(
    "2. Select Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

# Load saved models and scaler from root directory
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    models = {
        "Logistic Regression": joblib.load("logistic_regression.pkl"),
        "Decision Tree": joblib.load("decision_tree.pkl"),
        "KNN": joblib.load("knn.pkl"),
        "Naive Bayes": joblib.load("naive_bayes.pkl"),
        "Support Vector Machine": joblib.load("support_vector_machine.pkl")
    }
    return scaler, models

scaler, models = load_artifacts()

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    if "target" not in test_df.columns:
        st.error("Uploaded CSV must contain the 'target' column.")
    else:
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]
        
        # Apply scaling if the model requires scaled input
        if model_option in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test.values

        # Load selected model & generate predictions
        model = models[model_option]
        y_pred = model.predict(X_eval)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_eval)[:, 1]
        else:
            y_proba = y_pred

        # Compute Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader(f"📊 Evaluation Metrics - {model_option}")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")
        
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.subheader("📌 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                        xticklabels=["Class 0", "Class 1"], 
                        yticklabels=["Class 0", "Class 1"])
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
            
        with right_col:
            st.subheader("📄 Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.4f}"))

else:
    st.info("👈 Please upload test_data.csv from the sidebar to evaluate model predictions.")
