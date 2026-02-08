import streamlit as st
import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
from utils import FEATURES

# --- CONFIGURATION ---
MODEL_PATH = 'models/churn_pipeline.joblib'
METRICS_PATH = 'models/metrics.json'
CM_PLOT_PATH = 'models/confusion_matrix.png'

st.set_page_config(page_title="Churn Predictor", layout="wide")

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def main():
    st.title("📊 Customer Churn Prediction App")
    st.markdown("Use Machine Learning to predict if a customer will leave (churn).")

    # Load Model
    model = load_model()

    if model is None:
        st.error("Model not found! Please run `python train_model.py` first.")
        return

    # Sidebar Navigation
    page = st.sidebar.selectbox("Choose Mode", ["Single Prediction", "Batch Prediction", "Model Stats"])

    if page == "Single Prediction":
        show_single_prediction(model)
    elif page == "Batch Prediction":
        show_batch_prediction(model)
    elif page == "Model Stats":
        show_model_stats()

def show_single_prediction(model):
    st.header("Single Customer Prediction")
    
    # Create a form for inputs
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        
        # We need to collect input for all features used in training
        inputs = {}
        
        with col1:
            st.subheader("Demographics")
            inputs['gender'] = st.selectbox("Gender", ['Male', 'Female'])
            inputs['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
            inputs['Partner'] = st.selectbox("Partner", ['Yes', 'No'])
            inputs['Dependents'] = st.selectbox("Dependents", ['Yes', 'No'])
            
        with col2:
            st.subheader("Services")
            inputs['PhoneService'] = st.selectbox("Phone Service", ['Yes', 'No'])
            inputs['MultipleLines'] = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
            inputs['InternetService'] = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            inputs['OnlineSecurity'] = st.selectbox("Online Security", ['No internet service', 'No', 'Yes'])
            
        with col3:
            st.subheader("Account")
            inputs['Contract'] = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            inputs['PaperlessBilling'] = st.selectbox("Paperless Billing", ['Yes', 'No'])
            inputs['PaymentMethod'] = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            inputs['tenure'] = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
            inputs['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
            inputs['TotalCharges'] = st.number_input("Total Charges", min_value=0.0, value=500.0)

        # Remaining categorical inputs (adding defaults for brevity in UI, but necessary for model)
        # Hidden inputs for less critical features to keep UI clean, or set defaults
        inputs['OnlineBackup'] = 'No' 
        inputs['DeviceProtection'] = 'No'
        inputs['TechSupport'] = 'No'
        inputs['StreamingTV'] = 'No' 
        inputs['StreamingMovies'] = 'No'

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Create DataFrame from inputs
        input_df = pd.DataFrame([inputs])
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        st.divider()
        
        # Display Results
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediction == 1:
                st.error(f"**Prediction: CHURN (Yes)**")
                st.write(f"Probability: **{probability:.2%}**")
            else:
                st.success(f"**Prediction: STAY (No)**")
                st.write(f"Probability: **{probability:.2%}**")
        
        with col_res2:
            st.info("💡 **Model Logic:**\nHigh monthly charges and Month-to-month contracts often increase churn risk.")

def show_batch_prediction(model):
    st.header("Batch Prediction")
    st.write("Upload a CSV file containing customer data.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Ensure TotalCharges is numeric
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            
            if st.button("Score Data"):
                # Predict
                predictions = model.predict(df)
                probabilities = model.predict_proba(df)[:, 1]
                
                # Append results
                results_df = df.copy()
                results_df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
                results_df['Churn_Probability'] = probabilities
                
                st.write("### Results Preview")
                st.dataframe(results_df.head())
                
                # Download
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Predictions",
                    csv,
                    "churn_predictions.csv",
                    "text/csv",
                    key='download-csv'
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

def show_model_stats():
    st.header("Model Performance")
    
    # Load Metrics
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", metrics['accuracy'])
        col2.metric("Precision", metrics['precision'])
        col3.metric("Recall", metrics['recall'])
        col4.metric("ROC AUC", metrics['roc_auc'])
    else:
        st.warning("No metrics found.")

    # Show Plot
    if os.path.exists(CM_PLOT_PATH):
        st.subheader("Confusion Matrix")
        st.image(CM_PLOT_PATH)

if __name__ == "__main__":
    main()