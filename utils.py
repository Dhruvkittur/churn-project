import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Loads the churn dataset and performs initial type cleanup.
    """
    try:
        df = pd.read_csv(filepath)
        
        # 'TotalCharges' is often object type due to empty strings. Force to numeric.
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill NaN values created by coercion (usually new customers with 0 tenure)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Dictionary to separate features for easy access
FEATURES = {
    'numeric': ['tenure', 'MonthlyCharges', 'TotalCharges'],
    'categorical': [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
}