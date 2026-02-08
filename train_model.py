import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# CHANGED: Import Logistic Regression instead of Random Forest
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from utils import load_data, FEATURES

# --- CONFIGURATION ---
DATA_PATH = 'data/data.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_pipeline.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
CM_PLOT_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')

def train_and_evaluate():
    # 1. Load Data
    print("Loading data...")
    df = load_data(DATA_PATH)
    
    if df is None:
        return

    # Separate X and y
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build the Pipeline
    print("Building pipeline...")
    
    # Preprocessing for Numerical Data: Impute -> Scale
    # NOTE: Scaling is CRITICAL for Logistic Regression
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) 
    ])

    # Preprocessing for Categorical Data: Impute -> OneHotEncode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, FEATURES['numeric']),
            ('cat', categorical_transformer, FEATURES['categorical'])
        ]
    )

    # CHANGED: Use LogisticRegression
    # solver='liblinear' is good for small datasets and supports both L1 and L2 regularization
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
    ])

    # 4. Hyperparameter Tuning with Cross-Validation
    print("Tuning hyperparameters with Cross-Validation...")
    
    # CHANGED: Hyperparameters for Logistic Regression
    param_dist = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength (smaller = stronger regularization)
        'classifier__penalty': ['l1', 'l2']        # Lasso (l1) vs Ridge (l2)
    }

    # CV=5 means 5-Fold Cross Validation
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=5,            # <--- This performs the Cross-Validation
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"Best params: {search.best_params_}")
    print(f"Best Cross-Validation Score (ROC-AUC): {search.best_score_:.4f}")

    # 5. Evaluation on Hold-out Test Set
    print("Evaluating model on test set...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 4)
    }
    
    print(f"Test Set Metrics: {metrics}")

    # Create output directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save Metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)

    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Logistic Regression)')
    plt.savefig(CM_PLOT_PATH)
    plt.close()

    # 6. Save Model
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    train_and_evaluate()