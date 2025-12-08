import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from azure.identity import DefaultAzureCredential
import mlflow

# --- 1. PARAMETERS & INPUTS ---
parser = argparse.ArgumentParser()
parser.add_argument("--training_data_path", type=str, dest="training_data_path", help="Path to the training data CSV")
args = parser.parse_args()
data_path = args.training_data_path
RANDOM_STATE = 42
MODEL_NAME = "gwi-classifier-rf"
BLOB_CONTAINER_NAME = "data-raw"
BLOB_NAME = "gwi_training_data_v1.csv"
STORAGE_ACCOUNT_NAME = "agmitocloud01" # Your Data Lake

# --- 2. DATA LOADING FUNCTION ---
def load_data_from_lake():
    """Loads data from the Data Lake using the job's Managed Identity."""
    print("Connecting to Data Lake using Managed Identity...")
    
    full_path = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{BLOB_NAME}"
    
    try:
        # Uses the Managed Identity granted permission (via the CLI fix)
        # We explicitly pass the token for security and reliability with the Managed Identity
        credential = DefaultAzureCredential()
        token = credential.get_token("https://storage.azure.com/").token
        
        df = pd.read_csv(full_path, storage_options={"token": token})
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ ERROR loading data. Check IAM role assignment. Details: {e}")
        raise

# --- 3. MAIN TRAINING FUNCTION ---
def train_model(df):
    
    mlflow.set_autolog(False)
    mlflow.start_run()

    # Data Preparation: Separate features (X) and target (y)
    X = df[['NAD_NADH', 'PCr_ATP', 'GSH_GSSG']]
    y = df['Target_Class']

    # Convert target classes to numerical labels (0: Healthy, 1: Type 1, 2: Type 2)
    categories = ['Healthy', 'Type 1', 'Type 2']
    y_encoded = pd.Categorical(y, categories=categories).codes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded)

    # Train Random Forest Classifier
    print("Starting Enterprise Random Forest Model Training...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log Metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    
    print(f"\n--- Model Performance ---")
    print(f"Classification Accuracy: {accuracy:.4f}")
    
    # Model Registration
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )
    
    mlflow.end_run()
    return model

if __name__ == "__main__":
    # 1. Load the large dataset
    data_df = load_data_from_lake()
    
    # 2. Train and register the model
    trained_model = train_model(data_df)
    print(f"Model {MODEL_NAME} trained and registered successfully!")
