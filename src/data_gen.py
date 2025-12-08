import pandas as pd
import numpy as np
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
import os

# --- 1. PARAMETERS ---
N_SAMPLES = 10000  # Target enterprise-grade sample size
CLASSES = ['Healthy', 'Type 1', 'Type 2']
BLOB_CONTAINER_NAME = "data-raw"
STORAGE_ACCOUNT_NAME = "agmitocloud01" # Your Data Lake
BLOB_NAME = "gwi_training_data_v1.csv"

# --- 2. GENERATION LOGIC (Haley Criteria Simulation) ---
def generate_mitochondrial_data(n):
    """Generates synthetic 3-class data based on simulated mitochondrial ratios."""
    
    data = []
    
    # Simulate three key metabolites (simplified)
    # Ratios are adjusted to promote a clearer 3-class separation (Haley Criteria)
    np_ratio = np.random.normal(loc=1.5, scale=0.5, size=n) # NAD+/NADH ratio
    pcr_ratio = np.random.normal(loc=2.0, scale=0.6, size=n) # PCr/ATP ratio
    gsh_ratio = np.random.normal(loc=1.0, scale=0.3, size=n) # GSH/GSSG ratio
    
    for i in range(n):
        label = 'Healthy'
        
        # Type 1 (Low energy status, high oxidative stress)
        if np_ratio[i] < 1.0 and pcr_ratio[i] < 1.5:
            label = 'Type 1'
            
        # Type 2 (Normal energy, but high metabolic rate/dysfunction)
        elif np_ratio[i] > 2.0 and gsh_ratio[i] > 1.2:
            label = 'Type 2'
            
        data.append({
            'NAD_NADH': np_ratio[i],
            'PCr_ATP': pcr_ratio[i],
            'GSH_GSSG': gsh_ratio[i],
            'Target_Class': label
        })

    return pd.DataFrame(data)

# --- 3. STORAGE UPLOAD ---
def upload_data_to_data_lake(df):
    """Uploads the generated CSV to the Azure Data Lake using Managed Identity."""
    print(f"Connecting to Data Lake {STORAGE_ACCOUNT_NAME}...")
    
    credential = DefaultAzureCredential()

    blob_client = BlobClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        container_name=BLOB_CONTAINER_NAME,
        blob_name=BLOB_NAME,
        credential=credential
    )

    csv_data = df.to_csv(index=False).encode('utf-8')
    
    try:
        blob_client.upload_blob(csv_data, overwrite=True)
        print(f"✅ Data uploaded successfully to {BLOB_CONTAINER_NAME}/{BLOB_NAME}")
    except Exception as e:
        print(f"❌ ERROR uploading data: {e}")
        raise
        
if __name__ == "__main__":
    df_train = generate_mitochondrial_data(N_SAMPLES)
    print(f"Generated {len(df_train)} samples.")
    print("\nClass Distribution:")
    print(df_train['Target_Class'].value_counts())
    
    upload_data_to_data_lake(df_train)
