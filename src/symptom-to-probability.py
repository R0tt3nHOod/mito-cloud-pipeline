import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Load the "Real-Life" Data
# This file contains BOTH the noisy biomarkers AND the subjective symptoms
input_file = 'gwi_lifelike_full.csv'
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} records from '{input_file}'")

# 2. Define Your Feature Sets
# These MUST match the columns you generated in the previous script
symptom_cols = ['joint_pain', 'confusion', 'dizziness', 'fatigue']
biomarker_cols = ['NAD_NADH', 'PCr_ATP', 'GSH_GSSG', 'Metabolic_Index']
target = 'Haley_Syndrome'

# 3. Train the "Subjective Prior" Model (The Survey Analyzer)
# We use Logistic Regression because it gives us calibrated probabilities (0-100%)
print("Training Subjective Prior Model (Logistic Regression)...")
X_sym = df[symptom_cols]
y = df[target]

# Note: We fit on the whole dataset here to generate the feature for the next stage.
# In a production pipeline, this model would be saved and versioned separately.
prior_model = LogisticRegression(max_iter=1000)
prior_model.fit(X_sym, y)

# 4. Generate the "Symptom Probability" Feature
# We ask the model: "Based strictly on these symptoms, what is the probability this patient is sick?"
# predict_proba returns an array [Prob_Healthy, Prob_Type1, Prob_Type2, Prob_Type3]
# We take the MAXIMUM probability of any specific syndrome to capture "Confidence of Illness"
all_probs = prior_model.predict_proba(X_sym)
df['Symptom_Prior_Probability'] = np.max(all_probs[:, 1:], axis=1) # Max prob of class 1, 2, or 3

# Alternatively, if you want the probability of the *specific* predicted class:
# df['Symptom_Prior_Probability'] = all_probs.max(axis=1)

print("✅ Generated 'Symptom_Prior_Probability' column.")

# 5. Create the Final "Two-Stage" Dataset
# CRITICAL: We DROP the raw symptom columns now. 
# The final model will only see the Biomarkers + The Symptom Probability.
final_cols = biomarker_cols + ['Symptom_Prior_Probability', target]
df_final = df[final_cols]

output_file = 'gwi_bayesian_training_set.csv'
df_final.to_csv(output_file, index=False)

print(f"\nSUCCESS! Saved final training file: '{output_file}'")
print("-" * 30)
print(f"Columns included: {list(df_final.columns)}")
print("UPLOAD THIS FILE to Azure for your final training run.")
