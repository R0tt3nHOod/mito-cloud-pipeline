import pandas as pd
import numpy as np

# 1. Settings
N_SAMPLES_TRAIN = 28000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print(f"Generating {N_SAMPLES_TRAIN} balanced veteran records based on Haley Criteria...")

# 2. Define Classes & Prevalence
# --- Create Balanced Training Set (7000 samples for each of the 4 classes) ---
target_class_balanced = []
N_PER_CLASS = 7000 
for class_id in [0, 1, 2, 3]:
    # Generate 7000 samples of each class directly.
    target_class_balanced.extend(np.random.choice([class_id], size=N_PER_CLASS))

# Shuffle the final balanced target list to ensure no block training
np.random.shuffle(target_class_balanced)	np.random.shuffle(target_class_balanced)
target_class_balanced = np.array(target_class_balanced)

# 3. Generate Biomarkers with "Syndrome-Specific" shifts
# Initialize all feature arrays to the correct size (28,000)
nad_nadh = np.zeros(N_SAMPLES_TRAIN)
pcr_atp = np.zeros(N_SAMPLES_TRAIN)
gsh_gssg = np.zeros(N_SAMPLES_TRAIN)

# --- NAD/NADH Ratio ---
# We fill the pre-initialized array slices based on the balanced target classes
nad_nadh[target_class_balanced == 0] = np.random.normal(2.0, 0.4, np.sum(target_class_balanced == 0))
nad_nadh[target_class_balanced == 1] = np.random.normal(1.8, 0.4, np.sum(target_class_balanced == 1))
nad_nadh[target_class_balanced == 2] = np.random.normal(1.2, 0.3, np.sum(target_class_balanced == 2))
nad_nadh[target_class_balanced == 3] = np.random.normal(1.5, 0.4, np.sum(target_class_balanced == 3))

# --- PCr/ATP Ratio (Cellular Energy) ---
pcr_atp[target_class_balanced == 0] = np.random.normal(1.8, 0.3, np.sum(target_class_balanced == 0))
pcr_atp[target_class_balanced == 1] = np.random.normal(1.7, 0.3, np.sum(target_class_balanced == 1))
pcr_atp[target_class_balanced == 2] = np.random.normal(1.3, 0.3, np.sum(target_class_balanced == 2))
pcr_atp[target_class_balanced == 3] = np.random.normal(1.1, 0.2, np.sum(target_class_balanced == 3))

# --- GSH/GSSG Ratio (Oxidative Stress) ---
gsh_gssg[target_class_balanced == 0] = np.random.normal(30.0, 6.0, np.sum(target_class_balanced == 0))
gsh_gssg[target_class_balanced == 1] = np.random.normal(25.0, 5.0, np.sum(target_class_balanced == 1))
gsh_gssg[target_class_balanced == 2] = np.random.normal(18.0, 4.0, np.sum(target_class_balanced == 2))
gsh_gssg[target_class_balanced == 3] = np.random.normal(20.0, 5.0, np.sum(target_class_balanced == 3))

# 4. Assemble & Clean
df = pd.DataFrame({
    'NAD_NADH': nad_nadh,
    'PCr_ATP': pcr_atp,
    'GSH_GSSG': gsh_gssg,
    'Haley_Syndrome': target_class_balanced
})

# Round to 4 decimals to act as "floats" for Azure
df = df.round(4)

# 5. Save
output_filename = "gwi_haley_balanced_28k.csv" # CORRECTED FILENAME
df.to_csv(output_filename, index=False)

print(f"✅ Simulation Complete. Saved {len(df)} records to {output_filename}")
print("\nClass Distribution:")
print(df['Haley_Syndrome'].value_counts(normalize=True).sort_index())
print("0=Healthy, 1=Impaired Cognition, 2=Confusion-Ataxia, 3=Arthro-myo-neuropathy")
