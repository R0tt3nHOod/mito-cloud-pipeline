import pandas as pd
import numpy as np

# 1. Settings
N_PER_CLASS = 7000
N_SAMPLES_TRAIN = N_PER_CLASS * 4 # 28000 Total Samples
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print(f"Generating {N_SAMPLES_TRAIN} balanced veteran records based on Haley Criteria...")

# 2. Define Classes & Generate Balanced Targets
# Goal: 7000 samples for each of the 4 classes (0, 1, 2, 3)

target_class_balanced = []
for class_id in [0, 1, 2, 3]:
    # Generate N_PER_CLASS samples of each class directly.
    target_class_balanced.extend(np.repeat(class_id, N_PER_CLASS))

# CRITICAL FIX: Convert the Python list to a NumPy array for correct slicing
target_class_balanced = np.array(target_class_balanced, dtype=np.int32)

# Shuffle the final balanced target array to ensure no block training
np.random.shuffle(target_class_balanced)


# 3. Generate Biomarkers with "Syndrome-Specific" shifts
# Initialize all feature arrays to the correct size (28,000)
nad_nadh = np.empty(N_SAMPLES_TRAIN)
pcr_atp = np.empty(N_SAMPLES_TRAIN)
gsh_gssg = np.empty(N_SAMPLES_TRAIN)

# --- NAD/NADH Ratio ---
# Means (Healthy: 2.0, Syn 1: 1.8, Syn 2: 1.2, Syn 3: 1.5)
nad_nadh[target_class_balanced == 0] = np.random.normal(2.0, 0.4, N_PER_CLASS)
nad_nadh[target_class_balanced == 1] = np.random.normal(1.8, 0.4, N_PER_CLASS)
nad_nadh[target_class_balanced == 2] = np.random.normal(1.2, 0.3, N_PER_CLASS)
nad_nadh[target_class_balanced == 3] = np.random.normal(1.5, 0.4, N_PER_CLASS)

# --- PCr/ATP Ratio (Cellular Energy) ---
# Means (Healthy: 1.8, Syn 1: 1.7, Syn 2: 1.3, Syn 3: 1.1)
pcr_atp[target_class_balanced == 0] = np.random.normal(1.8, 0.3, N_PER_CLASS)
pcr_atp[target_class_balanced == 1] = np.random.normal(1.7, 0.3, N_PER_CLASS)
pcr_atp[target_class_balanced == 2] = np.random.normal(1.3, 0.3, N_PER_CLASS)
pcr_atp[target_class_balanced == 3] = np.random.normal(1.1, 0.2, N_PER_CLASS)

# --- GSH/GSSG Ratio (Oxidative Stress) ---
# Means (Healthy: 30.0, Syn 1: 25.0, Syn 2: 18.0, Syn 3: 20.0)
gsh_gssg[target_class_balanced == 0] = np.random.normal(30.0, 6.0, N_PER_CLASS)
gsh_gssg[target_class_balanced == 1] = np.random.normal(25.0, 5.0, N_PER_CLASS)
gsh_gssg[target_class_balanced == 2] = np.random.normal(18.0, 4.0, N_PER_CLASS)
gsh_gssg[target_class_balanced == 3] = np.random.normal(20.0, 5.0, N_PER_CLASS)

# 4. Assemble & Clean
df = pd.DataFrame({
    'NAD_NADH': nad_nadh,
    'PCr_ATP': pcr_atp,
    'GSH_GSSG': gsh_gssg,
    'Haley_Syndrome': target_class_balanced
})

# Round to 4 decimals
df = df.round(4)

# 5. Save
output_filename = "gwi_haley_balanced_28k.csv"
df.to_csv(output_filename, index=False)

print(f"\n✅ Simulation Complete. Saved {len(df)} records to {output_filename}")
print("\nClass Distribution (Must be 25% for all classes):")
print(df['Haley_Syndrome'].value_counts(normalize=True).sort_index())
print("0=Healthy, 1=Impaired Cognition, 2=Confusion-Ataxia, 3=Arthro-myo-neuropathy")
