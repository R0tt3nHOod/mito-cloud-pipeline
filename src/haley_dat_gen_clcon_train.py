import pandas as pd
import numpy as np

# 1. Settings
N_PER_CLASS = 7000  # Balanced
N_SAMPLES_TRAIN = N_PER_CLASS * 4 
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print(f"Generating {N_SAMPLES_TRAIN} records with ENGINEERED FEATURES...")

# 2. Generate Basic Data (Same 'Noisy' Logic as before)
target_class = np.concatenate([np.full(N_PER_CLASS, i) for i in range(4)])
np.random.shuffle(target_class)

# Initialize arrays
nad_nadh = np.empty(N_SAMPLES_TRAIN)
pcr_atp = np.empty(N_SAMPLES_TRAIN)
gsh_gssg = np.empty(N_SAMPLES_TRAIN)

# --- Apply Noisy Distributions (Standard Deviations from your difficult dataset) ---
# Healthy (0)
nad_nadh[target_class==0] = np.random.normal(2.0, 0.4, N_PER_CLASS)
pcr_atp[target_class==0]  = np.random.normal(1.8, 0.3, N_PER_CLASS)
gsh_gssg[target_class==0] = np.random.normal(30.0, 6.0, N_PER_CLASS)

# Type 1 (Cognitive - The "Hard" Class)
nad_nadh[target_class==1] = np.random.normal(1.8, 0.4, N_PER_CLASS)
pcr_atp[target_class==1]  = np.random.normal(1.7, 0.3, N_PER_CLASS)
gsh_gssg[target_class==1] = np.random.normal(25.0, 5.0, N_PER_CLASS)

# Type 2 (Ataxia)
nad_nadh[target_class==2] = np.random.normal(1.2, 0.3, N_PER_CLASS)
pcr_atp[target_class==2]  = np.random.normal(1.3, 0.3, N_PER_CLASS)
gsh_gssg[target_class==2] = np.random.normal(18.0, 4.0, N_PER_CLASS)

# Type 3 (Pain)
nad_nadh[target_class==3] = np.random.normal(1.5, 0.4, N_PER_CLASS)
pcr_atp[target_class==3]  = np.random.normal(1.1, 0.2, N_PER_CLASS)
gsh_gssg[target_class==3] = np.random.normal(20.0, 5.0, N_PER_CLASS)

# 3. FEATURE ENGINEERING (The "Secret Sauce")
# These columns mathematically amplify the small signal drops in Type 1

# Feature A: Total Metabolic Health (Sum of normalized scores)
# This aggregates the small "0.2" drops across all 3 markers into a larger single drop.
meta_health_index = (nad_nadh/2.0) + (pcr_atp/1.8) + (gsh_gssg/30.0)

# Feature B: Oxidative Energy Ratio
# Multiplies the effects. If both are low, this value drops drastically.
ox_energy_interaction = nad_nadh * gsh_gssg

# 4. Assemble
df = pd.DataFrame({
    'NAD_NADH': nad_nadh,
    'PCr_ATP': pcr_atp,
    'GSH_GSSG': gsh_gssg,
    'Metabolic_Index': meta_health_index,       # NEW COLUMN
    'Ox_Energy_Interaction': ox_energy_interaction, # NEW COLUMN
    'Haley_Syndrome': target_class
})

df = df.round(4)
df.to_csv("gwi_haley_engineered_28k.csv", index=False)
print("✅ Saved 'gwi_haley_engineered_28k.csv' with derived clinical features.")
