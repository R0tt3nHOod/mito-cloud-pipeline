import pandas as pd
import numpy as np

# 1. Settings
N_PER_CLASS = 7000
N_SAMPLES = N_PER_CLASS * 4 
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print(f"Generating {N_SAMPLES} records with REAL-LIFE VARIANCE (High Noise)...")

# 2. Generate Classes
target_class = np.concatenate([np.full(N_PER_CLASS, i) for i in range(4)])
np.random.shuffle(target_class)

# ---------------------------------------------------------
# PART A: "LIFELIKE" BIOMARKERS (High Variance/Overlap)
# ---------------------------------------------------------
nad_nadh = np.empty(N_SAMPLES)
pcr_atp = np.empty(N_SAMPLES)
gsh_gssg = np.empty(N_SAMPLES)

# Healthy (0) - Standard deviations bumped to ~0.4/0.3 to mimic reality
nad_nadh[target_class==0] = np.random.normal(2.0, 0.40, N_PER_CLASS)
pcr_atp[target_class==0]  = np.random.normal(1.8, 0.30, N_PER_CLASS)
gsh_gssg[target_class==0] = np.random.normal(30.0, 6.0, N_PER_CLASS)

# Type 1 (Cognitive) - Overlaps significantly with Healthy
nad_nadh[target_class==1] = np.random.normal(1.8, 0.40, N_PER_CLASS)
pcr_atp[target_class==1]  = np.random.normal(1.7, 0.30, N_PER_CLASS)
gsh_gssg[target_class==1] = np.random.normal(25.0, 5.0, N_PER_CLASS)

# Type 2 (Ataxia)
nad_nadh[target_class==2] = np.random.normal(1.2, 0.30, N_PER_CLASS)
pcr_atp[target_class==2]  = np.random.normal(1.3, 0.30, N_PER_CLASS)
gsh_gssg[target_class==2] = np.random.normal(18.0, 4.0, N_PER_CLASS)

# Type 3 (Pain)
nad_nadh[target_class==3] = np.random.normal(1.5, 0.40, N_PER_CLASS)
pcr_atp[target_class==3]  = np.random.normal(1.1, 0.20, N_PER_CLASS)
gsh_gssg[target_class==3] = np.random.normal(20.0, 5.0, N_PER_CLASS)

# Feature Engineering: The "Metabolic Index" (Still critical)
meta_index = (nad_nadh/2.0) + (pcr_atp/1.8) + (gsh_gssg/30.0)

# ---------------------------------------------------------
# PART B: REALISTIC SYMPTOM SURVEYS (0-10 Scale)
# ---------------------------------------------------------
pain = np.empty(N_SAMPLES)
confusion = np.empty(N_SAMPLES)
dizziness = np.empty(N_SAMPLES)
fatigue = np.empty(N_SAMPLES)

# Healthy: Real people aren't perfect. They have aches (Mean=2, SD=1.5)
pain[target_class==0]      = np.random.normal(2.0, 1.5, N_PER_CLASS)
confusion[target_class==0] = np.random.normal(1.0, 1.0, N_PER_CLASS)
dizziness[target_class==0] = np.random.normal(1.0, 1.0, N_PER_CLASS)
fatigue[target_class==0]   = np.random.normal(2.5, 1.5, N_PER_CLASS)

# Type 1 (Cognitive): "Brain Fog" dominates
pain[target_class==1]      = np.random.normal(4.0, 2.0, N_PER_CLASS)
confusion[target_class==1] = np.random.normal(8.0, 1.5, N_PER_CLASS) # Strong Signal
dizziness[target_class==1] = np.random.normal(3.0, 2.0, N_PER_CLASS)
fatigue[target_class==1]   = np.random.normal(6.0, 2.0, N_PER_CLASS)

# Type 2 (Ataxia): "Vertigo" dominates
pain[target_class==2]      = np.random.normal(5.0, 2.0, N_PER_CLASS)
confusion[target_class==2] = np.random.normal(5.0, 2.0, N_PER_CLASS)
dizziness[target_class==2] = np.random.normal(8.5, 1.5, N_PER_CLASS) # Strong Signal
fatigue[target_class==2]   = np.random.normal(6.0, 2.0, N_PER_CLASS)

# Type 3 (Pain): "Agony" dominates
pain[target_class==3]      = np.random.normal(8.5, 1.5, N_PER_CLASS) # Strong Signal
confusion[target_class==3] = np.random.normal(3.0, 2.0, N_PER_CLASS)
dizziness[target_class==3] = np.random.normal(2.0, 1.5, N_PER_CLASS)
fatigue[target_class==3]   = np.random.normal(9.0, 1.0, N_PER_CLASS)

# Clip to valid survey range (0-10)
for arr in [pain, confusion, dizziness, fatigue]:
    np.clip(arr, 0, 10, out=arr)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
df = pd.DataFrame({
    # Objective (Noisy)
    'NAD_NADH': nad_nadh,
    'PCr_ATP': pcr_atp,
    'GSH_GSSG': gsh_gssg,
    'Metabolic_Index': meta_index,
    # Subjective (Context)
    'joint_pain': pain,
    'confusion': confusion,
    'dizziness': dizziness,
    'fatigue': fatigue,
    # Target
    'Haley_Syndrome': target_class
})

df = df.round(4)
df.to_csv('gwi_lifelike_full.csv', index=False)
print("✅ Generated 'gwi_lifelike_full.csv' with REALISTIC NOISE.")
