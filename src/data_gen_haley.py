import pandas as pd
import numpy as np

# 1. Settings
N_SAMPLES = 10000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print(f"Generating {N_SAMPLES} realistic veteran records based on Haley Criteria...")

# 2. Define Classes & Prevalence
# 0 = Healthy Control (~70% of deployed population)
# 1 = Haley Syndrome 1: Impaired Cognition (~10%)
# 2 = Haley Syndrome 2: Confusion-Ataxia (~12% - often cited as most distinct/severe)
# 3 = Haley Syndrome 3: Arthro-myo-neuropathy (~8%)

# Create a list of class labels weighted by probability
class_probs = [0.70, 0.10, 0.12, 0.08]
target_class = np.random.choice([0, 1, 2, 3], size=N_SAMPLES, p=class_probs)

# 3. Generate Biomarkers with "Syndrome-Specific" shifts
# We initialize arrays with zero, then fill them based on the class.

# -- NAD/NADH Ratio --
# Healthy: Normal (2.0)
# Syn 1 (Cognitive): Mild reduction (1.8)
# Syn 2 (Ataxia): Severe reduction (1.2) - linked to brainstem/sarin neurotoxicity
# Syn 3 (Pain): Moderate reduction (1.5)
nad_nadh = np.zeros(N_SAMPLES)
nad_nadh[target_class == 0] = np.random.normal(2.0, 0.4, np.sum(target_class == 0))
nad_nadh[target_class == 1] = np.random.normal(1.8, 0.4, np.sum(target_class == 1))
nad_nadh[target_class == 2] = np.random.normal(1.2, 0.3, np.sum(target_class == 2))
nad_nadh[target_class == 3] = np.random.normal(1.5, 0.4, np.sum(target_class == 3))

# -- PCr/ATP Ratio (Cellular Energy) --
# Healthy: Normal (1.8)
# Syn 1: Normal/Low (1.7) - Cognitive issues may not show deep muscle energy deficits
# Syn 2: Low (1.3) - High energy demand/failure
# Syn 3: Very Low (1.1) - Muscle/Joint specific, so ATP depletion in muscle is high
pcr_atp = np.zeros(N_SAMPLES)
pcr_atp[target_class == 0] = np.random.normal(1.8, 0.3, np.sum(target_class == 0))
pcr_atp[target_class == 1] = np.random.normal(1.7, 0.3, np.sum(target_class == 1))
pcr_atp[target_class == 2] = np.random.normal(1.3, 0.3, np.sum(target_class == 2))
pcr_atp[target_class == 3] = np.random.normal(1.1, 0.2, np.sum(target_class == 3))

# -- GSH/GSSG Ratio (Oxidative Stress) --
# Healthy: High (30.0)
# Syn 1: Moderate Stress (25.0)
# Syn 2: High Stress (18.0)
# Syn 3: High Stress (20.0)
gsh_gssg = np.zeros(N_SAMPLES)
gsh_gssg[target_class == 0] = np.random.normal(30.0, 6.0, np.sum(target_class == 0))
gsh_gssg[target_class == 1] = np.random.normal(25.0, 5.0, np.sum(target_class == 1))
gsh_gssg[target_class == 2] = np.random.normal(18.0, 4.0, np.sum(target_class == 2))
gsh_gssg[target_class == 3] = np.random.normal(20.0, 5.0, np.sum(target_class == 3))

# 4. Assemble & Clean
df = pd.DataFrame({
    'NAD_NADH': nad_nadh,
    'PCr_ATP': pcr_atp,
    'GSH_GSSG': gsh_gssg,
    'Haley_Syndrome': target_class 
})

# Round to 4 decimals to act as "floats" for Azure
df = df.round(4)

# 5. Save
output_filename = "gwi_haley_10k.csv"
df.to_csv(output_filename, index=False)

print(f"✅ Simulation Complete.")
print(f"Saved {N_SAMPLES} records to {output_filename}")
print("\nClass Distribution:")
print(df['Haley_Syndrome'].value_counts(normalize=True).sort_index())
print("\n0=Healthy, 1=Impaired Cognition, 2=Confusion-Ataxia, 3=Arthro-myo-neuropathy")
