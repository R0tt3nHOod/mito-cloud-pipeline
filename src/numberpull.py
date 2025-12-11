import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Load the data file you generated yesterday
# Make sure this file is in the same folder as this script
df = pd.read_csv('gwi_lifelike_full.csv')

# 2. Define the inputs (Symptoms) and target (Diagnosis)
X = df[['joint_pain', 'confusion', 'dizziness', 'fatigue']]
y = df['Haley_Syndrome']

# 3. Train the simple model to get the weights
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 4. PRINT THE NUMBERS FOR YOUR UI
print("-" * 30)
print("COPY THESE INTO YOUR app.py FILE:")
print(f"INTERCEPT = {model.intercept_[0]:.5f}")
print(f"COEF_PAIN = {model.coef_[0][0]:.5f}")
print(f"COEF_CONFUSION = {model.coef_[0][1]:.5f}")
print(f"COEF_DIZZINESS = {model.coef_[0][2]:.5f}")
print(f"COEF_FATIGUE = {model.coef_[0][3]:.5f}")
print("-" * 30)
