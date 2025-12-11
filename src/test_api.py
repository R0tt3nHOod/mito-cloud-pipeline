import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # Bypass certificate errors if using a self-signed cert (common in dev)
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # Call the bypass

# --- CONFIGURATION ---
# I pulled this URL directly from your screenshot
url = 'https://proto-mtiocloud-jdakw.eastus2.inference.ml.azure.com/score'

# GO TO THE 'CONSUME' TAB IN AZURE TO GET THIS KEY
api_key = 'PASTE_YOUR_PRIMARY_KEY_HERE' 

# --- DATA GENERATION ---
# We will create a dictionary mimicking the pandas DataFrame structure the model expects
data = {
    "Inputs": {
        "data": [
            # PATIENT 1: HEALTHY VETERAN (Normal Biomarkers, Low Prior)
            {
                "NAD_NADH": 5.2,          # Normal (>4.0)
                "PCr_ATP": 4.8,           # Normal (>3.5)
                "GSH_GSSG": 45.0,         # Normal (>40)
                "Metabolic_Index": 12.5,  # High (Healthy)
                "Symptom_Prior_Probability": 0.15 # Low Risk Survey
            },
            # PATIENT 2: SICK VETERAN (Type 2 Pattern - Low NAD)
            {
                "NAD_NADH": 1.2,          # Critical Low
                "PCr_ATP": 3.9,           # Normal-ish
                "GSH_GSSG": 22.0,         # Low
                "Metabolic_Index": 4.5,   # Low (Sick)
                "Symptom_Prior_Probability": 0.88 # High Risk Survey
            },
            # PATIENT 3: SICK VETERAN (Type 3 Pattern - Low PCr)
            {
                "NAD_NADH": 4.1,          # Normal
                "PCr_ATP": 1.5,           # Critical Low
                "GSH_GSSG": 28.0,         # Low
                "Metabolic_Index": 5.1,   # Low (Sick)
                "Symptom_Prior_Probability": 0.92 # High Risk Survey
            }
        ]
    }
}

body = str.encode(json.dumps(data))

headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print("-" * 30)
    print("AZURE DIAGNOSIS RESULTS:")
    print("-" * 30)
    # The result usually comes back as a JSON list of predictions
    print(json.loads(result)) 
    print("-" * 30)
    print("Expected: [0, 2, 3] (or similar classes)")

except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
