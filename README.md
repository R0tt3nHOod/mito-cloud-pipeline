
# Mito-Cloud: AI-Driven Diagnostic Pipeline for Neuro-Metabolic Disorders

## 🚀 Overview
Mito-Cloud is a serverless, cloud-native diagnostic system designed to automate the analysis of complex 31P-Magnetic Resonance Spectroscopy (31P-MRS) data.

This project addresses the critical barrier between research and clinical application by transforming a manual, 4-hour medical analysis process into an instant, automated diagnostic classification.

## ✨ Key Differentiators
- **Cost Efficiency:** Computes a full diagnosis for **$0.04** per patient[cite: 100].
- **Speed:** Reduces processing time from hours to **under five seconds**[cite: 57].
- **Security:** Ensures **HIPAA compliance** via Azure Data Lake Storage Gen2, Entra ID RBAC, and Private Endpoints[cite: 41, 42].
- **Accuracy:** Achieved **>95% classification accuracy** on synthetic cohort data[cite: 56].

## 💻 Architecture
The system is built on Microsoft Azure and leverages three stages:
1.  **Data Ingestion:** Raw DICOM files are stored in a secure [Azure Data Lake Storage Gen2](https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction).
2.  **Feature Extraction:** [Azure Synapse Analytics](https://learn.microsoft.com/en-us/azure/synapse-analytics/overview-what-is) (Serverless Spark) extracts key metabolic biomarkers (PCr Recovery Time and NAA/tCr Ratio [cite: 48, 50]).
3.  **AI Classification:** [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/) (Random Forest) classifies the patient[cite: 53].

## 🔬 Scientific Context
This pipeline operationalizes seminal research linking mitochondrial dysfunction (specifically PCr recovery kinetics) to **Gulf War Illness (GWI)**[cite: 22]. The architecture is designed to scale to other metabolic disorders, including **TBI** and **early-stage Alzheimer's**[cite: 105, 106].
