# Use a standard, stable Azure ML base image
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:py38-cuda11.7-gpu-inteloneapi-aif

# Install Python packages using pip directly
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    mlflow \
    azure-identity \
    azure-storage-blob \
    joblib

# Copy your code into the container
COPY src /src
# Set the working directory
WORKDIR /
