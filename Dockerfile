# Use a standard, stable Azure ML base image
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into the container
COPY src /src
WORKDIR /
