# Use an official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the repository files into the container
COPY . /app

# Install system dependencies if required (e.g., git, build-essential)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone --depth 1 https://github.com/ANSLab-UHN/sEMG-TypingDatabase.git ./sEMG-TypingDatabase && cd sEMG-TypingDatabase

# Install Python dependencies
RUN pip install -e .

# Copy KeyPress Data
COPY ./CleanData/P* ./CleanData

RUN bash prepare_data.sh

# Expose a port if the application runs a server (adjust as needed)
# EXPOSE 8080

# Define the default command to run your application
# CMD ["python", "your_main_script.py"]