# AI Legal Chatbot - Dockerfile
# This file defines the container image setup for the Streamlit web application.

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies (curl is helpful for debugging connectivity to Ollama)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY "ai legal chatbot/requirements.txt" .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY "ai legal chatbot/" ./ai_legal_chatbot/

# Expose port 8501 for Streamlit
EXPOSE 8501

# Configure Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application
CMD ["streamlit", "run", "ai_legal_chatbot/app.py"]
