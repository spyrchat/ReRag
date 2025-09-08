# Use a slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system packages if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full source code
COPY . .
