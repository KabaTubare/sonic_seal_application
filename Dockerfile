# Use the official lightweight Python image
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the application directory
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PORT=8080

# Expose the application port
EXPOSE 8080

# Start the Flask application
CMD gunicorn --bind :$PORT --workers 1 --threads 8 app:app