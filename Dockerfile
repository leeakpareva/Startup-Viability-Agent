# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY chainlit.md .

# Create necessary directories first
RUN mkdir -p chroma_db .chainlit

# Copy .chainlit directory (optional)
COPY .chainlit/ .chainlit/

# Expose Chainlit default port
EXPOSE 8000

# Set environment variables for Chainlit
ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=8000

# Run the application
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--no-cache"]