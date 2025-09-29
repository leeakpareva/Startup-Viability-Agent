#!/bin/bash

# NAVADA FastAPI Deployment Script
echo "🚀 Deploying NAVADA with FastAPI..."

# Build and run with Docker Compose
echo "📦 Building FastAPI container..."
docker-compose up --build -d

# Wait for health check
echo "⏳ Waiting for service to be healthy..."
sleep 10

# Check health
echo "🔍 Checking service health..."
curl -f http://localhost:8000/health

echo "✅ Deployment complete!"
echo "🌐 API available at: http://localhost:8000"
echo "📚 Documentation at: http://localhost:8000/docs"