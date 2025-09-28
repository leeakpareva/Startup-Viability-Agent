"""
NAVADA - Vercel Serverless API Endpoint
Startup Viability Agent - API wrapper for Vercel deployment
"""

import os
import sys
from pathlib import Path

# Add parent directory to Python path to import app.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Set environment variables
os.environ.setdefault('CHAINLIT_HOST', '0.0.0.0')
os.environ.setdefault('CHAINLIT_PORT', '8000')

try:
    # Import Chainlit and the main app
    import chainlit as cl
    from app import main, SESSION_MEMORY, PERSONAS

    # Simple API response for health check
    def handler(request):
        """
        Simple API handler for Vercel deployment.
        Note: Full Chainlit functionality may be limited in serverless environment.
        """
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": {
                "status": "active",
                "app": "NAVADA - Startup Viability Agent",
                "version": "1.0.0",
                "features": [
                    "Interactive Dashboards",
                    "AI Personas & Memory",
                    "Web Scraping & Analysis",
                    "Advanced Analytics",
                    "Professional Reports"
                ],
                "message": "NAVADA is running! For full functionality, use the Chainlit interface.",
                "deployment": "vercel-serverless"
            }
        }

except ImportError as e:
    def handler(request):
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": {
                "error": f"Import error: {str(e)}",
                "message": "Dependencies not properly installed"
            }
        }