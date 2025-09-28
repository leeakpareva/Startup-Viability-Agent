#!/usr/bin/env python3
"""
NAVADA - Vercel Deployment Handler
Startup Viability Agent - Chainlit App for Vercel Serverless Functions
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set environment variables for Vercel deployment
os.environ.setdefault('CHAINLIT_HOST', '0.0.0.0')
os.environ.setdefault('CHAINLIT_PORT', '8000')

# Import and configure Chainlit
try:
    import chainlit as cl
    from chainlit.server import app

    # Import the main application
    import app as navada_app

    # Vercel serverless function handler
    def handler(request, response):
        """
        Vercel serverless function handler for Chainlit app.

        Args:
            request: Vercel request object
            response: Vercel response object

        Returns:
            Response from Chainlit app
        """
        return app(request, response)

    # For development/testing
    if __name__ == "__main__":
        print("Starting NAVADA for Vercel deployment...")
        cl.run(
            host="0.0.0.0",
            port=int(os.environ.get('PORT', 8000)),
            headless=False,
            watch=False
        )

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed")
    sys.exit(1)
except Exception as e:
    print(f"Error starting NAVADA: {e}")
    sys.exit(1)

# Export the app for Vercel
app = app