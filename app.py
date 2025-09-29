# app.py
# NAVADA (Startup Viability Agent) - A Chainlit-powered AI agent for analyzing startup risk and failure patterns
# Features: Investor Mode, Founder Mode, UK Economist Mode with macroeconomic analysis

# =============================
# IMPORTS
# =============================
import io  # For in-memory file operations (byte streams)
import os  # Operating system interface for environment variables and file operations
import time  # Time-related functions for delays and timing operations
import re  # Regular expressions for pattern matching and text processing
import requests  # HTTP library for making API calls
from datetime import datetime  # Date/time handling for timestamps
from typing import Dict, List, Optional, Any  # Type hints for better code documentation
import uuid  # UUID generation for unique identifiers
import math  # Mathematical operations (currently unused but available)
import json  # JSON parsing (currently unused but available)
import asyncio  # Async/await support for concurrent operations
import logging  # Logging system for error tracking and debugging
import traceback  # Detailed error tracebacks for debugging
import functools  # Function decorators and utilities
import chainlit as cl  # Chainlit framework for building conversational AI interfaces
import pandas as pd  # Data manipulation and analysis with DataFrames
import numpy as np  # Numerical operations for calculations
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt  # Core plotting library for creating visualizations
import seaborn as sns  # Statistical data visualization built on matplotlib
import plotly.express as px  # Interactive plotting library for dynamic visualizations
import plotly.graph_objects as go  # Low-level plotly interface for custom charts
import plotly.io as pio  # Plotly I/O utilities for saving/converting charts
from plotly.subplots import make_subplots  # Create subplot layouts for dashboards
import requests  # HTTP library for making web requests and scraping
from bs4 import BeautifulSoup  # HTML/XML parser for web scraping
from urllib.parse import urlparse  # URL validation and parsing utilities
import re  # Regular expressions for text processing and validation
import scipy.stats as stats  # Statistical functions for analysis
import random  # Random number generation for Monte Carlo simulations

from typing import Dict, Any, List, Optional  # Type hints for better code documentation
from dataclasses import dataclass, field  # For structured data classes
from openai import OpenAI  # OpenAI API client for GPT model interactions
from dotenv import load_dotenv  # Load environment variables from .env file
import uuid  # For generating unique thread/session IDs
from IPython.display import display  # IPython display utilities (not actively used)
# Optional ML imports - handle gracefully if not available (for Vercel size limits)
try:
    from sklearn.model_selection import train_test_split  # Split data for ML training
    from sklearn.ensemble import RandomForestClassifier  # Random Forest model for predictions
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available - ML features disabled")

# LangChain & LangSmith imports for hosting
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Optional vector store imports - handle gracefully if not available
try:
    from langchain_chroma import Chroma
    from langchain.chains import RetrievalQA
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("WARNING: Chroma vector store not available - RAG features disabled")
from langsmith import traceable, Client as LangSmithClient
from langsmith.wrappers import wrap_openai
import langsmith as ls

# =============================
# ERROR HANDLING & LOGGING SETUP
# =============================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('navada_app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def safe_api_call(func):
    """Decorator for safe API calls with comprehensive error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error in {func.__name__}: {e}")
            return {"error": f"Network error: {str(e)}", "success": False}
        except ValueError as e:
            logger.error(f"Value error in {func.__name__}: {e}")
            return {"error": f"Invalid data: {str(e)}", "success": False}
        except KeyError as e:
            logger.error(f"Missing key in {func.__name__}: {e}")
            return {"error": f"Missing required field: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Unexpected error: {str(e)}", "success": False}
    return wrapper

def safe_async_api_call(func):
    """Decorator for safe async API calls with comprehensive error handling."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error in {func.__name__}: {e}")
            return {"error": f"Network error: {str(e)}", "success": False}
        except ValueError as e:
            logger.error(f"Value error in {func.__name__}: {e}")
            return {"error": f"Invalid data: {str(e)}", "success": False}
        except KeyError as e:
            logger.error(f"Missing key in {func.__name__}: {e}")
            return {"error": f"Missing required field: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Unexpected error: {str(e)}", "success": False}
    return wrapper

def validate_environment():
    """Validate environment setup and dependencies."""
    logger.info("Starting environment validation...")

    issues = []

    # Check critical environment variables
    required_vars = ["OPENAI_API_KEY"]
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == "your_openai_api_key_here":
            issues.append(f"Missing or placeholder value for {var}")

    # Check optional but recommended environment variables
    optional_vars = {
        "LANGSMITH_API_KEY": "LangSmith tracing",
        "SEARCH_API_KEY": "web search functionality",
        "TTS_PROMPT_ID": "text-to-speech features",
        "LANGCHAIN_DATABASE_ID": "vector database features"
    }

    for var, feature in optional_vars.items():
        value = os.getenv(var)
        if not value or value.startswith("your_"):
            logger.warning(f"Optional {var} not configured - {feature} will be disabled")

    # Check critical imports
    try:
        import matplotlib.pyplot as plt
        logger.info("✅ Matplotlib available")
    except ImportError:
        issues.append("Matplotlib not available - chart generation will fail")

    try:
        from langchain_chroma import Chroma
        logger.info("✅ LangChain Chroma available")
    except ImportError:
        issues.append("LangChain Chroma not available - RAG features disabled")

    try:
        import openai
        logger.info("✅ OpenAI library available")
    except ImportError:
        issues.append("OpenAI library not available - core functionality will fail")

    if issues:
        logger.error("Environment validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False, issues
    else:
        logger.info("✅ Environment validation passed")
        return True, []

def create_startup_health_check():
    """Perform comprehensive health checks during startup."""
    logger.info("🔍 Performing startup health checks...")

    # Get environment variables for health checks
    api_key_check = os.getenv("OPENAI_API_KEY")
    langsmith_api_key_check = os.getenv("LANGSMITH_API_KEY")
    search_api_key_check = os.getenv("SEARCH_API_KEY")

    health_status = {
        "overall": True,
        "checks": {},
        "warnings": [],
        "errors": []
    }

    # Test matplotlib chart generation
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.plot([1, 2], [1, 2])
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        health_status["checks"]["matplotlib"] = "✅ Working"
        logger.info("✅ Matplotlib chart generation test passed")
    except Exception as e:
        health_status["checks"]["matplotlib"] = f"❌ Failed: {e}"
        health_status["errors"].append(f"Matplotlib test failed: {e}")
        health_status["overall"] = False
        logger.error(f"❌ Matplotlib test failed: {e}")

    # Test OpenAI client initialization
    try:
        if api_key_check and api_key_check != "your_openai_api_key_here":
            # Don't make actual API call, just test client creation
            from openai import OpenAI
            test_client = OpenAI(api_key=api_key_check)
            health_status["checks"]["openai"] = "✅ Client initialized"
            logger.info("✅ OpenAI client initialization test passed")
        else:
            health_status["checks"]["openai"] = "⚠️ No valid API key"
            health_status["warnings"].append("OpenAI API key not configured")
            logger.warning("⚠️ OpenAI API key not configured")
    except Exception as e:
        health_status["checks"]["openai"] = f"❌ Failed: {e}"
        health_status["errors"].append(f"OpenAI client test failed: {e}")
        logger.error(f"❌ OpenAI client test failed: {e}")

    # Test vector store functionality
    try:
        if CHROMA_AVAILABLE:
            # Test basic Chroma functionality without creating actual store
            from langchain_chroma import Chroma
            health_status["checks"]["vector_store"] = "✅ Available"
            logger.info("✅ Vector store (Chroma) available")
        else:
            health_status["checks"]["vector_store"] = "❌ Not available"
            health_status["warnings"].append("Vector store not available - RAG features disabled")
            logger.warning("⚠️ Vector store not available")
    except Exception as e:
        health_status["checks"]["vector_store"] = f"❌ Failed: {e}"
        health_status["errors"].append(f"Vector store test failed: {e}")
        logger.error(f"❌ Vector store test failed: {e}")

    # Test search API configuration
    if search_api_key_check and search_api_key_check != "your_brave_search_api_key_here":
        health_status["checks"]["search_api"] = "✅ Configured"
        logger.info("✅ Search API key configured")
    else:
        health_status["checks"]["search_api"] = "⚠️ Not configured"
        health_status["warnings"].append("Search API not configured - web search disabled")
        logger.warning("⚠️ Search API key not configured")

    # Test LangSmith configuration - need to check after client initialization
    langsmith_check = langsmith_api_key_check and langsmith_api_key_check != "your_langsmith_api_key_here"
    if langsmith_check:
        health_status["checks"]["langsmith"] = "✅ Configured"
        logger.info("✅ LangSmith API key configured")
    else:
        health_status["checks"]["langsmith"] = "⚠️ Not configured"
        health_status["warnings"].append("LangSmith tracing disabled")
        logger.warning("⚠️ LangSmith API key not configured")

    # Log summary
    if health_status["overall"]:
        logger.info("✅ All critical health checks passed")
    else:
        logger.error("❌ Some critical health checks failed")

    if health_status["warnings"]:
        logger.info(f"⚠️ {len(health_status['warnings'])} warnings found")

    if health_status["errors"]:
        logger.error(f"❌ {len(health_status['errors'])} errors found")

    return health_status

def retry_on_failure(max_retries=3, delay=1):
    """Decorator to retry functions on failure with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator

def create_error_recovery_system():
    """Create a comprehensive error recovery system."""
    recovery_handlers = {
        "api_timeout": lambda: "Service temporarily unavailable. Please try again in a moment.",
        "rate_limit": lambda: "Rate limit exceeded. Please wait a moment before trying again.",
        "authentication": lambda: "Authentication failed. Please check your API keys configuration.",
        "network_error": lambda: "Network connection error. Please check your internet connection.",
        "service_unavailable": lambda: "External service unavailable. Using fallback mode.",
    }

    return recovery_handlers

# Initialize recovery system
recovery_system = create_error_recovery_system()

# =============================
# AUTHENTICATION INTEGRATION
# =============================

# Import authentication system
try:
    from auth_manager import auth_manager
    AUTH_AVAILABLE = True
    logger.info("✅ Authentication system loaded")
except ImportError as e:
    AUTH_AVAILABLE = False
    logger.warning(f"⚠️ Authentication system not available: {e}")

def check_user_authentication() -> Dict[str, Any]:
    """Check if user is authenticated in current session."""
    if not AUTH_AVAILABLE:
        return {"authenticated": False, "reason": "auth_system_unavailable"}

    # Check for authentication token in session
    auth_token = cl.user_session.get("auth_token")
    session_token = cl.user_session.get("session_token")

    if not auth_token or not session_token:
        return {"authenticated": False, "reason": "no_token"}

    # Validate session with auth manager
    session_result = auth_manager.validate_session(session_token)

    if not session_result["valid"]:
        # Clear invalid session data
        cl.user_session.set("auth_token", None)
        cl.user_session.set("session_token", None)
        cl.user_session.set("user_id", None)
        cl.user_session.set("username", None)
        return {"authenticated": False, "reason": "invalid_session"}

    return {
        "authenticated": True,
        "user_id": session_result["user_id"],
        "username": session_result["username"],
        "email": session_result["email"],
        "subscription_tier": session_result["subscription_tier"]
    }

async def show_login_form():
    """Display login/register form to user."""
    await cl.Message(
        content="🔐 **Welcome to NAVADA!** Please log in or register to continue.\n\n"
               "**Login Instructions:**\n"
               "• Type: `login username password`\n"
               "• Example: `login john mypassword123`\n\n"
               "**Register Instructions:**\n"
               "• Type: `register username password email`\n"
               "• Example: `register john mypassword123 john@example.com`\n"
               "• Email is optional: `register john mypassword123`\n\n"
               "**Demo Account (for testing):**\n"
               "• Username: `demo`\n"
               "• Password: `demo123`\n"
               "• Type: `login demo demo123`"
    ).send()

async def handle_login_command(user_input: str) -> bool:
    """Handle login command and authenticate user."""
    parts = user_input.strip().split()

    if len(parts) < 3:
        await cl.Message(
            content="❌ **Invalid login format**\n\n"
                   "Please use: `login username password`\n"
                   "Example: `login john mypassword123`"
        ).send()
        return False

    username = parts[1]
    password = parts[2]

    # Show authentication progress
    auth_msg = cl.Message(content="🔄 Authenticating...")
    await auth_msg.send()

    # Attempt authentication
    auth_result = auth_manager.authenticate_user(username, password)

    if auth_result["success"]:
        # Store authentication data in session
        cl.user_session.set("auth_token", auth_result["jwt_token"])
        cl.user_session.set("session_token", auth_result["session_token"])
        cl.user_session.set("user_id", auth_result["user_id"])
        cl.user_session.set("username", auth_result["username"])
        cl.user_session.set("user_email", auth_result.get("email"))
        cl.user_session.set("subscription_tier", auth_result.get("subscription_tier", "free"))

        # Update message with success
        auth_msg.content = f"✅ **Welcome back, {auth_result['username']}!**\n\n" \
                          f"🎯 **Account Type:** {auth_result.get('subscription_tier', 'free').title()}\n" \
                          f"📧 **Email:** {auth_result.get('email', 'Not provided')}\n\n" \
                          f"You can now use all NAVADA features! Type **'help'** to get started."
        await auth_msg.update()

        # Log the login
        if AUTH_AVAILABLE:
            session_id = cl.user_session.get("session_id", get_session_id())
            auth_manager.log_user_action(
                auth_result["user_id"],
                "chainlit_login",
                "authentication",
                session_id=session_id
            )

        return True
    else:
        # Update message with error
        auth_msg.content = f"❌ **Login failed:** {auth_result['error']}\n\n" \
                          f"Please check your username and password and try again.\n" \
                          f"Format: `login username password`"
        await auth_msg.update()
        return False

async def handle_register_command(user_input: str) -> bool:
    """Handle register command and create new user."""
    parts = user_input.strip().split()

    if len(parts) < 3:
        await cl.Message(
            content="❌ **Invalid registration format**\n\n"
                   "Please use: `register username password [email]`\n"
                   "Examples:\n"
                   "• `register john mypassword123 john@example.com`\n"
                   "• `register john mypassword123` (without email)"
        ).send()
        return False

    username = parts[1]
    password = parts[2]
    email = parts[3] if len(parts) > 3 else None

    # Basic validation
    if len(username) < 3:
        await cl.Message(content="❌ Username must be at least 3 characters long").send()
        return False

    if len(password) < 6:
        await cl.Message(content="❌ Password must be at least 6 characters long").send()
        return False

    # Show registration progress
    reg_msg = cl.Message(content="🔄 Creating account...")
    await reg_msg.send()

    # Attempt registration
    reg_result = auth_manager.register_user(username, password, email)

    if reg_result["success"]:
        # Auto-login after successful registration
        auth_result = auth_manager.authenticate_user(username, password)

        if auth_result["success"]:
            # Store authentication data in session
            cl.user_session.set("auth_token", auth_result["jwt_token"])
            cl.user_session.set("session_token", auth_result["session_token"])
            cl.user_session.set("user_id", auth_result["user_id"])
            cl.user_session.set("username", auth_result["username"])
            cl.user_session.set("user_email", auth_result.get("email"))
            cl.user_session.set("subscription_tier", auth_result.get("subscription_tier", "free"))

            # Update message with success
            reg_msg.content = f"🎉 **Account created successfully!**\n\n" \
                             f"👤 **Username:** {auth_result['username']}\n" \
                             f"📧 **Email:** {auth_result.get('email', 'Not provided')}\n" \
                             f"🎯 **Account Type:** {auth_result.get('subscription_tier', 'free').title()}\n\n" \
                             f"You're now logged in and ready to use NAVADA! Type **'help'** to get started."
            await reg_msg.update()

            return True

    # Update message with error
    reg_msg.content = f"❌ **Registration failed:** {reg_result['error']}\n\n" \
                     f"Please try a different username or check your details.\n" \
                     f"Format: `register username password [email]`"
    await reg_msg.update()
    return False

# Create demo account on startup if it doesn't exist
if AUTH_AVAILABLE:
    try:
        # Try to create demo account (will fail silently if it already exists)
        demo_result = auth_manager.register_user("demo", "demo123", "demo@navada.ai")
        if demo_result["success"]:
            logger.info("✅ Demo account created: demo/demo123")
    except Exception:
        pass  # Demo account likely already exists

# =============================
# INITIAL SETUP & CONFIGURATION
# =============================
# Load environment variables (OPENAI_API_KEY, LANGSMITH_API_KEY) from .env file
# This keeps sensitive API keys out of the source code
load_dotenv(override=True)  # Force override of existing environment variables

# Validate environment setup
env_valid, env_issues = validate_environment()
if not env_valid:
    logger.warning("Environment validation found issues - some features may not work correctly")

# Get API keys from environment
api_key = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
search_api_key = os.getenv("SEARCH_API_KEY")
tts_prompt_id = os.getenv("TTS_PROMPT_ID")
langchain_database_id = os.getenv("LANGCHAIN_DATABASE_ID")

# Perform startup health checks after environment variables are loaded
health_status = create_startup_health_check()

# Configure LangSmith project name for tracing
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "navada-startup-agent")

# Initialize OpenAI client with optional LangSmith wrapping for tracing
if api_key:
    base_client = OpenAI(api_key=api_key)
    # Wrap with LangSmith if API key is available
    if langsmith_api_key:
        try:
            client = wrap_openai(base_client)
            langsmith_client = LangSmithClient(api_key=langsmith_api_key)
            print("SUCCESS: LangSmith tracing enabled")
        except Exception as e:
            print(f"WARNING: LangSmith initialization failed: {e}")
            print("INFO: Continuing without LangSmith tracing")
            client = base_client
            langsmith_client = None
    else:
        client = base_client
        langsmith_client = None
        print("INFO: LangSmith tracing disabled (no API key)")
else:
    client = OpenAI()  # Will use default OPENAI_API_KEY from environment
    langsmith_client = None

# =============================
# LANGSMITH THREAD MANAGEMENT
# =============================

def get_thread_history(thread_id: str, project_name: str):
    """Get conversation history for a thread using LangSmith."""
    if not langsmith_client:
        return []

    try:
        # Filter runs by the specific thread and project
        filter_string = f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "{thread_id}"))'
        # Only grab the LLM runs
        runs = [r for r in langsmith_client.list_runs(project_name=project_name, filter=filter_string, run_type="llm")]

        if not runs:
            return []

        # Sort by start time to get the most recent interaction
        runs = sorted(runs, key=lambda run: run.start_time, reverse=True)

        # Build conversation history from runs
        messages = []
        for run in reversed(runs):  # Reverse to get chronological order
            if run.inputs and 'messages' in run.inputs:
                # Add the user message from inputs
                user_messages = [msg for msg in run.inputs['messages'] if msg['role'] == 'user']
                if user_messages:
                    messages.extend(user_messages)

            if run.outputs and 'choices' in run.outputs:
                # Add the assistant response
                assistant_message = {
                    "role": "assistant",
                    "content": run.outputs['choices'][0]['message']['content']
                }
                messages.append(assistant_message)

        return messages

    except Exception as e:
        print(f"⚠️ Error getting thread history: {e}")
        return []

# =============================
# LANGSMITH SETUP FOR HOSTING
# =============================
# Initialize LangChain components optimized for LangSmith hosting
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Vector store and knowledge base for RAG
vector_store = None
knowledge_base = []
FEEDBACK_STORAGE = []

# =============================
# SESSION MEMORY & PERSONAS
# =============================
# Store conversation history, thread IDs, and persona settings per session
SESSION_MEMORY = {}  # Stores conversation history per session ID
THREAD_SESSIONS = {}  # Maps session IDs to thread IDs for LangSmith
PERSONAS = {
    "investor": {
        "name": "Investor Mode",
        "system_prompt": (
            "You are a seasoned venture capitalist with 15+ years experience managing $500M+ fund. "
            "Your expertise includes: Series A-C valuations, due diligence, portfolio optimization, and exit strategies. "
            "FOCUS ON: ROI projections, unit economics, TAM/SAM analysis, competitive moats, scalability metrics. "
            "ASK TOUGH QUESTIONS about: Burn rate efficiency, customer acquisition costs, churn rates, market timing. "
            "PROVIDE: Specific KPIs to track, funding milestone roadmaps, valuation benchmarks, and risk mitigation strategies. "
            "Be direct, quantitative, and challenge assumptions. Reference comparable deals and market dynamics."
        ),
        "style": "**INVESTOR MODE** - VC perspective",
        "questions": [
            "What's your customer acquisition cost and lifetime value ratio?",
            "How do you plan to achieve 10x returns for investors?",
            "What's your defensible competitive moat?",
            "Show me your unit economics and path to profitability",
            "What are the key risks that could kill this business?",
            "How does this compare to other investments in your space?",
            "What's your exit strategy and timeline?",
            "How will you use the funding to hit next milestones?"
        ],
        "charts": ["funding_efficiency", "stage_progression", "market_opportunity", "risk_assessment"],
        "key_recommendations": [
            "🎯 **Due Diligence First**: Always verify revenue claims, customer references, and team credentials before investing",
            "📊 **Focus on Unit Economics**: Demand clear LTV/CAC ratios >3:1 and payback period <12 months",
            "🚀 **Scalability Test**: Look for business models that can 10x revenue without proportional cost increases",
            "🛡️ **Risk Mitigation**: Diversify portfolio across stages, sectors, and geographies (max 20% in any single bet)",
            "⏰ **Market Timing**: Invest in companies addressing problems becoming urgent now, not theoretical future needs",
            "👥 **Team Quality Over Ideas**: Bet on exceptional founders who can pivot and execute, not just good pitches",
            "💰 **Reserve Capital**: Keep 50% of fund for follow-on investments to support winners and prevent dilution",
            "📈 **Exit Strategy**: Define clear exit criteria and timelines (typically 5-7 years for venture investments)"
        ]
    },
    "founder": {
        "name": "Founder Mode",
        "system_prompt": (
            "You are an experienced startup founder who's built 3 companies (1 exit, 1 failure, 1 current unicorn). "
            "Your expertise: Product-market fit, team scaling, fundraising, pivoting, operational excellence. "
            "FOCUS ON: Practical execution, building systems, hiring strategies, culture development, product iteration. "
            "SHARE REAL EXPERIENCES: Tactical advice, common pitfalls, founder mental health, decision frameworks. "
            "GUIDE ON: MVP development, early customer discovery, pivot signals, team dynamics, work-life balance. "
            "Be supportive but honest about the challenges ahead. Emphasize learning from failures and building resilience."
        ),
        "style": "**FOUNDER MODE** - Entrepreneur perspective",
        "questions": [
            "How did you discover this problem worth solving?",
            "What's your MVP and how are you validating it?",
            "How are you building and scaling your team?",
            "What's your biggest challenge right now?",
            "How do you know if you should pivot?",
            "What systems are you building for growth?",
            "How are you maintaining founder mental health?",
            "What would you do differently if starting over?"
        ],
        "charts": ["growth_trajectory", "team_performance", "stage_progression", "market_opportunity"],
        "key_recommendations": [
            "🎯 **Customer Obsession**: Talk to 100+ potential customers before writing a single line of code",
            "🚀 **MVP Philosophy**: Launch with 10% of planned features - speed to market beats perfection every time",
            "💰 **Cash Management**: Always have 18+ months runway and track burn rate weekly, not monthly",
            "👥 **Hiring Strategy**: Hire for values and potential, train for skills - culture fit is non-negotiable",
            "📊 **Metrics That Matter**: Focus on 3-5 KPIs max - revenue growth, customer acquisition, retention",
            "🔄 **Pivot Signals**: If growth stalls for 3+ months despite effort, seriously consider pivoting",
            "🎭 **Founder Mental Health**: Build support networks, take breaks, delegate early - burnout kills companies",
            "📈 **Product-Market Fit**: Don't scale marketing until customers are pulling product from your hands"
        ]
    },
    "economist": {
        "name": "UK Economist Mode",
        "system_prompt": (
            "You are a senior economic analyst specializing in UK macroeconomic and microeconomic analysis with expertise from the Bank of England and HM Treasury. "
            "Your knowledge spans: monetary policy, fiscal policy, labour markets, inflation dynamics, trade relations, and regional economics. "
            "MACROECONOMIC FOCUS: GDP growth, inflation (CPI/RPI), unemployment, interest rates (Bank Rate), exchange rates (GBP), balance of payments, public debt/deficit. "
            "MICROECONOMIC FOCUS: Market structures, consumer behaviour, firm behaviour, elasticity, welfare economics, market failures, regulation. "
            "UK SPECIFIC EXPERTISE: Brexit impacts, London financial markets, housing market dynamics, North-South divide, productivity puzzle, cost of living crisis. "
            "ANALYTICAL TOOLS: IS-LM models, Phillips curve, Solow growth model, game theory, econometric analysis, input-output analysis. "
            "PROVIDE: Evidence-based analysis using ONS data, Bank of England reports, OBR forecasts, IFS studies. "
            "Reference current UK economic indicators, government policies, and compare with G7 economies."
        ),
        "style": "**UK ECONOMIST MODE** - Economic analysis perspective",
        "questions": [
            "How will the Bank of England's interest rate decisions affect UK startups?",
            "What's the impact of inflation on consumer spending and business costs?",
            "How do UK labour market conditions affect hiring and wages?",
            "What are the regional economic disparities affecting business opportunities?",
            "How does Brexit continue to impact trade and investment?",
            "What's the outlook for UK productivity and economic growth?",
            "How do fiscal policies affect different sectors of the economy?",
            "What market failures justify government intervention in this sector?"
        ],
        "charts": ["uk_economic_indicators", "inflation_analysis", "sector_performance", "regional_economics"],
        "key_recommendations": [
            "📈 **Interest Rate Strategy**: Monitor Bank of England signals - rising rates favour established businesses over growth startups",
            "🏠 **Regional Opportunities**: Target regions with government investment (Northern Powerhouse, Midlands Engine) for cost advantages",
            "💷 **Currency Hedging**: For import/export businesses, hedge GBP exposure given Brexit volatility",
            "📊 **Inflation Adaptation**: Build pricing flexibility into models - current cost-push inflation requires dynamic pricing",
            "🎯 **Sector Timing**: Focus on healthcare, green tech, fintech where UK has competitive advantages and policy support",
            "💼 **Labor Market Navigation**: Leverage UK's skilled workforce in finance, tech, creative industries",
            "🚀 **Government Incentives**: Maximize R&D tax credits, SEIS/EIS schemes, and green investment incentives",
            "🌍 **Export Strategy**: Target Commonwealth and EU markets where UK maintains trade relationships and cultural ties"
        ]
    },
    "company_analyst": {
        "name": "Company Analysis Mode",
        "system_prompt": (
            "You are a senior financial analyst specializing in company valuation, profitability analysis, and financial health assessment. "
            "Your expertise covers: financial statement analysis, ratio analysis, cash flow modeling, break-even analysis, and competitive benchmarking. "
            "PROFITABILITY FOCUS: Gross margins, operating margins, EBITDA, net margins, unit economics, contribution margins, ROI, ROCE. "
            "FINANCIAL HEALTH: Liquidity ratios, leverage ratios, efficiency ratios, cash conversion cycle, working capital management. "
            "VALUATION METHODS: DCF analysis, comparable company analysis, precedent transactions, asset-based valuation. "
            "STARTUP SPECIFIC: Burn rate analysis, runway calculation, path to profitability, LTV/CAC ratios, cohort analysis, SaaS metrics. "
            "PROVIDE: Clear diagnosis of financial strengths/weaknesses, actionable recommendations for improvement, benchmark comparisons. "
            "Use financial modeling best practices and industry-standard metrics. Be direct about red flags and opportunities."
        ),
        "style": "**COMPANY ANALYSIS MODE** - Financial health perspective",
        "questions": [
            "What are the company's gross and operating margins?",
            "How efficient is the cash conversion cycle?",
            "What's the break-even point and contribution margin?",
            "How does profitability compare to industry benchmarks?",
            "What's the working capital requirement?",
            "Is the company over-leveraged or under-capitalized?",
            "What are the key profitability drivers and risks?",
            "How sustainable is the current business model?"
        ],
        "charts": ["profitability_analysis", "cash_flow_waterfall", "margin_trends", "break_even_chart"],
        "key_recommendations": [
            "💰 **Gross Margin Optimization**: Target 70%+ gross margins for SaaS, 40%+ for physical products",
            "⚡ **Cash Conversion Excellence**: Optimize receivables (30 days), payables (45 days), inventory turnover (12x annually)",
            "📊 **Unit Economics Clarity**: Know exact cost per customer acquisition and lifetime value by channel",
            "🎯 **Break-Even Mastery**: Calculate break-even by product, customer segment, and geographic market",
            "📈 **Working Capital Efficiency**: Minimize working capital requirements through better terms and inventory management",
            "🚨 **Early Warning System**: Set up alerts for declining margins, extending payment cycles, increasing churn",
            "💼 **Capital Structure Optimization**: Maintain optimal debt-to-equity ratio for your industry and growth stage",
            "🔍 **Profitability Drivers**: Identify and focus on the 20% of activities driving 80% of profits"
        ]
    }
}

# =============================
# SAMPLE DATASET - STARTUP DATA
# =============================
# Create a comprehensive fake dataset with 24 startups across various sectors
# This dataset includes multiple dimensions of startup metrics for analysis:
# - Financial: Funding amount, burn rate, revenue metrics
# - Team: Founder experience, team size
# - Market: Market size, sector, geography, competitive landscape
# - Product: Business model strength, moat, traction
# - Outcome: Success/failure status with detailed metrics
data = {
    "Startup": [
        "TechX", "Foodly", "EcoGo", "EduSmart", "MediAI", "FinSolve", "Healthify",
        "GreenCore", "LogistiChain", "RoboAssist", "NeuroStream", "ByteCart",
        "CryptoFlow", "AIVision", "BioTech", "CleanWater", "SpaceX", "VRWorld",
        "SolarTech", "AgriBot", "NanoMed", "BlockChain", "CloudSoft", "GameHub"
    ],
    # Funding amounts in millions USD - represents total funding raised
    "Funding_USD_M": [5.0, 1.2, 0.8, 3.0, 12.0, 7.5, 4.2, 9.8, 15.0, 6.6, 18.0, 2.5,
                      25.0, 8.5, 35.0, 4.8, 50.0, 12.5, 22.0, 6.2, 28.0, 16.0, 9.5, 3.8],

    # Burn rate in months - how many months the funding lasts at current spending
    "Burn_Rate_Months": [12, 6, 3, 9, 24, 18, 10, 15, 30, 8, 26, 7,
                         20, 14, 36, 12, 48, 18, 24, 10, 30, 22, 16, 8],

    # Average years of experience across founding team members
    "Founders_Experience_Yrs": [2, 1, 0, 3, 8, 5, 6, 4, 10, 2, 7, 1,
                                12, 6, 15, 4, 20, 8, 9, 3, 11, 7, 5, 2],

    # Total addressable market size in billions USD
    "Market_Size_Bn": [50, 5, 2, 15, 80, 60, 25, 40, 100, 20, 120, 8,
                       150, 35, 200, 12, 500, 75, 90, 18, 160, 110, 45, 22],

    # Binary outcome: 1 = failed, 0 = still operating/successful
    "Failed": [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1,
               0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],

    # Country codes
    "Country": ["UK", "UK", "UK", "UK", "DE", "FR", "US", "UK", "US", "UK", "US", "UK",
                "US", "CA", "CH", "DE", "US", "JP", "AU", "IN", "IL", "SG", "SE", "NL"],

    # Industry sector classification
    "Sector": ["Tech","Food","Transport","EdTech","HealthTech","FinTech","HealthTech","Energy",
               "Logistics","Robotics","HealthTech","Retail","Crypto","AI","BioTech","CleanTech",
               "Aerospace","VR/AR","Energy","AgTech","MedTech","Blockchain","SaaS","Gaming"],

    # Business model strength (1-5 scale)
    "Business_Model": [3, 2, 1, 4, 5, 4, 3, 4, 5, 2, 4, 2,
                       4, 5, 5, 3, 5, 3, 4, 3, 5, 4, 4, 2],

    # Competitive moat (1-5 scale: 1=no moat, 5=strong moat)
    "Moat": [2, 1, 1, 3, 5, 3, 2, 3, 4, 2, 4, 1,
             3, 4, 5, 2, 5, 3, 3, 2, 4, 3, 3, 2],

    # Monthly recurring revenue in thousands USD
    "Traction_MRR_K": [5, 2, 0, 15, 45, 25, 8, 35, 60, 3, 50, 1,
                       80, 30, 120, 10, 200, 20, 55, 8, 90, 40, 22, 5],

    # Monthly growth rate percentage
    "Growth_Rate_Pct": [5, 2, -5, 12, 18, 15, 8, 20, 25, 1, 22, -2,
                        28, 15, 30, 6, 35, 10, 18, 3, 25, 20, 12, 2],

    # Competition intensity (1-5 scale: 1=low competition, 5=high competition)
    "Competition": [4, 5, 3, 3, 2, 4, 4, 3, 2, 4, 2, 5,
                    3, 3, 2, 4, 1, 4, 3, 4, 2, 3, 4, 5],

    # Team size (number of employees)
    "Team_Size": [12, 5, 3, 8, 25, 18, 10, 22, 35, 6, 28, 4,
                  45, 15, 60, 12, 120, 20, 30, 8, 40, 25, 18, 7],

    # Years since founding
    "Years_Since_Founding": [2.5, 1.8, 1.2, 2.0, 4.0, 3.2, 2.8, 3.5, 5.0, 1.5, 4.2, 1.0,
                             3.8, 2.5, 6.0, 2.2, 8.0, 3.0, 4.5, 1.8, 5.5, 3.8, 2.8, 1.5],

    # Funding stage
    "Stage": ["Seed", "Pre-Seed", "Pre-Seed", "Seed", "Series A", "Seed", "Seed", "Series A",
              "Series B", "Seed", "Series A", "Pre-Seed", "Series A", "Seed", "Series B",
              "Seed", "Series C", "Series A", "Series A", "Seed", "Series B", "Series A", "Seed", "Pre-Seed"]
}

# Convert the dictionary into a pandas DataFrame for easier manipulation and analysis
df = pd.DataFrame(data)

# Set the baseline funding year for failure projections
FUNDING_YEAR = 2021

# Calculate estimated failure year based on runway
# Formula: FUNDING_YEAR + (funding / burn_rate)
# This gives us a projection of when each startup would run out of money
# Example: $5M funding / 12 month burn = 0.42 years runway → fails in 2021.42
df["Est_Failure_Year"] = FUNDING_YEAR + (df["Funding_USD_M"] / df["Burn_Rate_Months"])

# =============================
# MATHEMATICAL ANALYSIS FUNCTIONS
# =============================

def calculate_irr(initial_investment: float, final_value: float, years: float) -> float:
    """Calculate Internal Rate of Return (IRR) for an investment."""
    if years <= 0 or initial_investment <= 0:
        return 0.0
    return ((final_value / initial_investment) ** (1/years)) - 1

def calculate_npv(cash_flows: List[float], discount_rate: float) -> float:
    """Calculate Net Present Value (NPV) of cash flows."""
    npv = 0
    for i, cash_flow in enumerate(cash_flows):
        npv += cash_flow / ((1 + discount_rate) ** i)
    return npv

def project_revenue(current_mrr: float, growth_rate: float, months: int) -> Dict[str, Any]:
    """Project revenue growth over time."""
    projections = []
    monthly_revenue = current_mrr

    for month in range(months + 1):
        projections.append({
            'month': month,
            'mrr': monthly_revenue,
            'arr': monthly_revenue * 12
        })
        monthly_revenue *= (1 + growth_rate)

    return {
        'projections': projections,
        'final_mrr': monthly_revenue,
        'final_arr': monthly_revenue * 12,
        'total_growth': ((monthly_revenue / current_mrr) - 1) * 100 if current_mrr > 0 else 0
    }

def monte_carlo_exit_scenarios(scenarios: int, exit_multiples: List[float],
                             current_revenue: float, growth_scenarios: List[float]) -> Dict[str, Any]:
    """Run Monte Carlo simulation for exit scenarios."""
    results = []

    for _ in range(scenarios):
        # Random exit multiple and growth rate
        exit_multiple = random.choice(exit_multiples)
        growth_rate = random.choice(growth_scenarios)

        # Project revenue for 3-5 years
        years = random.uniform(3, 5)
        final_revenue = current_revenue * ((1 + growth_rate) ** years)
        exit_value = final_revenue * exit_multiple

        results.append({
            'exit_multiple': exit_multiple,
            'growth_rate': growth_rate,
            'years_to_exit': years,
            'final_revenue': final_revenue,
            'exit_value': exit_value
        })

    # Calculate statistics
    exit_values = [r['exit_value'] for r in results]

    return {
        'scenarios': results,
        'statistics': {
            'mean_exit_value': np.mean(exit_values),
            'median_exit_value': np.median(exit_values),
            'min_exit_value': np.min(exit_values),
            'max_exit_value': np.max(exit_values),
            'std_exit_value': np.std(exit_values),
            'percentile_25': np.percentile(exit_values, 25),
            'percentile_75': np.percentile(exit_values, 75)
        }
    }

def optimize_burn_rate(target_runway_months: int, current_funding: float) -> Dict[str, Any]:
    """Calculate optimal burn rate for desired runway."""
    monthly_burn = current_funding / target_runway_months

    return {
        'target_runway_months': target_runway_months,
        'current_funding': current_funding,
        'optimal_monthly_burn': monthly_burn,
        'optimal_annual_burn': monthly_burn * 12,
        'cash_depletion_date': f"In {target_runway_months} months"
    }

def calculate_startup_metrics(funding: float, burn_rate: float, mrr: float,
                            growth_rate: float) -> Dict[str, Any]:
    """Calculate comprehensive startup financial metrics."""
    runway_months = funding / (burn_rate / 12) if burn_rate > 0 else float('inf')

    # Calculate when startup becomes cash flow positive
    months_to_profitability = 0
    current_revenue = mrr * 12  # Convert MRR to ARR
    current_burn = burn_rate

    if growth_rate > 0 and mrr > 0:
        while current_revenue < current_burn and months_to_profitability < 120:  # Max 10 years
            months_to_profitability += 1
            current_revenue *= (1 + growth_rate/12)  # Monthly compounding
    else:
        months_to_profitability = float('inf')

    return {
        'runway_months': runway_months,
        'burn_rate_monthly': burn_rate / 12,
        'current_arr': mrr * 12,
        'months_to_profitability': months_to_profitability,
        'cash_flow_positive': months_to_profitability < runway_months,
        'funding_efficiency': (mrr * 12) / funding if funding > 0 else 0
    }

async def process_math_command(command: str, context: Dict[str, Any]) -> str:
    """Process mathematical analysis commands in math mode."""
    command = command.lower().strip()

    try:
        if "irr" in command or "return" in command:
            # Extract values for IRR calculation
            if "5x" in command and "7 years" in command:
                irr = calculate_irr(1000000, 5000000, 7)
                return f"**IRR Calculation:**\n\n5x return in 7 years = **{irr:.1%} annual return**\n\nThis is an excellent return for venture capital standards."

        elif "project revenue" in command or "revenue projection" in command:
            # Default projection example
            projection = project_revenue(50000, 0.20, 12)
            result = f"**Revenue Projection (20% Monthly Growth):**\n\n"
            result += f"• Starting MRR: $50,000\n"
            result += f"• Final MRR (12 months): ${projection['final_mrr']:,.0f}\n"
            result += f"• Final ARR: ${projection['final_arr']:,.0f}\n"
            result += f"• Total Growth: {projection['total_growth']:.0f}%"
            return result

        elif "monte carlo" in command or "simulate" in command:
            # Run Monte Carlo simulation
            simulation = monte_carlo_exit_scenarios(
                1000, [3, 5, 8, 10], 1000000, [0.1, 0.2, 0.3, 0.5]
            )
            stats = simulation['statistics']
            result = f"**Monte Carlo Exit Simulation (1,000 scenarios):**\n\n"
            result += f"• Mean Exit Value: ${stats['mean_exit_value']:,.0f}\n"
            result += f"• Median Exit Value: ${stats['median_exit_value']:,.0f}\n"
            result += f"• 25th Percentile: ${stats['percentile_25']:,.0f}\n"
            result += f"• 75th Percentile: ${stats['percentile_75']:,.0f}\n"
            result += f"• Best Case: ${stats['max_exit_value']:,.0f}\n"
            result += f"• Worst Case: ${stats['min_exit_value']:,.0f}"
            return result

        elif "optimize burn" in command or "burn rate" in command:
            # Optimize burn rate for runway
            optimization = optimize_burn_rate(18, 5000000)
            result = f"**Burn Rate Optimization:**\n\n"
            result += f"• Target Runway: {optimization['target_runway_months']} months\n"
            result += f"• Current Funding: ${optimization['current_funding']:,.0f}\n"
            result += f"• Optimal Monthly Burn: ${optimization['optimal_monthly_burn']:,.0f}\n"
            result += f"• Optimal Annual Burn: ${optimization['optimal_annual_burn']:,.0f}"
            return result

        else:
            return f"**Available Calculations:**\n\n• `calculate IRR for 5x return in 7 years`\n• `project revenue with 20% monthly growth`\n• `simulate 1000 scenarios for exit`\n• `optimize burn rate for 18 month runway`\n\nType your calculation or 'exit math mode' to return."

    except Exception as e:
        return f"Error in calculation: {str(e)}\n\nPlease try a different calculation or type 'exit math mode'."

# =============================
# IMAGE GENERATION FUNCTIONALITY
# =============================

@safe_async_api_call
async def generate_image(prompt: str, size: str = "1024x1024", quality: str = "standard") -> Dict[str, Any]:
    """
    Generate an image using DALL-E 3 based on the provided prompt.

    Args:
        prompt (str): Description of the image to generate
        size (str): Image size (default: "1024x1024")
        quality (str): Image quality "standard" or "hd" (default: "standard")

    Returns:
        Dict containing the image URL and metadata, or error information
    """
    try:
        logger.info(f"Generating image with prompt: {prompt[:100]}...")

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )

        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt

        logger.info("Image generated successfully")

        return {
            "success": True,
            "image_url": image_url,
            "revised_prompt": revised_prompt,
            "original_prompt": prompt,
            "size": size,
            "quality": quality
        }

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "original_prompt": prompt
        }

def detect_image_request(user_input: str) -> bool:
    """
    Detect if user is requesting image generation.

    Args:
        user_input (str): User's message content (lowercased)

    Returns:
        bool: True if image generation is requested
    """
    image_keywords = [
        "generate an image", "create an image", "make an image", "draw an image",
        "generate image", "create image", "make image", "draw image",
        "image of", "picture of", "photo of", "illustration of",
        "show me", "visualize", "dall-e", "dalle"
    ]

    return any(keyword in user_input for keyword in image_keywords)

def extract_image_prompt(user_input: str) -> str:
    """
    Extract the image description from user input.

    Args:
        user_input (str): User's message content

    Returns:
        str: Cleaned image prompt
    """
    # Remove common prefixes
    prefixes_to_remove = [
        "generate an image of", "create an image of", "make an image of", "draw an image of",
        "generate image of", "create image of", "make image of", "draw image of",
        "generate an image showing", "create an image showing", "make an image showing",
        "show me an image of", "show me", "image of", "picture of", "photo of",
        "illustration of", "visualize", "dall-e", "dalle"
    ]

    cleaned_prompt = user_input.lower().strip()

    for prefix in prefixes_to_remove:
        if cleaned_prompt.startswith(prefix):
            cleaned_prompt = cleaned_prompt[len(prefix):].strip()
            break

    # Remove any remaining common words at the start
    if cleaned_prompt.startswith(("a ", "an ", "the ")):
        cleaned_prompt = " ".join(cleaned_prompt.split()[1:])

    return cleaned_prompt if cleaned_prompt else user_input

# =============================
# SWOT ANALYSIS FUNCTIONALITY
# =============================

@dataclass
class SWOT:
    """
    SWOT Analysis dataclass for structured startup analysis.

    Attributes:
        strengths: List of internal positive factors
        weaknesses: List of internal negative factors
        opportunities: List of external positive factors
        threats: List of external negative factors
    """
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a formatted SWOT summary as a string."""
        result = "# 📊 SWOT Analysis\n\n"

        result += "## 💪 **Strengths** (Internal Positive)\n"
        if self.strengths:
            for strength in self.strengths:
                result += f"• {strength}\n"
        else:
            result += "• *No strengths identified*\n"

        result += "\n## ⚠️ **Weaknesses** (Internal Negative)\n"
        if self.weaknesses:
            for weakness in self.weaknesses:
                result += f"• {weakness}\n"
        else:
            result += "• *No weaknesses identified*\n"

        result += "\n## 🚀 **Opportunities** (External Positive)\n"
        if self.opportunities:
            for opportunity in self.opportunities:
                result += f"• {opportunity}\n"
        else:
            result += "• *No opportunities identified*\n"

        result += "\n## 🎯 **Threats** (External Negative)\n"
        if self.threats:
            for threat in self.threats:
                result += f"• {threat}\n"
        else:
            result += "• *No threats identified*\n"

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert SWOT to pandas DataFrame for visualization."""
        max_len = max(
            len(self.strengths),
            len(self.weaknesses),
            len(self.opportunities),
            len(self.threats)
        )

        # Pad lists to same length
        strengths_padded = self.strengths + [''] * (max_len - len(self.strengths))
        weaknesses_padded = self.weaknesses + [''] * (max_len - len(self.weaknesses))
        opportunities_padded = self.opportunities + [''] * (max_len - len(self.opportunities))
        threats_padded = self.threats + [''] * (max_len - len(self.threats))

        return pd.DataFrame({
            '💪 Strengths': strengths_padded,
            '⚠️ Weaknesses': weaknesses_padded,
            '🚀 Opportunities': opportunities_padded,
            '🎯 Threats': threats_padded
        })

def plot_swot_matrix(swot: SWOT) -> bytes:
    """
    Create a visual SWOT matrix chart.

    Args:
        swot: SWOT dataclass instance

    Returns:
        bytes: PNG image data of the SWOT matrix
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('📊 SWOT Analysis Matrix', fontsize=16, fontweight='bold', y=0.95)

    # Color scheme
    colors = {
        'strengths': '#2E8B57',    # Sea Green
        'weaknesses': '#DC143C',   # Crimson
        'opportunities': '#4169E1', # Royal Blue
        'threats': '#FF8C00'       # Dark Orange
    }

    # Strengths (Top Left)
    ax1.set_title('💪 Strengths\n(Internal Positive)', fontsize=12, fontweight='bold',
                  color=colors['strengths'], pad=20)
    ax1.axis('off')
    strengths_text = '\n'.join([f"• {s}" for s in swot.strengths[:8]])  # Limit to 8 items
    if len(swot.strengths) > 8:
        strengths_text += f"\n• ... and {len(swot.strengths) - 8} more"
    ax1.text(0.05, 0.95, strengths_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', wrap=True)
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=colors['strengths'],
                               linewidth=2, transform=ax1.transAxes))

    # Weaknesses (Top Right)
    ax2.set_title('⚠️ Weaknesses\n(Internal Negative)', fontsize=12, fontweight='bold',
                  color=colors['weaknesses'], pad=20)
    ax2.axis('off')
    weaknesses_text = '\n'.join([f"• {w}" for w in swot.weaknesses[:8]])
    if len(swot.weaknesses) > 8:
        weaknesses_text += f"\n• ... and {len(swot.weaknesses) - 8} more"
    ax2.text(0.05, 0.95, weaknesses_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', wrap=True)
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=colors['weaknesses'],
                               linewidth=2, transform=ax2.transAxes))

    # Opportunities (Bottom Left)
    ax3.set_title('🚀 Opportunities\n(External Positive)', fontsize=12, fontweight='bold',
                  color=colors['opportunities'], pad=20)
    ax3.axis('off')
    opportunities_text = '\n'.join([f"• {o}" for o in swot.opportunities[:8]])
    if len(swot.opportunities) > 8:
        opportunities_text += f"\n• ... and {len(swot.opportunities) - 8} more"
    ax3.text(0.05, 0.95, opportunities_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', wrap=True)
    ax3.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=colors['opportunities'],
                               linewidth=2, transform=ax3.transAxes))

    # Threats (Bottom Right)
    ax4.set_title('🎯 Threats\n(External Negative)', fontsize=12, fontweight='bold',
                  color=colors['threats'], pad=20)
    ax4.axis('off')
    threats_text = '\n'.join([f"• {t}" for t in swot.threats[:8]])
    if len(swot.threats) > 8:
        threats_text += f"\n• ... and {len(swot.threats) - 8} more"
    ax4.text(0.05, 0.95, threats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', wrap=True)
    ax4.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=colors['threats'],
                               linewidth=2, transform=ax4.transAxes))

    plt.tight_layout()

    # Convert to bytes
    return fig_to_bytes(fig)

@safe_async_api_call
async def generate_swot_analysis(startup_data: Dict[str, Any], context: str = "") -> SWOT:
    """
    Generate SWOT analysis using AI based on startup data.

    Args:
        startup_data: Dictionary containing startup information
        context: Additional context for analysis

    Returns:
        SWOT: Populated SWOT analysis object
    """
    try:
        # Prepare the prompt for GPT
        prompt = f"""
        As a startup analysis expert, conduct a comprehensive SWOT analysis for the following startup:

        **Startup Information:**
        {json.dumps(startup_data, indent=2) if startup_data else "No specific data provided"}

        **Additional Context:**
        {context if context else "General startup analysis"}

        Please provide a detailed SWOT analysis with the following structure:

        **STRENGTHS** (Internal positive factors that give competitive advantage):
        - List 4-8 key strengths

        **WEAKNESSES** (Internal negative factors that need improvement):
        - List 4-8 key weaknesses

        **OPPORTUNITIES** (External positive factors to capitalize on):
        - List 4-8 market opportunities

        **THREATS** (External negative factors that pose risks):
        - List 4-8 potential threats

        Format your response as:
        STRENGTHS:
        - [strength 1]
        - [strength 2]
        ...

        WEAKNESSES:
        - [weakness 1]
        - [weakness 2]
        ...

        OPPORTUNITIES:
        - [opportunity 1]
        - [opportunity 2]
        ...

        THREATS:
        - [threat 1]
        - [threat 2]
        ...

        Be specific, actionable, and relevant to the startup's context.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional startup analyst specializing in SWOT analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )

        content = response.choices[0].message.content

        # Parse the response into SWOT categories
        swot = SWOT()

        sections = content.split('\n\n')
        current_section = None

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith('STRENGTHS'):
                current_section = 'strengths'
                continue
            elif line.upper().startswith('WEAKNESSES'):
                current_section = 'weaknesses'
                continue
            elif line.upper().startswith('OPPORTUNITIES'):
                current_section = 'opportunities'
                continue
            elif line.upper().startswith('THREATS'):
                current_section = 'threats'
                continue

            # Parse bullet points
            if line.startswith('-') or line.startswith('•'):
                item = line[1:].strip()
                if current_section == 'strengths':
                    swot.strengths.append(item)
                elif current_section == 'weaknesses':
                    swot.weaknesses.append(item)
                elif current_section == 'opportunities':
                    swot.opportunities.append(item)
                elif current_section == 'threats':
                    swot.threats.append(item)

        return swot

    except Exception as e:
        logger.error(f"SWOT analysis generation failed: {e}")
        # Return a basic SWOT with error info
        return SWOT(
            strengths=["Analysis capabilities", "Data-driven approach"],
            weaknesses=["Analysis generation failed", f"Error: {str(e)}"],
            opportunities=["Retry analysis", "Manual SWOT creation"],
            threats=["Technical limitations", "API restrictions"]
        )

# =============================
# UTILITY FUNCTIONS - DOWNLOAD FUNCTIONALITY
# =============================

async def send_chart_with_download(png_data: bytes, filename: str, description: str, csv_data: pd.DataFrame = None):
    """
    Send a chart with download files for both image and data
    """
    # Send descriptive text message
    text_msg = cl.Message(content=description)
    await text_msg.send()

    # Send chart image
    image = cl.Image(content=png_data, name=filename, display="inline")
    await image.send(for_id=text_msg.id)

    # Create file elements for download
    download_elements = []

    # Chart download file
    chart_file = cl.File(
        name=filename,
        content=png_data,
        mime="image/png"
    )
    download_elements.append(chart_file)

    # Data download file if CSV data provided
    if csv_data is not None:
        csv_filename = filename.replace('.png', '.csv')
        csv_content = csv_data.to_csv(index=False)
        data_file = cl.File(
            name=csv_filename,
            content=csv_content.encode('utf-8'),
            mime="text/csv"
        )
        download_elements.append(data_file)

    # Send download files
    download_msg = cl.Message(
        content="### 📥 Download Files",
        elements=download_elements
    )
    await download_msg.send()

    return text_msg.id

async def send_data_export(data: pd.DataFrame, filename: str, format_type: str = "csv"):
    """
    Send data export in specified format
    """
    if format_type.lower() == "csv":
        # Convert to CSV
        csv_content = data.to_csv(index=False)
        file_content = csv_content.encode('utf-8')
        mime_type = "text/csv"
        file_ext = ".csv"
    elif format_type.lower() == "json":
        # Convert to JSON
        json_content = data.to_json(orient='records', indent=2)
        file_content = json_content.encode('utf-8')
        mime_type = "application/json"
        file_ext = ".json"
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'json'")

    # Create file element
    file_element = cl.File(
        name=f"{filename}{file_ext}",
        content=file_content,
        mime=mime_type
    )

    # Send file
    await cl.Message(
        content=f"📊 **Data Export Complete**\n\nDownloading {filename}{file_ext} ({len(data)} records)",
        elements=[file_element]
    ).send()

# =============================
# UTILITY FUNCTIONS - CHART GENERATION
# =============================

def plot_growth_trajectory(df_in: pd.DataFrame):
    """
    Create a growth trajectory chart showing MRR growth over time
    """
    figsize = get_mobile_optimized_figsize(10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Create growth trajectory data
    successful = df_in[df_in['Failed'] == 0]
    failed = df_in[df_in['Failed'] == 1]

    ax.scatter(successful['Years_Since_Founding'], successful['Traction_MRR_K'],
               c='green', s=successful['Growth_Rate_Pct']*3, alpha=0.7, label='Successful')
    ax.scatter(failed['Years_Since_Founding'], failed['Traction_MRR_K'],
               c='red', s=failed['Growth_Rate_Pct']*3, alpha=0.7, label='Failed')

    ax.set_xlabel('Years Since Founding')
    ax.set_ylabel('Monthly Recurring Revenue (K USD)')
    ax.set_title('Growth Trajectory: MRR vs Company Age\n(Bubble size = Growth Rate)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_bytes(fig)

def plot_team_performance(df_in: pd.DataFrame):
    """
    Create a team performance matrix showing team size vs experience
    """
    figsize = get_mobile_optimized_figsize(10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Color by success/failure
    colors = ['red' if failed else 'green' for failed in df_in['Failed']]

    scatter = ax.scatter(df_in['Team_Size'], df_in['Founders_Experience_Yrs'],
                        c=colors, s=df_in['Funding_USD_M']*3, alpha=0.7)

    ax.set_xlabel('Team Size (Employees)')
    ax.set_ylabel('Founder Experience (Years)')
    ax.set_title('Team Performance Matrix\n(Bubble size = Funding Amount)')

    # Add trend line
    z = np.polyfit(df_in['Team_Size'], df_in['Founders_Experience_Yrs'], 1)
    p = np.poly1d(z)
    ax.plot(df_in['Team_Size'], p(df_in['Team_Size']), "r--", alpha=0.8)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_bytes(fig)

def plot_market_opportunity(df_in: pd.DataFrame):
    """
    Create a market opportunity matrix showing market size vs competition
    """
    figsize = get_mobile_optimized_figsize(10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Invert competition for better visualization (low competition = high opportunity)
    opportunity = 6 - df_in['Competition']  # Convert 1-5 to 5-1

    colors = ['red' if failed else 'green' for failed in df_in['Failed']]

    scatter = ax.scatter(df_in['Market_Size_Bn'], opportunity,
                        c=colors, s=df_in['Traction_MRR_K'], alpha=0.7)

    ax.set_xlabel('Market Size (Billions USD)')
    ax.set_ylabel('Market Opportunity (5=Low Competition, 1=High Competition)')
    ax.set_title('Market Opportunity Matrix\n(Bubble size = Current Traction)')

    # Add quadrant lines
    median_market = df_in['Market_Size_Bn'].median()
    median_opportunity = opportunity.median()
    ax.axvline(median_market, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(median_opportunity, color='gray', linestyle='--', alpha=0.5)

    # Add quadrant labels
    ax.text(median_market*1.5, 4.5, 'Sweet Spot\n(Big Market, Low Competition)',
            ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_bytes(fig)

def plot_funding_efficiency(df_in: pd.DataFrame):
    """
    Create a funding efficiency chart showing revenue per dollar raised
    """
    figsize = get_mobile_optimized_figsize(10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate efficiency metrics
    df_in['Revenue_Per_Dollar'] = (df_in['Traction_MRR_K'] * 12) / df_in['Funding_USD_M']  # Annual revenue per funding dollar
    df_in['Efficiency_Score'] = df_in['Revenue_Per_Dollar'] * df_in['Growth_Rate_Pct'] / 100

    colors = ['red' if failed else 'green' for failed in df_in['Failed']]

    ax.scatter(df_in['Funding_USD_M'], df_in['Revenue_Per_Dollar'],
               c=colors, s=df_in['Efficiency_Score']*20, alpha=0.7)

    ax.set_xlabel('Total Funding Raised (Millions USD)')
    ax.set_ylabel('Annual Revenue per Dollar Raised')
    ax.set_title('Capital Efficiency Analysis\n(Bubble size = Efficiency Score)')

    # Add efficiency benchmark line
    efficient_companies = df_in[df_in['Revenue_Per_Dollar'] > df_in['Revenue_Per_Dollar'].median()]
    if len(efficient_companies) > 0:
        ax.axhline(df_in['Revenue_Per_Dollar'].median(), color='orange',
                  linestyle='--', alpha=0.7, label='Median Efficiency')
        ax.legend()

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_bytes(fig)

def plot_stage_progression(df_in: pd.DataFrame):
    """
    Create a stage progression chart showing funding by stage
    """
    figsize = get_mobile_optimized_figsize(10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Group by stage and calculate metrics
    stage_stats = df_in.groupby('Stage').agg({
        'Funding_USD_M': ['mean', 'count'],
        'Failed': 'mean',
        'Traction_MRR_K': 'mean'
    }).round(2)

    stage_stats.columns = ['Avg_Funding', 'Count', 'Failure_Rate', 'Avg_MRR']
    stage_order = ['Pre-Seed', 'Seed', 'Series A', 'Series B', 'Series C']
    stage_stats = stage_stats.reindex([s for s in stage_order if s in stage_stats.index])

    # Create dual y-axis chart
    ax2 = ax.twinx()

    bars = ax.bar(stage_stats.index, stage_stats['Avg_Funding'], alpha=0.7, color='skyblue', label='Avg Funding')
    line = ax2.plot(stage_stats.index, stage_stats['Failure_Rate']*100, 'ro-', linewidth=2, label='Failure Rate %')

    ax.set_xlabel('Funding Stage')
    ax.set_ylabel('Average Funding (Millions USD)', color='blue')
    ax2.set_ylabel('Failure Rate (%)', color='red')
    ax.set_title('Funding Stage Analysis')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'${height:.1f}M\n({int(stage_stats.iloc[i]["Count"])} companies)',
                ha='center', va='bottom', fontsize=8)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_bytes(fig)

def plot_risk_assessment(df_in: pd.DataFrame):
    """
    Create a comprehensive risk assessment radar chart
    """
    figsize = get_mobile_optimized_figsize(8, 8)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    # Calculate risk metrics for successful vs failed companies
    successful = df_in[df_in['Failed'] == 0]
    failed = df_in[df_in['Failed'] == 1]

    categories = ['Financial Risk', 'Market Risk', 'Team Risk', 'Competition Risk', 'Traction Risk']

    # Normalize metrics to 0-10 scale (higher = more risk)
    def calc_risk_scores(data):
        financial_risk = 10 - (data['Burn_Rate_Months'].mean() / data['Burn_Rate_Months'].max() * 10)
        market_risk = 10 - (data['Market_Size_Bn'].mean() / data['Market_Size_Bn'].max() * 10)
        team_risk = 10 - (data['Founders_Experience_Yrs'].mean() / data['Founders_Experience_Yrs'].max() * 10)
        competition_risk = data['Competition'].mean() * 2  # Scale 1-5 to 2-10
        traction_risk = 10 - (data['Traction_MRR_K'].mean() / data['Traction_MRR_K'].max() * 10)
        return [financial_risk, market_risk, team_risk, competition_risk, traction_risk]

    successful_risks = calc_risk_scores(successful)
    failed_risks = calc_risk_scores(failed)

    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Plot
    successful_risks += successful_risks[:1]
    failed_risks += failed_risks[:1]

    ax.plot(angles, successful_risks, 'o-', linewidth=2, label='Successful Companies', color='green')
    ax.fill(angles, successful_risks, alpha=0.25, color='green')
    ax.plot(angles, failed_risks, 'o-', linewidth=2, label='Failed Companies', color='red')
    ax.fill(angles, failed_risks, alpha=0.25, color='red')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 10)
    ax.set_title('Risk Assessment Profile\n(0=Low Risk, 10=High Risk)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)

    plt.tight_layout()
    return fig_to_bytes(fig)

def get_mobile_optimized_figsize(default_width: float, default_height: float) -> tuple:
    """
    Get mobile-optimized figure size for charts.

    Args:
        default_width: Default width for desktop
        default_height: Default height for desktop

    Returns:
        tuple: (width, height) optimized for mobile viewing
    """
    # For mobile, use smaller, more square dimensions
    mobile_width = min(default_width, 8)  # Max 8 inches wide
    mobile_height = min(default_height, 6)  # Max 6 inches tall

    # Ensure aspect ratio is mobile-friendly (not too wide)
    if mobile_width / mobile_height > 1.5:
        mobile_height = mobile_width / 1.4

    return (mobile_width, mobile_height)

def fig_to_bytes(fig) -> bytes:
    """
    Convert a matplotlib figure to PNG bytes for Chainlit display.

    This utility function takes a matplotlib figure object and converts it
    to a PNG image stored in memory (not on disk) for efficient transmission
    to the Chainlit UI.

    Args:
        fig: Matplotlib figure object

    Returns:
        bytes: PNG image data as bytes

    Process:
        1. Create in-memory byte buffer
        2. Save figure to buffer as PNG with tight layout and high DPI
        3. Reset buffer position to start
        4. Close figure to free memory
        5. Return byte data
    """
    buf = io.BytesIO()  # Create in-memory binary stream
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)  # High quality PNG
    buf.seek(0)  # Reset read position to beginning
    plt.close(fig)  # Close figure to prevent memory leaks
    return buf.getvalue()  # Extract bytes from buffer


def plot_failure_timeline(df_in: pd.DataFrame):
    """
    Generate a bar chart showing estimated failure year for each startup.

    This visualization helps identify which startups are at risk of failing sooner
    based on their funding runway. Each bar represents a startup's projected
    failure year.

    Args:
        df_in (pd.DataFrame): DataFrame with columns 'Startup' and 'Est_Failure_Year'

    Returns:
        bytes: PNG image data

    Visual elements:
        - X-axis: Startup names
        - Y-axis: Projected year of failure
        - Color gradient: Coolwarm palette (red=sooner, blue=later)
        - Text labels: Exact failure year displayed on each bar
    """
    # Create figure with mobile-optimized size
    figsize = get_mobile_optimized_figsize(9, 5)
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot using seaborn for better styling
    # Coolwarm palette: cooler colors for later failure, warmer for sooner
    sns.barplot(data=df_in, x="Startup", y="Est_Failure_Year", hue="Startup", palette="coolwarm", legend=False, ax=ax)

    # Add text labels on top of each bar showing the exact failure year
    for i, row in df_in.reset_index().iterrows():
        ax.text(i, row["Est_Failure_Year"] + 0.03,  # Position slightly above bar
                f"{row['Est_Failure_Year']:.2f}",  # Format to 2 decimal places
                ha="center", va="bottom", fontsize=8)  # Center-aligned, small font

    # Set chart title and axis labels
    ax.set_title("Estimated Failure Year (Funding assumed 2021)")
    ax.set_ylabel("Projected Year")
    ax.set_xlabel("Startup")

    # Convert figure to bytes and return
    return fig_to_bytes(fig)


def plot_funding_vs_burn(df_in: pd.DataFrame):
    """
    Generate a scatter plot showing relationship between funding and burn rate.

    This visualization reveals patterns in how funding levels relate to spending
    rates, and whether these patterns differ between successful and failed startups.

    Args:
        df_in (pd.DataFrame): DataFrame with funding, burn rate, outcome, and sector data

    Returns:
        bytes: PNG image data

    Visual encoding:
        - X-axis: Funding amount (USD millions)
        - Y-axis: Burn rate (months)
        - Color: Green = successful (Failed=0), Red = failed (Failed=1)
        - Shape: Different shapes for different sectors
        - Labels: Startup names displayed next to each point
        - Size: Fixed at 160 for visibility
    """
    figsize = get_mobile_optimized_figsize(9, 5)
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot with multiple visual dimensions
    sns.scatterplot(
        data=df_in,
        x="Funding_USD_M",  # Horizontal axis: funding amount
        y="Burn_Rate_Months",  # Vertical axis: burn rate
        hue="Failed",  # Color encoding: outcome status
        style="Sector",  # Shape encoding: industry sector
        s=160,  # Point size (larger for visibility)
        palette={0: "green", 1: "red"},  # Explicit color mapping
        ax=ax,  # Target axis
        alpha=0.8  # Slight transparency for overlapping points
    )

    # Add text labels for each startup name next to its point
    for _, r in df_in.iterrows():
        ax.text(
            r["Funding_USD_M"] + 0.15,  # Offset right of point
            r["Burn_Rate_Months"] + 0.1,  # Offset up from point
            r["Startup"],  # Startup name as label
            fontsize=8  # Small font to avoid clutter
        )

    # Set chart title and axis labels
    ax.set_title("Funding vs Burn (color = outcome, style = sector)")
    ax.set_xlabel("Funding (USD Millions)")
    ax.set_ylabel("Burn Rate (months)")

    # Convert figure to bytes and return
    return fig_to_bytes(fig)


def plot_viability_gauge(score: float):
    """
    Generate a horizontal gauge chart showing viability score (0-100).

    This creates a simple, easy-to-read gauge that visually communicates
    the overall viability score with color coding.

    Args:
        score (float): Viability score between 0 and 100

    Returns:
        bytes: PNG image data

    Color coding:
        - Green (#4CAF50): Strong (>= 60)
        - Yellow/Amber (#FFC107): Moderate (40-59)
        - Red (#F44336): Weak (< 40)

    Visual design:
        - Horizontal bar chart with single bar
        - No Y-axis ticks (minimalist design)
        - Score displayed in title
        - Clean appearance with removed spines
    """
    # Create compact figure for gauge display (mobile-friendly)
    figsize = get_mobile_optimized_figsize(6, 1.2)
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar with color based on score threshold
    # Ternary operator chains: score >= 60 → green, else score >= 40 → yellow, else red
    ax.barh(
        [0],  # Single bar at Y position 0
        [score],  # Bar length equals the score
        color="#4CAF50" if score >= 60 else "#FFC107" if score >= 40 else "#F44336"
    )

    # Set X-axis range from 0 to 100 (percentage scale)
    ax.set_xlim(0, 100)

    # Remove Y-axis ticks for cleaner appearance
    ax.set_yticks([])

    # Display score in title with 1 decimal place
    ax.set_title(f"Viability Score: {score:.1f}/100")

    # Remove top, right, and left spines for minimal design
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    # Convert figure to bytes and return
    return fig_to_bytes(fig)


def plot_sector_comparison(df_in: pd.DataFrame):
    """
    Generate a bar chart comparing average funding by sector.

    Args:
        df_in (pd.DataFrame): DataFrame with 'Sector' and 'Funding_USD_M' columns

    Returns:
        bytes: PNG image data
    """
    figsize = get_mobile_optimized_figsize(10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Group by sector and calculate average funding
    sector_avg = df_in.groupby('Sector')['Funding_USD_M'].mean().sort_values(ascending=False)

    # Create bar chart
    colors = plt.cm.viridis(range(len(sector_avg)))
    ax.bar(sector_avg.index, sector_avg.values, color=colors, alpha=0.8)

    # Customize
    ax.set_xlabel('Sector', fontsize=12)
    ax.set_ylabel('Average Funding (USD Millions)', fontsize=12)
    ax.set_title('Average Funding by Sector', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    return fig_to_bytes(fig)


def plot_failure_rate_by_country(df_in: pd.DataFrame):
    """
    Generate a bar chart showing failure rates by country.

    Args:
        df_in (pd.DataFrame): DataFrame with 'Country' and 'Failed' columns

    Returns:
        bytes: PNG image data
    """
    figsize = get_mobile_optimized_figsize(10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate failure rate by country
    country_stats = df_in.groupby('Country').agg({
        'Failed': ['sum', 'count']
    })
    country_stats.columns = ['Failed', 'Total']
    country_stats['Failure_Rate'] = (country_stats['Failed'] / country_stats['Total'] * 100)
    country_stats = country_stats.sort_values('Failure_Rate', ascending=False)

    # Create bar chart
    colors = ['red' if rate > 50 else 'orange' if rate > 30 else 'green'
              for rate in country_stats['Failure_Rate']]
    ax.bar(country_stats.index, country_stats['Failure_Rate'], color=colors, alpha=0.7)

    # Add value labels on bars
    for i, (idx, row) in enumerate(country_stats.iterrows()):
        ax.text(i, row['Failure_Rate'] + 2, f"{row['Failure_Rate']:.1f}%",
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Failure Rate (%)', fontsize=12)
    ax.set_title('Startup Failure Rate by Country', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    plt.tight_layout()

    return fig_to_bytes(fig)


def plot_experience_vs_success(df_in: pd.DataFrame):
    """
    Generate a scatter plot showing founder experience vs success.

    Args:
        df_in (pd.DataFrame): DataFrame with 'Founders_Experience_Yrs' and 'Failed' columns

    Returns:
        bytes: PNG image data
    """
    figsize = get_mobile_optimized_figsize(10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    # Separate successful and failed startups
    successful = df_in[df_in['Failed'] == 0]
    failed = df_in[df_in['Failed'] == 1]

    # Scatter plot
    ax.scatter(successful['Founders_Experience_Yrs'], successful['Funding_USD_M'],
               c='green', s=150, alpha=0.6, label='Successful', marker='o')
    ax.scatter(failed['Founders_Experience_Yrs'], failed['Funding_USD_M'],
               c='red', s=150, alpha=0.6, label='Failed', marker='x')

    # Add labels for each point
    for _, row in df_in.iterrows():
        ax.annotate(row['Startup'],
                   (row['Founders_Experience_Yrs'], row['Funding_USD_M']),
                   fontsize=8, alpha=0.7)

    ax.set_xlabel('Founder Experience (Years)', fontsize=12)
    ax.set_ylabel('Funding (USD Millions)', fontsize=12)
    ax.set_title('Founder Experience vs Funding & Success', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig_to_bytes(fig)


def plot_custom_chart(df_in: pd.DataFrame, chart_type: str, x_col: str, y_col: str, title: str = None):
    """
    Generate a custom chart based on user specifications.

    Args:
        df_in (pd.DataFrame): DataFrame to visualize
        chart_type (str): Type of chart ('bar', 'scatter', 'line', 'pie')
        x_col (str): Column for x-axis
        y_col (str): Column for y-axis (not used for pie)
        title (str): Chart title

    Returns:
        bytes: PNG image data
    """
    figsize = get_mobile_optimized_figsize(10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    try:
        if chart_type == 'bar':
            ax.bar(df_in[x_col], df_in[y_col], color='steelblue', alpha=0.7)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.tick_params(axis='x', rotation=45)

        elif chart_type == 'scatter':
            ax.scatter(df_in[x_col], df_in[y_col], s=100, alpha=0.6, color='coral')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)

        elif chart_type == 'line':
            ax.plot(df_in[x_col], df_in[y_col], marker='o', linewidth=2, color='darkgreen')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.tick_params(axis='x', rotation=45)

        elif chart_type == 'pie':
            # For pie charts, x_col is used as labels, y_col as values
            ax.pie(df_in[y_col], labels=df_in[x_col], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{chart_type.capitalize()} Chart: {x_col} vs {y_col}",
                        fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig_to_bytes(fig)

    except Exception as e:
        # If error, return a simple error message chart
        ax.text(0.5, 0.5, f"Error generating chart:\n{str(e)}",
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig_to_bytes(fig)

# =============================
# UK ECONOMICS ANALYSIS MODULE
# =============================

class UKEconomicsAnalyzer:
    """UK-specific economic analysis for startups"""

    def __init__(self):
        self.uk_data = {
            'gdp_growth': 0.3,  # Q3 2024 estimate
            'inflation_cpi': 2.3,  # Current CPI
            'bank_rate': 4.75,  # Current Bank of England rate
            'unemployment': 4.2,  # Current unemployment rate
            'gbp_usd': 1.27,  # Current exchange rate
            'london_weight': 0.23,  # London's share of UK GDP
        }

    def analyze_macro_impact(self, startup_data: dict) -> dict:
        """Analyze macroeconomic impacts on startup"""

        impacts = {
            'interest_rate_impact': self._calculate_interest_impact(startup_data),
            'inflation_impact': self._calculate_inflation_impact(startup_data),
            'labour_market_impact': self._calculate_labour_impact(startup_data),
            'regional_factors': self._analyze_regional_factors(startup_data),
            'sector_outlook': self._analyze_sector_outlook(startup_data)
        }

        return impacts

    def _calculate_interest_impact(self, data: dict) -> dict:
        """Calculate how UK interest rates affect the startup"""

        funding = data.get('funding_usd_m', 5)
        debt_ratio = data.get('debt_ratio', 0.3)

        # Cost of capital impact
        base_rate = self.uk_data['bank_rate']
        risk_premium = 5.0  # Startup risk premium
        cost_of_debt = base_rate + risk_premium

        # Calculate impact
        annual_interest_cost = funding * debt_ratio * (cost_of_debt / 100)

        return {
            'cost_of_capital': cost_of_debt,
            'annual_interest_cost': annual_interest_cost,
            'impact_level': 'High' if cost_of_debt > 10 else 'Medium' if cost_of_debt > 7 else 'Low',
            'recommendation': self._get_interest_recommendation(cost_of_debt)
        }

    def _calculate_inflation_impact(self, data: dict) -> dict:
        """Calculate inflation impact on costs and pricing"""

        burn_rate = data.get('burn_rate_months', 100) * 1000  # Convert to pounds
        inflation = self.uk_data['inflation_cpi']

        # Real cost increase
        real_cost_increase = burn_rate * (inflation / 100) * 12  # Annual

        # Pricing power assessment
        b2b = data.get('is_b2b', True)
        pricing_power = 'Strong' if b2b else 'Moderate'

        return {
            'current_inflation': inflation,
            'real_cost_increase_annual': real_cost_increase,
            'pricing_power': pricing_power,
            'wage_pressure': 'High' if inflation > 3 else 'Moderate'
        }

    def _calculate_labour_impact(self, data: dict) -> dict:
        """Analyze UK labour market impact"""

        team_size = data.get('team_size', 10)
        location = data.get('location', 'London')

        # Regional wage differentials
        wage_multiplier = 1.3 if location == 'London' else 1.0

        # Skills shortage premium
        tech_premium = 1.2 if data.get('sector') == 'Tech' else 1.0

        # Calculate labour cost index
        labour_cost_index = 100 * wage_multiplier * tech_premium

        return {
            'unemployment_rate': self.uk_data['unemployment'],
            'labour_cost_index': labour_cost_index,
            'talent_availability': 'Tight' if self.uk_data['unemployment'] < 4 else 'Balanced',
            'wage_growth_pressure': 'High' if labour_cost_index > 120 else 'Moderate'
        }

    def _analyze_regional_factors(self, data: dict) -> dict:
        """Analyze UK regional economic factors"""

        location = data.get('location', 'London')

        regional_data = {
            'London': {'growth': 2.1, 'cost_index': 150, 'talent_pool': 'Deep'},
            'Manchester': {'growth': 1.8, 'cost_index': 85, 'talent_pool': 'Growing'},
            'Edinburgh': {'growth': 1.5, 'cost_index': 90, 'talent_pool': 'Specialized'},
            'Birmingham': {'growth': 1.3, 'cost_index': 80, 'talent_pool': 'Developing'},
            'Bristol': {'growth': 1.9, 'cost_index': 95, 'talent_pool': 'Tech-focused'},
            'Cambridge': {'growth': 2.3, 'cost_index': 110, 'talent_pool': 'Research-heavy'}
        }

        region_info = regional_data.get(location, regional_data['London'])

        return {
            'location': location,
            'regional_growth': region_info['growth'],
            'cost_index': region_info['cost_index'],
            'talent_pool': region_info['talent_pool'],
            'competitiveness': 'High' if region_info['cost_index'] < 100 else 'Challenging'
        }

    def _analyze_sector_outlook(self, data: dict) -> dict:
        """UK sector-specific analysis"""

        sector = data.get('sector', 'Tech')

        sector_outlooks = {
            'FinTech': {'growth': 4.5, 'regulation': 'High', 'opportunity': 'Strong'},
            'HealthTech': {'growth': 3.8, 'regulation': 'High', 'opportunity': 'NHS partnerships'},
            'GreenTech': {'growth': 6.2, 'regulation': 'Medium', 'opportunity': 'Net Zero targets'},
            'RetailTech': {'growth': 2.1, 'regulation': 'Low', 'opportunity': 'Digital transformation'},
            'EdTech': {'growth': 3.5, 'regulation': 'Medium', 'opportunity': 'Skills gap'},
            'PropTech': {'growth': 2.8, 'regulation': 'Medium', 'opportunity': 'Housing crisis'}
        }

        outlook = sector_outlooks.get(sector, {'growth': 2.5, 'regulation': 'Medium', 'opportunity': 'General'})

        return outlook

    def _get_interest_recommendation(self, cost: float) -> str:
        """Generate interest rate recommendations"""

        if cost > 12:
            return "Consider equity financing over debt given high interest costs"
        elif cost > 8:
            return "Lock in current rates if possible; consider revenue-based financing"
        else:
            return "Favorable borrowing environment; consider leveraging debt strategically"

def plot_uk_economic_indicators(df_in: pd.DataFrame):
    """Create UK economic indicators dashboard"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # UK GDP Growth vs Startup Funding
    ax1 = axes[0, 0]
    quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024', 'Q2 2024']
    gdp_growth = [0.1, 0.2, 0.0, -0.1, 0.6, 0.7]
    startup_funding = [1.2, 1.5, 1.1, 0.9, 1.3, 1.4]  # Billions

    ax1_twin = ax1.twinx()
    ax1.bar(quarters, gdp_growth, alpha=0.7, color='navy', label='GDP Growth %')
    ax1_twin.plot(quarters, startup_funding, 'ro-', label='Startup Funding (£B)')

    ax1.set_ylabel('GDP Growth (%)', color='navy')
    ax1_twin.set_ylabel('Startup Funding (£B)', color='red')
    ax1.set_title('UK GDP Growth vs Startup Funding', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Interest Rate Impact
    ax2 = axes[0, 1]
    rates = np.linspace(0, 10, 50)
    startup_viability = 100 - (rates ** 1.5) * 3

    ax2.plot(rates, startup_viability, linewidth=2, color='darkred')
    ax2.axvline(x=4.75, color='green', linestyle='--', label='Current Bank Rate')
    ax2.fill_between(rates, 0, startup_viability, alpha=0.3, color='lightblue')

    ax2.set_xlabel('Interest Rate (%)')
    ax2.set_ylabel('Startup Viability Score')
    ax2.set_title('Interest Rate Impact on Startups', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Regional Distribution
    ax3 = axes[1, 0]
    regions = ['London', 'South East', 'North West', 'Scotland', 'West Midlands', 'Other']
    startup_dist = [42, 18, 12, 8, 7, 13]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(regions)))

    wedges, texts, autotexts = ax3.pie(startup_dist, labels=regions, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
    ax3.set_title('UK Startup Distribution by Region', fontweight='bold')

    # Sector Performance
    ax4 = axes[1, 1]
    sectors = ['FinTech', 'HealthTech', 'GreenTech', 'EdTech', 'RetailTech']
    performance = [4.5, 3.8, 6.2, 3.5, 2.1]

    bars = ax4.barh(sectors, performance, color='teal')
    ax4.set_xlabel('Expected Growth Rate (%)')
    ax4.set_title('UK Sector Growth Outlook', fontweight='bold')

    for bar, value in zip(bars, performance):
        ax4.text(value + 0.1, bar.get_y() + bar.get_height()/2,
                f'{value}%', va='center')

    plt.suptitle('UK Economic Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig_to_bytes(fig)

def plot_profitability_analysis(analysis_data: dict):
    """Create profitability analysis charts"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Margin Waterfall
    ax1 = axes[0, 0]
    margins = ['Revenue', 'COGS', 'Gross Profit', 'OpEx', 'Operating Profit', 'Net Profit']
    values = [100, -40, 60, -35, 25, 18]
    colors = ['green', 'red', 'green', 'red', 'green', 'darkgreen']

    ax1.bar(margins, values, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Profitability Waterfall', fontweight='bold')
    ax1.set_ylabel('% of Revenue')
    ax1.tick_params(axis='x', rotation=45)

    # Unit Economics
    ax2 = axes[0, 1]
    ltv_cac = analysis_data.get('ltv_cac_ratio', 3.0)
    benchmark = 3.0

    bars = ax2.bar(['LTV/CAC Ratio', 'Benchmark'], [ltv_cac, benchmark],
                   color=['green' if ltv_cac > benchmark else 'red', 'gray'])
    ax2.axhline(y=3, color='blue', linestyle='--', alpha=0.5, label='Healthy Threshold')
    ax2.set_title('Unit Economics Health', fontweight='bold')
    ax2.set_ylabel('Ratio')
    ax2.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')

    # Break-even Analysis
    ax3 = axes[1, 0]
    units = np.linspace(0, 2000, 100)
    fixed_costs = 50000
    variable_cost = 30
    price = 100

    revenue_line = units * price
    total_cost_line = fixed_costs + (units * variable_cost)

    ax3.plot(units, revenue_line, 'g-', label='Revenue', linewidth=2)
    ax3.plot(units, total_cost_line, 'r-', label='Total Cost', linewidth=2)
    ax3.fill_between(units, revenue_line, total_cost_line,
                     where=(revenue_line > total_cost_line), alpha=0.3, color='green', label='Profit Zone')
    ax3.fill_between(units, revenue_line, total_cost_line,
                     where=(revenue_line <= total_cost_line), alpha=0.3, color='red', label='Loss Zone')

    # Mark break-even point
    break_even_units = fixed_costs / (price - variable_cost)
    ax3.plot(break_even_units, break_even_units * price, 'ko', markersize=8)
    ax3.annotate(f'Break-even: {break_even_units:.0f} units',
                xy=(break_even_units, break_even_units * price),
                xytext=(break_even_units + 200, break_even_units * price),
                arrowprops=dict(arrowstyle='->'))

    ax3.set_xlabel('Units Sold')
    ax3.set_ylabel('Revenue/Cost ($)')
    ax3.set_title('Break-even Analysis', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Cash Runway
    ax4 = axes[1, 1]
    months = np.arange(0, 25)
    cash_balance = 500000
    monthly_burn = 50000

    cash_projection = [cash_balance - (monthly_burn * m) for m in months]
    cash_projection = [max(0, c) for c in cash_projection]  # Can't go below 0

    ax4.fill_between(months, 0, cash_projection, alpha=0.3, color='blue')
    ax4.plot(months, cash_projection, 'b-', linewidth=2)
    ax4.axhline(y=100000, color='orange', linestyle='--', label='Danger Zone')
    ax4.axhline(y=0, color='red', linestyle='--', label='Out of Cash')

    # Mark runway
    runway = cash_balance / monthly_burn
    ax4.axvline(x=runway, color='green', linestyle='--', alpha=0.7, label=f'Runway: {runway:.0f} months')

    ax4.set_xlabel('Months')
    ax4.set_ylabel('Cash Balance ($)')
    ax4.set_title('Cash Runway Projection', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Company Financial Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig_to_bytes(fig)

def plot_margin_trends(historical_data: list):
    """Plot historical margin trends"""

    fig, ax = plt.subplots(figsize=(12, 6))

    quarters = [d['quarter'] for d in historical_data]
    gross_margins = [d['gross_margin'] for d in historical_data]
    operating_margins = [d['operating_margin'] for d in historical_data]
    net_margins = [d['net_margin'] for d in historical_data]

    ax.plot(quarters, gross_margins, 'g-', marker='o', linewidth=2, label='Gross Margin')
    ax.plot(quarters, operating_margins, 'b-', marker='s', linewidth=2, label='Operating Margin')
    ax.plot(quarters, net_margins, 'r-', marker='^', linewidth=2, label='Net Margin')

    ax.fill_between(range(len(quarters)), gross_margins, alpha=0.1, color='green')
    ax.fill_between(range(len(quarters)), operating_margins, alpha=0.1, color='blue')
    ax.fill_between(range(len(quarters)), net_margins, alpha=0.1, color='red')

    ax.set_xlabel('Quarter')
    ax.set_ylabel('Margin (%)')
    ax.set_title('Margin Trends Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add trend lines
    z_gross = np.polyfit(range(len(quarters)), gross_margins, 1)
    p_gross = np.poly1d(z_gross)
    ax.plot(range(len(quarters)), p_gross(range(len(quarters))), 'g--', alpha=0.5)

    plt.tight_layout()
    return fig_to_bytes(fig)

def plot_cash_flow_waterfall(cash_data: dict):
    """Create cash flow waterfall chart"""

    fig, ax = plt.subplots(figsize=(12, 6))

    categories = ['Starting Cash', 'Operations', 'Investing', 'Financing', 'Ending Cash']
    values = [
        cash_data.get('starting_cash', 500000),
        cash_data.get('cash_from_operations', -200000),
        cash_data.get('cash_from_investing', -50000),
        cash_data.get('cash_from_financing', 300000),
        0  # Will calculate
    ]

    # Calculate ending cash
    values[4] = sum(values[:4])

    # Create cumulative values for positioning
    cumulative = [values[0]]
    for i in range(1, len(values)-1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(values[4])

    # Plot bars
    colors = ['blue', 'red' if values[1] < 0 else 'green',
              'red' if values[2] < 0 else 'green',
              'green' if values[3] > 0 else 'red', 'blue']

    for i, (cat, val, cum) in enumerate(zip(categories, values, cumulative)):
        if i == 0 or i == len(categories) - 1:
            # Starting and ending cash - full bars
            ax.bar(cat, val, color=colors[i], alpha=0.7)
        else:
            # Flow bars - positioned relative to cumulative
            bottom = cum - val if val > 0 else cum
            height = abs(val)
            ax.bar(cat, height, bottom=bottom, color=colors[i], alpha=0.7)

        # Add value labels
        ax.text(i, cum + 10000, f'${val:,.0f}', ha='center', va='bottom')

    ax.set_title('Cash Flow Waterfall Analysis', fontweight='bold')
    ax.set_ylabel('Cash ($)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_bytes(fig)

def plot_break_even_chart(financials: dict):
    """Create detailed break-even analysis chart"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Break-even line chart
    units = np.linspace(0, 2000, 100)
    fixed_costs = financials.get('fixed_costs', 100000)
    variable_cost_per_unit = financials.get('variable_cost_per_unit', 40)
    price_per_unit = financials.get('price_per_unit', 100)

    revenue = units * price_per_unit
    total_costs = fixed_costs + (units * variable_cost_per_unit)
    profit = revenue - total_costs

    ax1.plot(units, revenue, 'g-', linewidth=2, label='Revenue')
    ax1.plot(units, total_costs, 'r-', linewidth=2, label='Total Costs')
    ax1.fill_between(units, revenue, total_costs, where=(revenue > total_costs),
                     alpha=0.3, color='green', label='Profit Zone')
    ax1.fill_between(units, revenue, total_costs, where=(revenue <= total_costs),
                     alpha=0.3, color='red', label='Loss Zone')

    # Mark break-even point
    break_even_units = fixed_costs / (price_per_unit - variable_cost_per_unit)
    break_even_revenue = break_even_units * price_per_unit

    ax1.plot(break_even_units, break_even_revenue, 'ko', markersize=8)
    ax1.annotate(f'Break-even\n{break_even_units:.0f} units\n${break_even_revenue:,.0f}',
                xy=(break_even_units, break_even_revenue),
                xytext=(break_even_units + 300, break_even_revenue),
                arrowprops=dict(arrowstyle='->'))

    ax1.set_xlabel('Units Sold')
    ax1.set_ylabel('Amount ($)')
    ax1.set_title('Break-even Analysis', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Sensitivity analysis
    scenarios = ['Conservative', 'Base Case', 'Optimistic']
    price_variations = [price_per_unit * 0.9, price_per_unit, price_per_unit * 1.1]
    break_even_scenarios = [fixed_costs / (p - variable_cost_per_unit) for p in price_variations]

    bars = ax2.bar(scenarios, break_even_scenarios, color=['red', 'orange', 'green'], alpha=0.7)
    ax2.set_ylabel('Break-even Units')
    ax2.set_title('Break-even Sensitivity Analysis', fontweight='bold')

    # Add value labels
    for bar, value in zip(bars, break_even_scenarios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{value:.0f}', ha='center', va='bottom')

    plt.tight_layout()
    return fig_to_bytes(fig)

# =============================
# COMPANY ANALYSIS MODULE
# =============================

class CompanyAnalyzer:
    """Comprehensive company financial analysis"""

    def __init__(self):
        self.industry_benchmarks = {
            'SaaS': {'gross_margin': 75, 'operating_margin': 20, 'ltv_cac': 3.0},
            'E-commerce': {'gross_margin': 40, 'operating_margin': 10, 'ltv_cac': 2.5},
            'Marketplace': {'gross_margin': 60, 'operating_margin': 15, 'ltv_cac': 4.0},
            'Hardware': {'gross_margin': 35, 'operating_margin': 8, 'ltv_cac': 2.0},
            'Services': {'gross_margin': 50, 'operating_margin': 12, 'ltv_cac': 2.8},
            'FinTech': {'gross_margin': 65, 'operating_margin': 18, 'ltv_cac': 3.5}
        }

    def analyze_profitability(self, financials: dict) -> dict:
        """Complete profitability analysis"""

        # Extract key metrics
        revenue = financials.get('revenue', 0)
        cogs = financials.get('cogs', 0)
        opex = financials.get('opex', 0)
        sales_marketing = financials.get('sales_marketing', 0)
        rd_expense = financials.get('rd_expense', 0)
        admin_expense = financials.get('admin_expense', 0)

        # Calculate margins
        gross_profit = revenue - cogs
        gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0

        operating_profit = gross_profit - opex
        operating_margin = (operating_profit / revenue * 100) if revenue > 0 else 0

        ebitda = operating_profit + financials.get('depreciation', 0)
        ebitda_margin = (ebitda / revenue * 100) if revenue > 0 else 0

        net_profit = operating_profit - financials.get('interest', 0) - financials.get('tax', 0)
        net_margin = (net_profit / revenue * 100) if revenue > 0 else 0

        return {
            'gross_profit': gross_profit,
            'gross_margin': gross_margin,
            'operating_profit': operating_profit,
            'operating_margin': operating_margin,
            'ebitda': ebitda,
            'ebitda_margin': ebitda_margin,
            'net_profit': net_profit,
            'net_margin': net_margin,
            'profit_health': self._assess_profit_health(gross_margin, operating_margin, net_margin)
        }

    def analyze_unit_economics(self, metrics: dict) -> dict:
        """Analyze unit-level profitability"""

        # Customer economics
        cac = metrics.get('customer_acquisition_cost', 100)
        ltv = metrics.get('lifetime_value', 300)
        ltv_cac_ratio = ltv / cac if cac > 0 else 0

        # Unit contribution
        revenue_per_unit = metrics.get('revenue_per_unit', 50)
        variable_cost_per_unit = metrics.get('variable_cost_per_unit', 20)
        contribution_margin = revenue_per_unit - variable_cost_per_unit
        contribution_margin_pct = (contribution_margin / revenue_per_unit * 100) if revenue_per_unit > 0 else 0

        # Payback period
        monthly_revenue_per_customer = metrics.get('monthly_revenue', 100)
        payback_months = cac / monthly_revenue_per_customer if monthly_revenue_per_customer > 0 else 999

        return {
            'ltv': ltv,
            'cac': cac,
            'ltv_cac_ratio': ltv_cac_ratio,
            'contribution_margin': contribution_margin,
            'contribution_margin_pct': contribution_margin_pct,
            'payback_months': payback_months,
            'unit_economics_health': 'Strong' if ltv_cac_ratio > 3 else 'Moderate' if ltv_cac_ratio > 1 else 'Weak'
        }

    def analyze_cash_flow(self, cash_data: dict) -> dict:
        """Analyze cash flow and runway"""

        # Operating cash flow
        cash_from_operations = cash_data.get('cash_from_operations', -50000)
        cash_from_investing = cash_data.get('cash_from_investing', -20000)
        cash_from_financing = cash_data.get('cash_from_financing', 100000)

        # Net cash flow
        net_cash_flow = cash_from_operations + cash_from_investing + cash_from_financing

        # Burn rate and runway
        monthly_burn = -cash_from_operations / 12 if cash_from_operations < 0 else 0
        cash_balance = cash_data.get('cash_balance', 500000)
        runway_months = cash_balance / monthly_burn if monthly_burn > 0 else 999

        # Cash conversion cycle
        dso = cash_data.get('days_sales_outstanding', 45)
        dio = cash_data.get('days_inventory_outstanding', 30)
        dpo = cash_data.get('days_payables_outstanding', 30)
        cash_conversion_cycle = dso + dio - dpo

        return {
            'operating_cash_flow': cash_from_operations,
            'net_cash_flow': net_cash_flow,
            'monthly_burn': monthly_burn,
            'runway_months': runway_months,
            'cash_conversion_cycle': cash_conversion_cycle,
            'cash_efficiency': 'Efficient' if cash_conversion_cycle < 30 else 'Moderate' if cash_conversion_cycle < 60 else 'Inefficient'
        }

    def calculate_break_even(self, financials: dict) -> dict:
        """Calculate break-even analysis"""

        fixed_costs = financials.get('fixed_costs', 100000)
        variable_cost_ratio = financials.get('variable_cost_ratio', 0.4)
        price_per_unit = financials.get('price_per_unit', 100)
        variable_cost_per_unit = price_per_unit * variable_cost_ratio

        # Break-even units
        contribution_per_unit = price_per_unit - variable_cost_per_unit
        break_even_units = fixed_costs / contribution_per_unit if contribution_per_unit > 0 else 999999

        # Break-even revenue
        break_even_revenue = break_even_units * price_per_unit

        # Margin of safety
        current_revenue = financials.get('current_revenue', 150000)
        margin_of_safety = ((current_revenue - break_even_revenue) / current_revenue * 100) if current_revenue > 0 else -100

        return {
            'break_even_units': break_even_units,
            'break_even_revenue': break_even_revenue,
            'contribution_per_unit': contribution_per_unit,
            'margin_of_safety': margin_of_safety,
            'months_to_break_even': self._calculate_months_to_break_even(financials)
        }

    def benchmark_performance(self, company_metrics: dict, industry: str) -> dict:
        """Compare against industry benchmarks"""

        benchmarks = self.industry_benchmarks.get(
            industry,
            self.industry_benchmarks['Services']  # Default
        )

        comparisons = {
            'gross_margin': {
                'company': company_metrics.get('gross_margin', 0),
                'industry': benchmarks['gross_margin'],
                'delta': company_metrics.get('gross_margin', 0) - benchmarks['gross_margin'],
                'performance': 'Above' if company_metrics.get('gross_margin', 0) > benchmarks['gross_margin'] else 'Below'
            },
            'operating_margin': {
                'company': company_metrics.get('operating_margin', 0),
                'industry': benchmarks['operating_margin'],
                'delta': company_metrics.get('operating_margin', 0) - benchmarks['operating_margin'],
                'performance': 'Above' if company_metrics.get('operating_margin', 0) > benchmarks['operating_margin'] else 'Below'
            },
            'ltv_cac': {
                'company': company_metrics.get('ltv_cac_ratio', 0),
                'industry': benchmarks['ltv_cac'],
                'delta': company_metrics.get('ltv_cac_ratio', 0) - benchmarks['ltv_cac'],
                'performance': 'Above' if company_metrics.get('ltv_cac_ratio', 0) > benchmarks['ltv_cac'] else 'Below'
            }
        }

        # Overall rating
        above_count = sum(1 for metric in comparisons.values() if metric['performance'] == 'Above')
        overall_rating = 'Outperforming' if above_count >= 2 else 'Underperforming'

        return {
            'comparisons': comparisons,
            'overall_rating': overall_rating,
            'recommendations': self._generate_recommendations(comparisons)
        }

    def _assess_profit_health(self, gross, operating, net):
        """Assess overall profitability health"""
        if net > 10 and operating > 15 and gross > 50:
            return "Excellent"
        elif net > 0 and operating > 5 and gross > 30:
            return "Good"
        elif net > -10 and gross > 20:
            return "Moderate"
        else:
            return "Poor"

    def _calculate_months_to_break_even(self, financials):
        """Calculate months to reach break-even"""
        current_loss = financials.get('monthly_loss', 50000)
        growth_rate = financials.get('growth_rate', 0.1)

        if current_loss <= 0:  # Already profitable
            return 0

        months = 0
        while current_loss > 0 and months < 60:  # Max 60 months
            current_loss *= (1 - growth_rate)
            months += 1

        return months if months < 60 else 999

    def _generate_recommendations(self, comparisons):
        """Generate improvement recommendations"""
        recommendations = []

        if comparisons['gross_margin']['delta'] < 0:
            recommendations.append("Focus on pricing optimization or reducing COGS")
        if comparisons['operating_margin']['delta'] < 0:
            recommendations.append("Improve operational efficiency and reduce overhead")
        if comparisons['ltv_cac']['delta'] < 0:
            recommendations.append("Optimize customer acquisition channels or increase retention")

        return recommendations

# =============================
# INTERACTIVE DASHBOARD MODULE
# =============================

class InteractiveDashboard:
    """Interactive dashboard with real-time data visualization"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.filtered_df = df.copy()
        self.selected_startups = []

    def create_executive_summary(self) -> Dict[str, Any]:
        """Create executive summary metrics"""
        total_startups = len(self.filtered_df)
        total_funding = self.filtered_df['Funding_USD_M'].sum()
        success_rate = ((self.filtered_df['Failed'] == 0).sum() / total_startups * 100) if total_startups > 0 else 0
        avg_runway = self.filtered_df['Burn_Rate_Months'].mean()

        return {
            'total_startups': total_startups,
            'total_funding': total_funding,
            'success_rate': success_rate,
            'avg_runway': avg_runway,
            'top_sector': self.filtered_df['Sector'].mode().iloc[0] if len(self.filtered_df) > 0 else 'N/A',
            'risk_level': 'High' if success_rate < 40 else 'Medium' if success_rate < 70 else 'Low'
        }

    def create_interactive_scatter(self, x_col: str = 'Funding_USD_M', y_col: str = 'Burn_Rate_Months') -> bytes:
        """Create interactive scatter plot with hover details and filtering"""

        fig = go.Figure()

        # Split data by success/failure for different colors
        success_df = self.filtered_df[self.filtered_df['Failed'] == 0]
        failed_df = self.filtered_df[self.filtered_df['Failed'] == 1]

        # Add successful startups
        if len(success_df) > 0:
            fig.add_trace(go.Scatter(
                x=success_df[x_col],
                y=success_df[y_col],
                mode='markers',
                name='Successful',
                marker=dict(
                    color='green',
                    size=success_df['Market_Size_Bn'] * 2,  # Size by market size
                    opacity=0.7,
                    line=dict(width=1, color='darkgreen')
                ),
                text=success_df['Startup'],
                hovertemplate=
                '<b>%{text}</b><br>' +
                f'{x_col}: %{{x}}<br>' +
                f'{y_col}: %{{y}}<br>' +
                'Sector: %{customdata[0]}<br>' +
                'Market Size: $%{customdata[1]}B<br>' +
                'Experience: %{customdata[2]} years<br>' +
                '<extra></extra>',
                customdata=success_df[['Sector', 'Market_Size_Bn', 'Founders_Experience_Yrs']].values,
                selected=dict(marker=dict(color='gold', size=20))
            ))

        # Add failed startups
        if len(failed_df) > 0:
            fig.add_trace(go.Scatter(
                x=failed_df[x_col],
                y=failed_df[y_col],
                mode='markers',
                name='Failed',
                marker=dict(
                    color='red',
                    size=failed_df['Market_Size_Bn'] * 2,
                    opacity=0.7,
                    line=dict(width=1, color='darkred')
                ),
                text=failed_df['Startup'],
                hovertemplate=
                '<b>%{text}</b><br>' +
                f'{x_col}: %{{x}}<br>' +
                f'{y_col}: %{{y}}<br>' +
                'Sector: %{customdata[0]}<br>' +
                'Market Size: $%{customdata[1]}B<br>' +
                'Experience: %{customdata[2]} years<br>' +
                '<extra></extra>',
                customdata=failed_df[['Sector', 'Market_Size_Bn', 'Founders_Experience_Yrs']].values,
                selected=dict(marker=dict(color='orange', size=20))
            ))

        # Update layout for interactivity
        fig.update_layout(
            title=f'Interactive Analysis: {x_col} vs {y_col}',
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            hovermode='closest',
            clickmode='event+select',
            showlegend=True,
            height=600,
            template='plotly_white',
            annotations=[
                dict(
                    text="💡 Click points to select • Drag to zoom • Double-click to reset",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=1.02, xanchor='center', yanchor='bottom',
                    font=dict(size=12, color="gray")
                )
            ]
        )

        return pio.to_image(fig, format='png')

    def create_multi_dimensional_heatmap(self) -> bytes:
        """Create correlation heatmap with interactive features"""

        # Select numeric columns for correlation
        numeric_cols = ['Funding_USD_M', 'Burn_Rate_Months', 'Founders_Experience_Yrs',
                       'Market_Size_Bn', 'Business_Model_Strength', 'Moat_Defensibility',
                       'MRR_K', 'Monthly_Growth_Rate', 'Competition_Intensity']

        # Calculate correlation matrix
        corr_matrix = self.filtered_df[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[col.replace('_', ' ').title() for col in corr_matrix.columns],
            y=[col.replace('_', ' ').title() for col in corr_matrix.index],
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title='📊 Interactive Correlation Heatmap',
            height=600,
            template='plotly_white'
        )

        return pio.to_image(fig, format='png')

    def create_real_time_metrics_dashboard(self) -> bytes:
        """Create real-time style metrics dashboard"""

        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['💰 Funding Distribution', '📈 Success Rate by Sector',
                          '⏱️ Runway Analysis', '🌍 Geographic Distribution'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "pie"}]]
        )

        # 1. Funding distribution
        funding_bins = pd.cut(self.filtered_df['Funding_USD_M'], bins=5)
        funding_dist = funding_bins.value_counts().sort_index()

        fig.add_trace(
            go.Bar(x=[str(interval) for interval in funding_dist.index],
                   y=funding_dist.values,
                   name="Funding",
                   marker_color='lightblue'),
            row=1, col=1
        )

        # 2. Success rate by sector
        sector_success = self.filtered_df.groupby('Sector')['Failed'].agg(['count', 'sum'])
        sector_success['success_rate'] = (1 - sector_success['sum'] / sector_success['count']) * 100

        fig.add_trace(
            go.Bar(x=sector_success.index,
                   y=sector_success['success_rate'],
                   name="Success Rate",
                   marker_color='lightgreen'),
            row=1, col=2
        )

        # 3. Runway distribution
        fig.add_trace(
            go.Histogram(x=self.filtered_df['Burn_Rate_Months'],
                        name="Runway",
                        marker_color='orange',
                        opacity=0.7),
            row=2, col=1
        )

        # 4. Geographic distribution
        country_dist = self.filtered_df['Country'].value_counts()

        fig.add_trace(
            go.Pie(labels=country_dist.index,
                   values=country_dist.values,
                   name="Geography"),
            row=2, col=2
        )

        fig.update_layout(
            title_text="📊 Real-Time Dashboard Metrics",
            height=800,
            showlegend=False,
            template='plotly_white'
        )

        return pio.to_image(fig, format='png')

    def filter_data(self, filters: Dict[str, Any]) -> None:
        """Apply filters to the dataset"""
        self.filtered_df = self.df.copy()

        if 'sectors' in filters and filters['sectors']:
            self.filtered_df = self.filtered_df[self.filtered_df['Sector'].isin(filters['sectors'])]

        if 'countries' in filters and filters['countries']:
            self.filtered_df = self.filtered_df[self.filtered_df['Country'].isin(filters['countries'])]

        if 'funding_range' in filters:
            min_funding, max_funding = filters['funding_range']
            self.filtered_df = self.filtered_df[
                (self.filtered_df['Funding_USD_M'] >= min_funding) &
                (self.filtered_df['Funding_USD_M'] <= max_funding)
            ]

        if 'success_only' in filters and filters['success_only']:
            self.filtered_df = self.filtered_df[self.filtered_df['Failed'] == 0]

    def compare_startups(self, startup_names: List[str]) -> bytes:
        """Create comparison chart for selected startups"""

        comparison_df = self.df[self.df['Startup'].isin(startup_names)]

        if len(comparison_df) == 0:
            return None

        # Create radar chart for comparison
        categories = ['Funding (Scaled)', 'Experience', 'Market Size',
                     'Business Model', 'Moat', 'MRR (Scaled)', 'Growth Rate']

        fig = go.Figure()

        for _, startup in comparison_df.iterrows():
            values = [
                startup['Funding_USD_M'] / 20,  # Scale to 0-5
                startup['Founders_Experience_Yrs'],
                startup['Market_Size_Bn'],
                startup['Business_Model_Strength'],
                startup['Moat_Defensibility'],
                startup['MRR_K'] / 200,  # Scale to 0-5
                startup['Monthly_Growth_Rate'] / 10  # Scale to 0-5
            ]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=startup['Startup'],
                line_color='red' if startup['Failed'] == 1 else 'green'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )),
            showlegend=True,
            title="🔍 Startup Comparison Analysis",
            height=600
        )

        return pio.to_image(fig, format='png')

def create_dashboard_summary(dashboard: InteractiveDashboard) -> str:
    """Create text summary of dashboard metrics"""

    summary = dashboard.create_executive_summary()

    return f"""
## 📊 Dashboard Executive Summary

### 🎯 **Key Metrics**
- **Total Startups:** {summary['total_startups']}
- **Total Funding:** ${summary['total_funding']:.1f}M
- **Success Rate:** {summary['success_rate']:.1f}%
- **Avg Runway:** {summary['avg_runway']:.1f} months

### 📈 **Risk Assessment**
- **Overall Risk Level:** {summary['risk_level']}
- **Top Sector:** {summary['top_sector']}
- **Recommendation:** {'Focus on due diligence' if summary['risk_level'] == 'High' else 'Balanced portfolio approach' if summary['risk_level'] == 'Medium' else 'Strong investment opportunities'}

### 🔍 **Interactive Features Available:**
- Click charts to drill down
- Filter by sector, country, funding range
- Compare multiple startups
- Real-time metric updates
"""

# =============================
# VIABILITY SCORING MODEL
# =============================

def viability_score(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive viability score for a startup based on multiple factors.

    This heuristic model evaluates startup viability across 8 dimensions and combines
    them into a single score (0-100). Higher scores indicate better viability.

    Scoring Methodology:
    --------------------
    1. Each dimension is normalized to 0-1 scale
    2. Dimensions are weighted based on importance
    3. Weighted scores are summed and scaled to 0-100
    4. Additional metrics (runway, failure year) are calculated
    5. Rule-based tips are generated based on weak areas

    Input Features:
    ---------------
    Args:
        features (Dict[str, Any]): Dictionary containing:
            - funding_usd_m (float): Funding in millions USD
            - burn_rate_months (float): Burn rate in months
            - team_experience_years (float): Average team experience in years
            - market_size_bn (float): Market size in billions USD
            - business_model_strength_1_5 (int): 1-5 scale of business model quality
            - moat_1_5 (int): 1-5 scale of competitive moat/defensibility
            - traction_mrr_k (float): Monthly recurring revenue in thousands USD
            - growth_rate_pct (float): Monthly growth rate as percentage
            - competition_intensity_1_5 (int): 1-5 scale of competition (higher = more intense)

    Returns:
        Dict[str, Any]: Results containing:
            - score (float): Overall viability score (0-100)
            - survival_months (float): Estimated months until failure
            - est_failure_year (float): Projected year of failure
            - components (dict): Individual component scores (0-1)
            - tips (list): Actionable recommendations

    Scoring Components & Weights:
    -----------------------------
    1. Runway (18%): How long funding lasts - capped at 2 years = 1.0
    2. Experience (14%): Team experience - 10 years = 1.0
    3. Market (14%): Market size - $100B = 1.0
    4. Business Model (12%): Strength rating mapped to 0-1
    5. Moat (10%): Defensibility rating mapped to 0-1
    6. Traction (14%): MRR - $100k = 1.0
    7. Growth (12%): Monthly growth - 25% = 1.0
    8. Competition (6%): Inverse of competition intensity

    Example:
    --------
    >>> features = {
    ...     'funding_usd_m': 5.0,
    ...     'burn_rate_months': 12,
    ...     'team_experience_years': 5,
    ...     'market_size_bn': 50,
    ...     'business_model_strength_1_5': 3,
    ...     'moat_1_5': 3,
    ...     'traction_mrr_k': 25,
    ...     'growth_rate_pct': 10,
    ...     'competition_intensity_1_5': 3
    ... }
    >>> result = viability_score(features)
    >>> print(result['score'])  # Overall score out of 100
    """
    f = features  # Shorthand for cleaner code

    # -------------------------
    # 1. RUNWAY SCORE (0-1)
    # -------------------------
    # Calculate how many years the funding will last
    # Division approximation: funding (M) / burn_rate (months) ≈ years
    # Max protects against division by zero
    runway_years = (f["funding_usd_m"] / max(f["burn_rate_months"], 1))

    # Normalize to 0-1 scale, capping at 2 years
    # 2+ years of runway = perfect score (1.0)
    # 0 years = worst score (0.0)
    runway_score = max(0, min(1, runway_years / 2.0))

    # -------------------------
    # 2. EXPERIENCE SCORE (0-1)
    # -------------------------
    # Team experience normalized to 0-1, capped at 10 years
    # 10+ years = 1.0, 0 years = 0.0
    exp_score = max(0, min(1, f["team_experience_years"] / 10))

    # -------------------------
    # 3. MARKET SIZE SCORE (0-1)
    # -------------------------
    # Market opportunity normalized to 0-1, capped at $100B
    # $100B+ market = 1.0, $0 market = 0.0
    market_score = max(0, min(1, f["market_size_bn"] / 100))

    # -------------------------
    # 4. BUSINESS MODEL SCORE (0-1)
    # -------------------------
    # Convert 1-5 scale to 0-1 by subtracting 1 and dividing by 4
    # Rating 5 → 4/4 = 1.0, Rating 1 → 0/4 = 0.0
    bm_score = (f["business_model_strength_1_5"] - 1) / 4

    # -------------------------
    # 5. MOAT/DEFENSIBILITY SCORE (0-1)
    # -------------------------
    # Same conversion as business model: 1-5 scale → 0-1 scale
    moat_score = (f["moat_1_5"] - 1) / 4

    # -------------------------
    # 6. TRACTION SCORE (0-1)
    # -------------------------
    # MRR normalized to 0-1, capped at $100k
    # $100k+ MRR = 1.0, $0 MRR = 0.0
    traction_score = max(0, min(1, (f["traction_mrr_k"] / 100)))

    # -------------------------
    # 7. GROWTH SCORE (0-1)
    # -------------------------
    # Monthly growth rate normalized to 0-1, capped at 25%
    # 25%+ monthly growth = 1.0, 0% growth = 0.0
    growth_score = max(0, min(1, f["growth_rate_pct"] / 25))

    # -------------------------
    # 8. COMPETITION SCORE (0-1)
    # -------------------------
    # Competition is inverted: higher intensity = worse for startup
    # Convert 1-5 scale to 0-1 penalty, then invert
    # Low competition (1) → penalty 0 → score 1.0
    # High competition (5) → penalty 1 → score 0.0
    competition_penalty = (f["competition_intensity_1_5"] - 1) / 4
    competition_score = 1 - competition_penalty

    # -------------------------
    # WEIGHTED COMPOSITE SCORE
    # -------------------------
    # Define weights for each component (must sum to 1.0)
    # These weights reflect the relative importance of each factor
    weights = {
        "runway": 0.18,        # 18% - Most immediate concern
        "experience": 0.14,    # 14% - Critical for execution
        "market": 0.14,        # 14% - Ceiling for growth
        "bm": 0.12,           # 12% - Revenue sustainability
        "moat": 0.10,         # 10% - Long-term defensibility
        "traction": 0.14,     # 14% - Proof of product-market fit
        "growth": 0.12,       # 12% - Momentum indicator
        "competition": 0.06   # 6% - External threat level
    }

    # Calculate weighted sum of all components
    composite = (
        runway_score * weights["runway"] +
        exp_score * weights["experience"] +
        market_score * weights["market"] +
        bm_score * weights["bm"] +
        moat_score * weights["moat"] +
        traction_score * weights["traction"] +
        growth_score * weights["growth"] +
        competition_score * weights["competition"]
    )

    # Scale composite score (0-1) to percentage (0-100)
    score_100 = composite * 100.0

    # -------------------------
    # SURVIVAL METRICS
    # -------------------------
    # Calculate estimated months until money runs out
    # Formula: funding (M) × 12 months/year ÷ burn_rate
    # Max protects against division by very small numbers
    survival_months = max(1, f["funding_usd_m"] * (12 / max(f["burn_rate_months"], 0.1)))

    # Calculate projected failure year
    # Start from FUNDING_YEAR and add runway in years
    est_failure_year = FUNDING_YEAR + (f["funding_usd_m"] / max(f["burn_rate_months"], 0.1))

    # -------------------------
    # RULE-BASED RECOMMENDATIONS
    # -------------------------
    # Generate actionable tips based on weak areas
    # Each condition checks if a metric falls below a threshold
    tips = []

    # Runway too short (< 9 months)
    if runway_years < 0.75:
        tips.append("Increase runway (more funding or lower burn).")

    # Team lacks experience (< 3 years average)
    if f["team_experience_years"] < 3:
        tips.append("Augment team with experienced operators.")

    # Market too small (< $10B)
    if f["market_size_bn"] < 10:
        tips.append("Target a larger wedge or adjacent segments.")

    # Low traction (< $20k MRR)
    if f["traction_mrr_k"] < 20:
        tips.append("Focus on early, repeatable revenue (>$20k MRR).")

    # Slow growth (< 8% monthly)
    if f["growth_rate_pct"] < 8:
        tips.append("Drive growth via channels with clear CAC/LTV.")

    # Weak moat (rating <= 2)
    if f["moat_1_5"] <= 2:
        tips.append("Strengthen defensibility (IP, data, network effects).")

    # Weak business model (rating <= 2)
    if f["business_model_strength_1_5"] <= 2:
        tips.append("Clarify pricing & unit economics.")

    # High competition (rating >= 4)
    if f["competition_intensity_1_5"] >= 4:
        tips.append("Differentiate positioning vs strong incumbents.")

    # -------------------------
    # RETURN RESULTS
    # -------------------------
    # Package all results into a dictionary for easy access
    return {
        "score": score_100,  # Overall viability score (0-100)
        "survival_months": survival_months,  # Months until out of money
        "est_failure_year": est_failure_year,  # Projected year of failure
        "components": {  # Individual component scores for transparency
            "runway": runway_score,
            "experience": exp_score,
            "market": market_score,
            "business_model": bm_score,
            "moat": moat_score,
            "traction": traction_score,
            "growth": growth_score,
            "competition": competition_score
        },
        "tips": tips  # Actionable recommendations
    }


def generate_investment_report(df_in: pd.DataFrame, startup_name: str = None) -> bytes:
    """
    Generate a comprehensive PDF investment analysis report.

    Args:
        df_in (pd.DataFrame): Startup dataset
        startup_name (str, optional): Specific startup to analyze. If None, analyzes entire dataset.

    Returns:
        bytes: PDF file as bytes
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from datetime import datetime

    # Create in-memory buffer for PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)

    # Container for PDF elements
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#ff4b4b'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#ff4b4b'),
        spaceAfter=12,
        spaceBefore=12
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=6,
        spaceBefore=6
    )

    # =============================
    # PAGE 1: TITLE & EXECUTIVE SUMMARY
    # =============================

    if startup_name:
        startup_data = df_in[df_in['Startup'] == startup_name].iloc[0]
        report_title = f"Investment Analysis: {startup_name}"
    else:
        report_title = "Startup Portfolio Analysis Report"

    story.append(Paragraph(report_title, title_style))
    story.append(Paragraph(f"Generated by NAVADA | {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))

    if startup_name:
        # Single startup analysis
        funding = startup_data['Funding_USD_M']
        burn = startup_data['Burn_Rate_Months']
        sector = startup_data['Sector']
        country = startup_data['Country']
        failed = startup_data['Failed']
        experience = startup_data['Founders_Experience_Yrs']
        market = startup_data['Market_Size_Bn']

        status = "Failed" if failed == 1 else "Active/Successful"
        runway_months = (funding / burn) * 12 if burn > 0 else 0

        summary_text = f"""
        <b>{startup_name}</b> is a {sector} startup based in {country} with ${funding:.1f}M in funding.
        The company has a burn rate of {burn} months, resulting in an estimated runway of {runway_months:.1f} months.
        The founding team has {experience} years of average experience in a market valued at ${market}B.
        Current status: <b>{status}</b>.
        """
    else:
        # Portfolio analysis
        total_startups = len(df_in)
        total_funding = df_in['Funding_USD_M'].sum()
        failed_count = df_in['Failed'].sum()
        success_rate = ((total_startups - failed_count) / total_startups) * 100
        avg_funding = df_in['Funding_USD_M'].mean()

        summary_text = f"""
        This report analyzes a portfolio of <b>{total_startups} startups</b> with total funding of
        <b>${total_funding:.1f}M</b>. The portfolio shows a success rate of <b>{success_rate:.1f}%</b>
        ({total_startups - failed_count} successful, {failed_count} failed). Average funding per startup
        is <b>${avg_funding:.1f}M</b>.
        """

    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))

    # =============================
    # KEY METRICS TABLE
    # =============================

    story.append(Paragraph("Key Metrics", heading_style))

    if startup_name:
        metrics_data = [
            ['Metric', 'Value'],
            ['Funding Amount', f'${funding:.1f}M'],
            ['Burn Rate', f'{burn} months'],
            ['Estimated Runway', f'{runway_months:.1f} months'],
            ['Sector', sector],
            ['Country', country],
            ['Founder Experience', f'{experience} years'],
            ['Market Size', f'${market}B'],
            ['Status', status]
        ]
    else:
        sectors = df_in['Sector'].nunique()
        countries = df_in['Country'].nunique()
        avg_experience = df_in['Founders_Experience_Yrs'].mean()

        metrics_data = [
            ['Metric', 'Value'],
            ['Total Startups', str(total_startups)],
            ['Total Funding', f'${total_funding:.1f}M'],
            ['Success Rate', f'{success_rate:.1f}%'],
            ['Average Funding', f'${avg_funding:.1f}M'],
            ['Sectors Covered', str(sectors)],
            ['Countries', str(countries)],
            ['Avg Founder Experience', f'{avg_experience:.1f} years']
        ]

    metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff4b4b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))

    # =============================
    # VISUALIZATIONS
    # =============================

    story.append(PageBreak())
    story.append(Paragraph("Data Visualizations", heading_style))
    story.append(Spacer(1, 0.2*inch))

    # Chart 1: Timeline
    story.append(Paragraph("1. Failure Timeline Analysis", subheading_style))
    timeline_bytes = plot_failure_timeline(df_in)
    timeline_img = Image(io.BytesIO(timeline_bytes), width=6*inch, height=3*inch)
    story.append(timeline_img)
    story.append(Spacer(1, 0.2*inch))

    # Chart 2: Funding vs Burn
    story.append(Paragraph("2. Funding vs Burn Rate", subheading_style))
    funding_burn_bytes = plot_funding_vs_burn(df_in)
    funding_burn_img = Image(io.BytesIO(funding_burn_bytes), width=6*inch, height=3*inch)
    story.append(funding_burn_img)
    story.append(Spacer(1, 0.2*inch))

    story.append(PageBreak())

    # Chart 3: Sector Comparison
    story.append(Paragraph("3. Sector Analysis", subheading_style))
    sector_bytes = plot_sector_comparison(df_in)
    sector_img = Image(io.BytesIO(sector_bytes), width=6*inch, height=3*inch)
    story.append(sector_img)
    story.append(Spacer(1, 0.2*inch))

    # Chart 4: Country Analysis
    story.append(Paragraph("4. Geographic Distribution", subheading_style))
    country_bytes = plot_failure_rate_by_country(df_in)
    country_img = Image(io.BytesIO(country_bytes), width=6*inch, height=3*inch)
    story.append(country_img)

    # =============================
    # RISK ANALYSIS
    # =============================

    story.append(PageBreak())
    story.append(Paragraph("Risk Analysis", heading_style))

    # Calculate risk factors
    high_risk = df_in[df_in['Burn_Rate_Months'] < 10]
    low_funding = df_in[df_in['Funding_USD_M'] < 3.0]
    inexperienced = df_in[df_in['Founders_Experience_Yrs'] < 3]

    risk_text = f"""
    <b>High-Risk Indicators:</b><br/>
    - {len(high_risk)} startups with burn rate under 10 months (high risk)<br/>
    - {len(low_funding)} startups with funding under $3M (undercapitalized)<br/>
    - {len(inexperienced)} startups with founders having less than 3 years experience<br/><br/>

    <b>Key Risks:</b><br/>
    1. <b>Runway Risk:</b> Startups with short runways may fail before achieving product-market fit<br/>
    2. <b>Market Risk:</b> Smaller markets limit growth potential and exit opportunities<br/>
    3. <b>Team Risk:</b> Inexperienced founders may lack operational expertise<br/>
    4. <b>Competitive Risk:</b> Crowded sectors reduce differentiation and margins
    """

    story.append(Paragraph(risk_text, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))

    # =============================
    # RECOMMENDATIONS
    # =============================

    story.append(Paragraph("Investment Recommendations", heading_style))

    if startup_name:
        # Calculate viability score
        viability_features = {
            'funding_usd_m': funding,
            'burn_rate_months': burn,
            'team_experience_years': experience,
            'market_size_bn': market,
            'business_model_strength_1_5': 3,  # Default
            'moat_1_5': 3,  # Default
            'traction_mrr_k': 10,  # Default
            'growth_rate_pct': 5,  # Default
            'competition_intensity_1_5': 3  # Default
        }

        viability_result = viability_score(viability_features)
        score = viability_result['score']

        if score >= 60:
            recommendation = "<b>INVEST</b> - Strong fundamentals with acceptable risk profile"
            color_code = "green"
        elif score >= 40:
            recommendation = "<b>MONITOR</b> - Moderate risk, requires additional due diligence"
            color_code = "orange"
        else:
            recommendation = "<b>PASS</b> - High risk factors outweigh potential returns"
            color_code = "red"

        rec_text = f"""
        <b>Viability Score: {score:.1f}/100</b><br/><br/>

        <b>Recommendation: {recommendation}</b><br/><br/>

        <b>Rationale:</b><br/>
        - Runway of {runway_months:.1f} months provides {'adequate' if runway_months > 18 else 'limited'} time to achieve milestones<br/>
        - Market size of ${market}B offers {'strong' if market > 50 else 'moderate'} growth potential<br/>
        - Team experience of {experience} years is {'above' if experience >= 5 else 'below'} industry average<br/>
        - Current status: {status}
        """
    else:
        # Portfolio recommendations
        top_performers = df_in[df_in['Failed'] == 0].nlargest(3, 'Funding_USD_M')

        rec_text = f"""
        <b>Portfolio Recommendations:</b><br/><br/>

        1. <b>Diversify Sector Exposure:</b> Current portfolio concentrated in certain sectors<br/>
        2. <b>Monitor High-Risk Startups:</b> {len(high_risk)} companies need immediate attention<br/>
        3. <b>Increase Follow-On Funding:</b> Top performers may benefit from additional capital<br/><br/>

        <b>Top 3 Performing Startups:</b><br/>
        """

        for idx, row in top_performers.iterrows():
            rec_text += f"- {row['Startup']} ({row['Sector']}) - ${row['Funding_USD_M']}M funding<br/>"

    story.append(Paragraph(rec_text, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))

    # =============================
    # FOOTER
    # =============================

    story.append(Spacer(1, 0.5*inch))
    footer_text = """
    <i>This report was automatically generated by NAVADA (Startup Viability Agent).
    All analysis is based on provided data and should be supplemented with additional due diligence.
    Past performance does not guarantee future results.</i>
    """
    story.append(Paragraph(footer_text, styles['Italic']))

    # Build PDF
    doc.build(story)

    # Get PDF bytes
    buffer.seek(0)
    return buffer.getvalue()


def train_ml_model(df: pd.DataFrame):
    """
    Train a Random Forest classifier to predict startup failure.

    Args:
        df (pd.DataFrame): Startup dataset with features and 'Failed' column

    Returns:
        RandomForestClassifier: Trained model

    Features used:
        - Funding_USD_M: Total funding in millions
        - Burn_Rate_Months: Burn rate in months
        - Founders_Experience_Yrs: Founder experience in years
        - Market_Size_Bn: Market size in billions

    Target:
        - Failed: 0 = success, 1 = failed
    """
    # Select feature columns
    X = df[["Funding_USD_M", "Burn_Rate_Months", "Founders_Experience_Yrs", "Market_Size_Bn"]]

    # Target variable
    y = df["Failed"]

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

# =============================
# INTERACTIVE DASHBOARD FUNCTIONS
# =============================

def create_interactive_scatter(df_in: pd.DataFrame, title: str = "Interactive Startup Analysis") -> str:
    """
    Create an interactive Plotly scatter plot with hover details and click functionality.

    This function generates a comprehensive interactive visualization that allows users to:
    - Hover over points to see detailed startup information
    - Zoom and pan to explore specific regions of the data
    - Filter by clicking legend items
    - Export the chart in various formats

    The scatter plot uses multiple visual encodings:
    - X-axis: Funding amount (USD Millions)
    - Y-axis: Burn rate (Months of runway)
    - Size: Market size (larger bubble = bigger market)
    - Color: Success/Failure status (green = success, red = failure)
    - Symbol: Sector (different shapes for different industries)

    Args:
        df_in (pd.DataFrame): Input startup dataset with required columns:
            - Funding_USD_M: Funding in millions USD
            - Burn_Rate_Months: Burn rate in months
            - Market_Size_Bn: Market size in billions
            - Failed: 0 = success, 1 = failed
            - Sector: Industry sector
            - Country: Country of operation
        title (str): Chart title for display

    Returns:
        str: Complete HTML string containing the interactive chart with embedded Plotly.js
    """
    # Create the base scatter plot with multiple visual dimensions
    # Each data point represents one startup with 6+ attributes encoded visually
    fig = px.scatter(
        df_in,                              # Input DataFrame
        x="Funding_USD_M",                  # X-axis: funding amount
        y="Burn_Rate_Months",               # Y-axis: burn rate (runway)
        size="Market_Size_Bn",              # Bubble size: market opportunity
        color="Failed",                     # Color coding: success (0) vs failure (1)
        symbol="Sector",                    # Shape coding: different sectors
        hover_data={
            "Startup": True,                    # Show startup name on hover
            "Founders_Experience_Yrs": True,   # Show founder experience years
            "Country": True,                    # Show country of operation
            "Market_Size_Bn": ":.1f",         # Format market size to 1 decimal
            "Funding_USD_M": ":.1f",          # Format funding to 1 decimal
            "Burn_Rate_Months": ":.1f"        # Format burn rate to 1 decimal
        },
        title=title,                        # Dynamic title from parameter
        labels={
            "Funding_USD_M": "Funding (USD Millions)",      # User-friendly axis label
            "Burn_Rate_Months": "Burn Rate (Months)",       # User-friendly axis label
            "Failed": "Status"                              # User-friendly legend label
        },
        color_discrete_map={0: "green", 1: "red"},          # Explicit color mapping
        width=800,                          # Fixed width for consistency
        height=600                          # Fixed height for consistency
    )

    # Customize layout for enhanced user experience and interactivity
    fig.update_layout(
        hovermode="closest",                # Show hover info for nearest point only
        showlegend=True,                    # Display legend for color/symbol mapping
        legend=dict(                        # Position legend horizontally at top
            orientation="h",                # Horizontal orientation saves space
            yanchor="bottom",               # Anchor to bottom of legend box
            y=1.02,                        # Position slightly above chart
            xanchor="right",               # Align to right side
            x=1                            # Full width positioning
        ),
        margin=dict(                       # Mobile-optimized margins
            t=40,                          # Reduced top margin
            b=40,                          # Reduced bottom margin
            l=40,                          # Reduced left margin
            r=40                           # Reduced right margin
        ),
        autosize=True,                      # Enable responsive sizing for mobile
        font=dict(size=10)                  # Smaller font size for mobile readability
    )

    # Add user instruction annotation for better UX
    # This helps users understand they can interact with the chart
    fig.add_annotation(
        text="Click and drag to zoom, hover for details, double-click to reset",  # Clear instructions
        showarrow=False,                   # No arrow pointing to anything
        xref="paper", yref="paper",        # Use paper coordinates (0-1 range)
        x=0.5, y=-0.1,                    # Center horizontally, below chart
        xanchor='center', yanchor='top',   # Center the text anchor point
        font=dict(size=12, color="gray")   # Subtle gray color, readable size
    )

    # Convert Plotly figure to standalone HTML string
    # include_plotlyjs='cdn' loads Plotly.js from CDN (smaller file size)
    # div_id provides unique identifier for multiple charts on same page
    return fig.to_html(include_plotlyjs='cdn', div_id="interactive-chart")

def create_interactive_timeline(df_in: pd.DataFrame) -> str:
    """
    Create an interactive timeline showing failure progression over time.

    Args:
        df_in (pd.DataFrame): Input startup dataset

    Returns:
        str: HTML string of the interactive timeline
    """
    # Calculate failure timeline (same logic as static version)
    timeline_data = []
    for _, row in df_in.iterrows():
        if row["Funding_USD_M"] > 0 and row["Burn_Rate_Months"] > 0:
            failure_time = row["Funding_USD_M"] / (12 / row["Burn_Rate_Months"])
            timeline_data.append({
                "Startup": row["Startup"],
                "Failure_Time_Years": failure_time,
                "Sector": row["Sector"],
                "Funding": row["Funding_USD_M"],
                "Status": "Failed" if row["Failed"] else "Active"
            })

    timeline_df = pd.DataFrame(timeline_data).sort_values("Failure_Time_Years")

    # Create interactive bar chart
    fig = px.bar(
        timeline_df,
        x="Failure_Time_Years",
        y="Startup",
        color="Status",
        hover_data=["Sector", "Funding"],
        title="📈 Interactive Failure Timeline - Hover for Details",
        labels={
            "Failure_Time_Years": "Estimated Failure Time (Years)",
            "Startup": "Startup Name"
        },
        color_discrete_map={"Failed": "red", "Active": "green"},
        orientation="h",
        width=None,                    # Responsive width for mobile
        height=400,                    # Reduced height for mobile
        autosize=True                  # Enable auto-sizing
    )

    fig.update_layout(
        hovermode="y unified",
        yaxis={'categoryorder':'total ascending'}
    )

    return fig.to_html(include_plotlyjs='cdn', div_id="interactive-timeline")

def create_sector_dashboard(df_in: pd.DataFrame) -> str:
    """
    Create an interactive multi-chart dashboard for sector analysis.

    Args:
        df_in (pd.DataFrame): Input startup dataset

    Returns:
        str: HTML string of the interactive dashboard
    """
    from plotly.subplots import make_subplots

    # Create subplot figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Average Funding by Sector",
            "Failure Rate by Sector",
            "Experience vs Funding",
            "Market Size Distribution"
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )

    # Chart 1: Average funding by sector
    sector_avg = df_in.groupby("Sector")["Funding_USD_M"].mean().reset_index()
    fig.add_trace(
        go.Bar(x=sector_avg["Sector"], y=sector_avg["Funding_USD_M"],
               name="Avg Funding", marker_color="lightblue"),
        row=1, col=1
    )

    # Chart 2: Failure rate by sector
    sector_failure = df_in.groupby("Sector")["Failed"].mean().reset_index()
    fig.add_trace(
        go.Bar(x=sector_failure["Sector"], y=sector_failure["Failed"],
               name="Failure Rate", marker_color="salmon"),
        row=1, col=2
    )

    # Chart 3: Experience vs Funding scatter
    fig.add_trace(
        go.Scatter(
            x=df_in["Founders_Experience_Yrs"],
            y=df_in["Funding_USD_M"],
            mode="markers",
            marker=dict(
                size=8,
                color=df_in["Failed"],
                colorscale="RdYlGn_r",
                showscale=True
            ),
            name="Startups",
            text=df_in["Startup"],
            hovertemplate="<b>%{text}</b><br>Experience: %{x} years<br>Funding: $%{y}M<extra></extra>"
        ),
        row=2, col=1
    )

    # Chart 4: Market size distribution
    fig.add_trace(
        go.Histogram(x=df_in["Market_Size_Bn"], nbinsx=10,
                    name="Market Size", marker_color="lightgreen"),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=600,                    # Reduced height for mobile
        title_text="🏭 Interactive Sector Dashboard - Click and Zoom to Explore",
        showlegend=True
    )

    return fig.to_html(include_plotlyjs='cdn', div_id="sector-dashboard")

# =============================
# SESSION MEMORY FUNCTIONS
# =============================

def get_session_id() -> str:
    """Get or create session ID for memory tracking."""
    session = cl.user_session.get("session_id")
    if not session:
        import uuid
        session = str(uuid.uuid4())[:8]
        cl.user_session.set("session_id", session)
    return session

def add_to_memory(session_id: str, role: str, content: str):
    """Add a message to session memory."""
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    SESSION_MEMORY[session_id].append({
        "role": role,
        "content": content,
        "timestamp": pd.Timestamp.now()
    })

    # Keep only last 20 messages to avoid token limits
    if len(SESSION_MEMORY[session_id]) > 20:
        SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-20:]

def get_memory_context(session_id: str) -> str:
    """Get formatted conversation history for context."""
    if session_id not in SESSION_MEMORY:
        return ""

    history = SESSION_MEMORY[session_id][-10:]  # Last 10 messages
    context = "Recent conversation history:\n"
    for msg in history:
        context += f"- {msg['role']}: {msg['content'][:100]}...\n"
    return context

def get_current_persona() -> Dict[str, str]:
    """Get current persona settings from session."""
    persona_name = cl.user_session.get("persona", "founder")
    return PERSONAS.get(persona_name, PERSONAS["founder"])

def format_persona_recommendations(persona_name: str) -> str:
    """Format key recommendations for a persona mode."""
    persona = PERSONAS.get(persona_name, {})
    recommendations = persona.get("key_recommendations", [])

    if not recommendations:
        return ""

    formatted = f"\n\n🎯 **Key {persona.get('name', persona_name)} Recommendations:**\n\n"
    for rec in recommendations:
        formatted += f"• {rec}\n\n"

    return formatted

# =============================
# LANGSMITH THREAD MANAGEMENT
# =============================

def get_thread_history(thread_id: str, project_name: str) -> List[Dict[str, str]]:
    """
    Gets a history of all LLM calls in the thread to construct conversation history

    Args:
        thread_id (str): The thread/session ID to retrieve history for
        project_name (str): LangSmith project name

    Returns:
        List[Dict[str, str]]: List of message objects with role and content
    """
    if not langsmith_client:
        return []

    try:
        # Filter runs by the specific thread and project
        filter_string = f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "{thread_id}"))'

        # Only grab the LLM runs
        runs = [r for r in langsmith_client.list_runs(
            project_name=project_name,
            filter=filter_string,
            run_type="llm"
        )]

        # Sort by start time to get chronological order
        runs = sorted(runs, key=lambda run: run.start_time)

        # Extract conversation history
        messages = []
        for run in runs:
            if hasattr(run, 'inputs') and 'messages' in run.inputs:
                # Add input messages
                messages.extend(run.inputs['messages'])

                # Add assistant response
                if hasattr(run, 'outputs') and 'choices' in run.outputs:
                    assistant_msg = run.outputs['choices'][0]['message']
                    messages.append(assistant_msg)

        return messages

    except Exception as e:
        print(f"Error retrieving thread history: {str(e)}")
        return []


@traceable(
    name="NAVADA Chat Pipeline",
    run_type="chain",
    tags=["navada", "startup-analysis", "conversational-ai"],
    metadata={
        "app_name": "navada",
        "app_version": "2.0.0",
        "environment": "production"
    }
)
def navada_chat_pipeline(question: str, session_id: str, persona: str, get_chat_history: bool = False) -> str:
    """
    Enhanced chat pipeline with LangSmith thread management for NAVADA

    Args:
        question (str): User's question/input
        session_id (str): Unique session identifier for thread tracking
        persona (str): Current persona mode (investor/founder)
        get_chat_history (bool): Whether to retrieve conversation history

    Returns:
        str: AI response content
    """
    try:
        # Get current run tree for dynamic metadata and tags
        current_run = ls.get_current_run_tree()

        # Add dynamic metadata based on current context
        if current_run:
            current_run.metadata.update({
                "session_id": session_id,
                "persona_mode": persona,
                "conversation_type": "thread_continuation" if get_chat_history else "new_conversation",
                "question_length": len(question),
                "timestamp": pd.Timestamp.now().isoformat()
            })

            # Add dynamic tags based on persona and question type
            dynamic_tags = [f"persona-{persona}"]

            # Detect question type and add relevant tags
            question_lower = question.lower()
            if "funding" in question_lower or "investment" in question_lower:
                dynamic_tags.append("funding-analysis")
            if "market" in question_lower or "competition" in question_lower:
                dynamic_tags.append("market-analysis")
            if "team" in question_lower or "founder" in question_lower:
                dynamic_tags.append("team-analysis")
            if "chart" in question_lower or "plot" in question_lower or "visualization" in question_lower:
                dynamic_tags.append("data-visualization")

            current_run.tags.extend(dynamic_tags)

        # Set up LangSmith metadata for thread tracking
        langsmith_extra = {
            "project_name": LANGSMITH_PROJECT,
            "metadata": {
                "session_id": session_id,
                "persona": persona,
                "app": "navada",
                "trace_type": "chat_pipeline"
            },
            "tags": [f"session-{session_id[:8]}", f"persona-{persona}"]
        }

        # Build conversation context
        if get_chat_history and langsmith_client:
            # Get LangSmith thread history
            thread_messages = get_thread_history(session_id, LANGSMITH_PROJECT)

            # Combine with new user question
            messages = thread_messages + [{"role": "user", "content": question}]
        else:
            # Start fresh conversation
            messages = [{"role": "user", "content": question}]

        # Get current persona information
        current_persona = PERSONAS.get(persona, PERSONAS["investor"])

        # Add persona system message if starting fresh or no history
        if not get_chat_history or not messages:
            system_msg = {
                "role": "system",
                "content": current_persona["system_prompt"]
            }
            messages = [system_msg] + messages

        # Create chat completion with LangSmith metadata
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in NAVADA chat pipeline: {str(e)}")
        return f"I apologize, but I encountered an error processing your request: {str(e)}"


def get_session_metadata(session_id: str) -> Dict[str, Any]:
    """
    Get metadata for a session including conversation count and current persona

    Args:
        session_id (str): Session identifier

    Returns:
        Dict[str, Any]: Session metadata
    """
    if not langsmith_client:
        return {"conversation_count": 0, "persona": "investor"}

    try:
        filter_string = f'and(in(metadata_key, ["session_id"]), eq(metadata_value, "{session_id}"))'
        runs = list(langsmith_client.list_runs(
            project_name=LANGSMITH_PROJECT,
            filter=filter_string,
            run_type="llm"
        ))

        return {
            "conversation_count": len(runs),
            "persona": runs[-1].extra.get("metadata", {}).get("persona", "investor") if runs else "investor",
            "last_interaction": runs[-1].start_time if runs else None
        }

    except Exception as e:
        print(f"Error getting session metadata: {str(e)}")
        return {"conversation_count": 0, "persona": "investor"}


# =============================
# AUTO-GENERATED INSIGHTS FUNCTIONS
# =============================

def generate_insights(df_in: pd.DataFrame, analysis_type: str = "general") -> Dict[str, List[str]]:
    """
    Generate automated insights based on data analysis.

    Args:
        df_in (pd.DataFrame): Input dataset
        analysis_type (str): Type of analysis performed

    Returns:
        Dict with risks, opportunities, and recommendations
    """
    insights = {
        "risks": [],
        "opportunities": [],
        "recommendations": []
    }

    # Calculate key metrics
    avg_funding = df_in["Funding_USD_M"].mean()
    failure_rate = df_in["Failed"].mean()
    avg_burn = df_in["Burn_Rate_Months"].mean()
    avg_experience = df_in["Founders_Experience_Yrs"].mean()

    # Risk Detection
    if failure_rate > 0.5:
        insights["risks"].append(f"🔴 High failure rate detected: {failure_rate:.0%} of startups failed")

    if avg_burn < 6:
        insights["risks"].append(f"🔴 Short runway alert: Average burn rate is only {avg_burn:.1f} months")

    if avg_experience < 3:
        insights["risks"].append(f"🔴 Inexperienced teams: Average founder experience is {avg_experience:.1f} years")

    # Opportunity Detection
    high_funding_sectors = df_in.groupby("Sector")["Funding_USD_M"].mean().sort_values(ascending=False).head(2)
    for sector, funding in high_funding_sectors.items():
        if funding > avg_funding * 1.5:
            insights["opportunities"].append(f"🟢 Hot sector identified: {sector} (avg funding ${funding:.1f}M)")

    successful_patterns = df_in[df_in["Failed"] == 0]
    if len(successful_patterns) > 0:
        success_funding = successful_patterns["Funding_USD_M"].mean()
        insights["opportunities"].append(f"🟢 Success pattern: Successful startups raised avg ${success_funding:.1f}M")

    # Recommendations
    if avg_burn < 12:
        insights["recommendations"].append("💡 Extend runway: Focus on increasing funding or reducing burn rate")

    if failure_rate > 0.4:
        insights["recommendations"].append("💡 De-risk strategy: Consider pivot to sectors with lower failure rates")

    insights["recommendations"].append("💡 Track metrics: Monitor burn rate, customer acquisition, and team experience")

    return insights

def format_insights_message(insights: Dict[str, List[str]]) -> str:
    """Format insights into a readable message."""
    message = "## 🤖 Auto-Generated Insights\n\n"

    if insights["risks"]:
        message += "### ⚠️ Top Risks Detected:\n"
        for risk in insights["risks"][:3]:  # Top 3 risks
            message += f"- {risk}\n"
        message += "\n"

    if insights["opportunities"]:
        message += "### 🎯 Opportunities Identified:\n"
        for opp in insights["opportunities"][:3]:  # Top 3 opportunities
            message += f"- {opp}\n"
        message += "\n"

    if insights["recommendations"]:
        message += "### 💡 Next Steps:\n"
        for rec in insights["recommendations"][:3]:  # Top 3 recommendations
            message += f"- {rec}\n"

    return message

# =============================
# WEB SCRAPING FUNCTIONS
# =============================

def validate_url(url: str) -> bool:
    """
    Validate URL format and check for security concerns.

    Args:
        url (str): URL to validate

    Returns:
        bool: True if URL is valid and safe, False otherwise
    """
    try:
        # Parse URL to check structure
        parsed = urlparse(url)

        # Must have scheme (http/https) and netloc (domain)
        if not all([parsed.scheme, parsed.netloc]):
            return False

        # Only allow HTTP/HTTPS protocols for security
        if parsed.scheme not in ['http', 'https']:
            return False

        # Block dangerous or inappropriate domains
        blocked_domains = [
            'localhost', '127.0.0.1', '0.0.0.0',  # Local addresses
            'file://', 'ftp://',                   # Non-web protocols
        ]

        netloc_lower = parsed.netloc.lower()
        for blocked in blocked_domains:
            if blocked in netloc_lower:
                return False

        return True

    except Exception:
        return False

def scrape_site(url: str, selector: str = "p") -> Dict[str, Any]:
    """
    Scrape text content from a website with comprehensive safety measures and error handling.

    This function performs web scraping with multiple safeguards:
    - URL validation and security checks
    - Request timeouts and size limits
    - Content filtering and cleaning
    - Structured error reporting

    Args:
        url (str): Website URL to scrape (must be http/https)
        selector (str): CSS selector for content extraction
            - "p" = paragraphs (default)
            - "h1, h2, h3" = headings
            - ".class-name" = by CSS class
            - "#id-name" = by element ID

    Returns:
        Dict[str, Any]: Scraping results containing:
            - success (bool): Whether scraping succeeded
            - data (pd.DataFrame): Scraped content (if successful)
            - url (str): Original URL
            - count (int): Number of items scraped
            - error (str): Error message (if failed)
            - size_mb (float): Content size in megabytes
    """
    result = {
        "success": False,
        "data": pd.DataFrame(),
        "url": url,
        "count": 0,
        "error": "",
        "size_mb": 0.0
    }

    try:
        # Step 1: Validate URL for security and format
        if not validate_url(url):
            result["error"] = "Invalid or unsafe URL. Use http/https URLs only."
            return result

        # Step 2: Configure HTTP request with safety limits
        headers = {
            'User-Agent': 'NAVADA-Bot/1.0 (Educational Web Scraping Tool)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        # Step 3: Make HTTP request with timeout and size limits
        response = requests.get(
            url,
            headers=headers,
            timeout=15,                    # 15 second timeout
            stream=True,                   # Stream to check size before downloading
            allow_redirects=True,          # Follow redirects (max 30 by default)
            verify=True                    # Verify SSL certificates
        )

        # Step 4: Check response size before processing (max 5MB)
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 5 * 1024 * 1024:  # 5MB limit
            result["error"] = "Content too large (>5MB). Choose a smaller page."
            return result

        # Step 5: Check HTTP status code
        response.raise_for_status()  # Raises exception for 4xx/5xx status codes

        # Step 6: Get content and check actual size
        content = response.text
        content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
        result["size_mb"] = round(content_size_mb, 2)

        if content_size_mb > 5:  # Double-check size after download
            result["error"] = f"Content too large ({content_size_mb:.1f}MB). Choose a smaller page."
            return result

        # Step 7: Parse HTML with BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")

        # Step 8: Remove script and style elements (they contain non-content)
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Step 9: Extract content using CSS selector
        elements = soup.select(selector)

        # Step 10: Clean and filter extracted text
        scraped_content = []
        for element in elements:
            text = element.get_text(strip=True)

            # Filter out empty content and very short text
            if text and len(text) > 10:
                # Clean whitespace and normalize
                text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
                text = text.strip()

                # Limit individual text length to prevent spam
                if len(text) <= 2000:
                    scraped_content.append(text)

        # Step 11: Create structured DataFrame
        if scraped_content:
            result["data"] = pd.DataFrame({
                "content": scraped_content,
                "length": [len(text) for text in scraped_content],
                "source": [url] * len(scraped_content)
            })
            result["count"] = len(scraped_content)
            result["success"] = True
        else:
            result["error"] = f"No content found using selector '{selector}'. Try different selectors like 'h1', 'div', or 'span'."

    except requests.exceptions.Timeout:
        result["error"] = "Request timed out. The website may be slow or unresponsive."
    except requests.exceptions.ConnectionError:
        result["error"] = "Could not connect to the website. Check the URL and internet connection."
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP error {e.response.status_code}: {e.response.reason}"
    except requests.exceptions.RequestException as e:
        result["error"] = f"Request failed: {str(e)}"
    except Exception as e:
        result["error"] = f"Scraping failed: {str(e)}"

    return result

def analyze_scraped_content(scraped_data: pd.DataFrame, url: str, persona: Dict[str, str]) -> str:
    """
    Use GPT to analyze scraped website content with persona-specific focus.

    Args:
        scraped_data (pd.DataFrame): DataFrame containing scraped content
        url (str): Original URL for context
        persona (Dict[str, str]): Current user persona (investor/founder)

    Returns:
        str: AI analysis of the scraped content
    """
    if scraped_data.empty:
        return "No content available for analysis."

    # Prepare content for GPT analysis
    # Take top 20 content items to stay within token limits
    content_items = scraped_data.head(20)["content"].tolist()
    content_text = "\n\n".join([f"Section {i+1}: {text}" for i, text in enumerate(content_items)])

    # Truncate if too long (approximately 3000 tokens = 12000 characters)
    if len(content_text) > 10000:
        content_text = content_text[:10000] + "\n\n[Content truncated for analysis...]"

    # Create persona-specific analysis prompt
    persona_focus = persona.get('system_prompt', '')
    analysis_style = ""

    if 'investor' in persona.get('name', '').lower():
        analysis_style = (
            "Focus on investment opportunities, market analysis, business models, "
            "competitive landscape, and financial indicators. Identify potential risks and ROI factors."
        )
    else:  # founder mode
        analysis_style = (
            "Focus on actionable insights, operational strategies, market positioning, "
            "customer needs, and execution opportunities. Provide tactical recommendations."
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are analyzing web content from {url}. "
                        f"{analysis_style}\n\n"
                        "Provide a structured analysis with:\n"
                        "1. Key insights (3-5 bullet points)\n"
                        "2. Notable patterns or trends\n"
                        "3. Actionable recommendations\n"
                        "4. Potential concerns or red flags\n\n"
                        "Keep analysis concise but insightful."
                    )
                },
                {
                    "role": "user",
                    "content": f"Analyze this website content:\n\nURL: {url}\n\nContent:\n{content_text}"
                }
            ],
            max_tokens=600,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Analysis failed: {str(e)}"


# =============================
# INTERNET SEARCH FUNCTIONALITY
# =============================

@traceable(
    name="NAVADA Internet Search",
    run_type="tool",
    tags=["navada", "search", "brave-api", "market-intelligence"],
    metadata={"tool_type": "internet_search", "api_provider": "brave_search"}
)
def search_internet(query: str, count: int = 5) -> Dict[str, Any]:
    """
    Search the internet using Brave Search API

    Args:
        query (str): Search query
        count (int): Number of results to return (default 5)

    Returns:
        dict: Search results with titles, descriptions, and URLs
    """
    # Add dynamic metadata to current run
    current_run = ls.get_current_run_tree()
    if current_run:
        current_run.metadata.update({
            "search_query": query,
            "requested_count": count,
            "query_length": len(query),
            "timestamp": pd.Timestamp.now().isoformat()
        })

        # Add query-specific tags
        query_lower = query.lower()
        search_tags = []
        if "startup" in query_lower:
            search_tags.append("startup-search")
        if "funding" in query_lower or "investment" in query_lower:
            search_tags.append("funding-search")
        if "market" in query_lower:
            search_tags.append("market-research")
        if "competition" in query_lower:
            search_tags.append("competitive-analysis")

        current_run.tags.extend(search_tags)

    if not search_api_key or search_api_key == "your_brave_search_api_key_here":
        logger.warning("Search API key not configured or using placeholder value")
        return {
            "success": False,
            "error": "Search API key not configured. Please set SEARCH_API_KEY in your .env file with a valid Brave Search API key.",
            "results": [],
            "fallback_available": True
        }

    try:
        # Brave Search API endpoint
        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": search_api_key
        }

        params = {
            "q": query,
            "count": count,
            "offset": 0,
            "mkt": "en-US",
            "safesearch": "moderate",
            "freshness": "pw",  # Past week for fresh results
            "text_decorations": False,
            "search_lang": "en"
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Extract relevant information from search results
        results = []
        if "web" in data and "results" in data["web"]:
            for result in data["web"]["results"][:count]:
                results.append({
                    "title": result.get("title", ""),
                    "description": result.get("description", ""),
                    "url": result.get("url", ""),
                    "age": result.get("age", ""),
                    "language": result.get("language", "en")
                })

        return {
            "success": True,
            "query": query,
            "results": results,
            "total_results": len(results)
        }

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during search: {e}")
        if e.response.status_code == 422:
            return {
                "success": False,
                "error": "Search API authentication failed. Please check your SEARCH_API_KEY in .env file.",
                "results": [],
                "fallback_available": True,
                "status_code": 422
            }
        else:
            return {
                "success": False,
                "error": f"Search request failed: HTTP {e.response.status_code}",
                "results": [],
                "fallback_available": True
            }
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during search: {e}")
        return {
            "success": False,
            "error": f"Search request failed: {str(e)}",
            "results": [],
            "fallback_available": True
        }
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Search error: {str(e)}",
            "results": [],
            "fallback_available": True
        }


def analyze_search_results(search_data: Dict[str, Any], persona: Dict[str, str], context: str = "", session_id: str = None) -> str:
    """
    Analyze search results using AI based on current persona mode

    Args:
        search_data (dict): Search results from search_internet()
        persona (dict): Current persona (investor/founder mode)
        context (str): Additional context for analysis

    Returns:
        str: AI analysis of search results
    """
    if not search_data["success"] or not search_data["results"]:
        return "No search results to analyze or search failed."

    try:
        # Format search results for analysis
        results_text = f"Search Query: {search_data['query']}\n\n"
        results_text += f"Found {search_data['total_results']} results:\n\n"

        for i, result in enumerate(search_data['results'], 1):
            results_text += f"{i}. **{result['title']}**\n"
            results_text += f"   URL: {result['url']}\n"
            results_text += f"   Description: {result['description']}\n"
            if result.get('age'):
                results_text += f"   Age: {result['age']}\n"
            results_text += "\n"

        # Create persona-specific analysis prompt
        analysis_prompt = f"{persona['system_prompt']}\n\n"
        analysis_prompt += f"Analyze these search results from a {persona['name']} perspective.\n\n"

        if context:
            analysis_prompt += f"Context: {context}\n\n"

        analysis_prompt += "Provide insights, opportunities, risks, and actionable recommendations based on the search results."

        # Include LangSmith metadata if session_id is available
        if langsmith_client and session_id:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": analysis_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Analyze these search results:\n\n{results_text}"
                    }
                ],
                max_tokens=800,
                temperature=0.7,
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": analysis_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Analyze these search results:\n\n{results_text}"
                    }
                ],
                max_tokens=800,
                temperature=0.7
            )

        return response.choices[0].message.content

    except Exception as e:
        return f"Analysis failed: {str(e)}"


def generate_search_query(user_question: str, persona: str) -> str:
    """
    Generate intelligent search queries based on user question and persona mode

    Args:
        user_question (str): User's original question
        persona (str): Current persona mode (investor/founder)

    Returns:
        str: Optimized search query for Brave Search API
    """
    try:
        # Persona-specific search query templates
        investor_keywords = {
            "market": "startup market trends valuation 2025",
            "competition": "startup competitive landscape industry analysis",
            "funding": "venture capital funding trends startup investment 2025",
            "exit": "startup exit strategies IPO acquisition trends",
            "valuation": "startup valuation metrics Series A B C funding",
            "growth": "startup growth metrics scaling strategies",
            "roi": "startup ROI investment returns venture capital",
            "due diligence": "startup due diligence checklist investment"
        }

        founder_keywords = {
            "market": "startup market validation product-market fit",
            "competition": "startup competitor analysis differentiation",
            "funding": "startup fundraising tips pitch deck Series A",
            "growth": "startup growth hacking scaling strategies",
            "team": "startup team building hiring strategies",
            "product": "startup product development MVP strategies",
            "customer": "startup customer acquisition retention strategies",
            "pivot": "startup pivot strategies when to pivot"
        }

        # Select keyword set based on persona
        keywords = investor_keywords if persona == "investor" else founder_keywords

        # Extract key topics from user question
        question_lower = user_question.lower()

        # Find matching keywords and build search query
        search_terms = []

        for topic, search_template in keywords.items():
            if topic in question_lower:
                search_terms.append(search_template)
                break  # Use first match to avoid overly complex queries

        # Add year context for recent information
        current_year = "2025"
        if current_year not in user_question:
            search_terms.append(current_year)

        # If no specific keywords found, use general startup search
        if not search_terms:
            if persona == "investor":
                search_terms = ["startup investment trends 2025", "venture capital market"]
            else:
                search_terms = ["startup trends 2025", "founder advice entrepreneurship"]

        # Combine and return the search query
        return " ".join(search_terms[:2])  # Limit to 2 main search terms

    except Exception as e:
        print(f"Error generating search query: {str(e)}")
        return f"startup {persona} trends 2025"


# =============================
# TEXT-TO-SPEECH FUNCTIONALITY
# =============================

def generate_speech(text: str, voice: str = "alloy") -> bytes:
    """
    Generate speech from text using OpenAI TTS API

    Args:
        text (str): Text to convert to speech
        voice (str): Voice to use (alloy, echo, fable, onyx, nova, shimmer)

    Returns:
        bytes: Audio data in MP3 format
    """
    try:
        if not api_key:
            raise Exception("OpenAI API key not configured")

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=1.0
        )

        return response.content

    except Exception as e:
        print(f"TTS generation failed: {str(e)}")
        return b""


def create_audio_message(content: str, voice: str = "alloy") -> cl.Audio:
    """
    Create Chainlit audio message with TTS

    Args:
        content (str): Text content to convert to speech
        voice (str): Voice to use for TTS

    Returns:
        cl.Audio: Chainlit audio element
    """
    try:
        # Limit text length for TTS (OpenAI has character limits)
        max_length = 4000
        if len(content) > max_length:
            # Truncate but try to end at a sentence
            truncated = content[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.8:  # If period is reasonably close to end
                content = truncated[:last_period + 1]
            else:
                content = truncated + "..."

        # Generate speech
        audio_data = generate_speech(content, voice)

        if not audio_data:
            return None

        # Create audio element
        audio = cl.Audio(
            content=audio_data,
            name="navada_response.mp3",
            display="inline",
            auto_play=False
        )

        return audio

    except Exception as e:
        print(f"Audio message creation failed: {str(e)}")
        return None


def benchmark_founder_idea(features: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Benchmark a founder's idea against dataset averages and percentiles.

    Args:
        features (dict): Founder's startup metrics
        df (pd.DataFrame): Dataset to benchmark against

    Returns:
        dict: Benchmarking results with percentiles and recommendations
    """
    results = {
        "metrics": {},
        "insights": [],
        "risk_level": "",
        "recommendations": []
    }

    # Calculate dataset statistics
    stats = {
        "funding": {
            "median": df["Funding_USD_M"].median(),
            "mean": df["Funding_USD_M"].mean(),
            "p20": df["Funding_USD_M"].quantile(0.2),
            "p80": df["Funding_USD_M"].quantile(0.8)
        },
        "burn_rate": {
            "median": df["Burn_Rate_Months"].median(),
            "mean": df["Burn_Rate_Months"].mean(),
            "p20": df["Burn_Rate_Months"].quantile(0.2),
            "p80": df["Burn_Rate_Months"].quantile(0.8)
        },
        "experience": {
            "median": df["Founders_Experience_Yrs"].median(),
            "mean": df["Founders_Experience_Yrs"].mean(),
            "p20": df["Founders_Experience_Yrs"].quantile(0.2),
            "p80": df["Founders_Experience_Yrs"].quantile(0.8)
        },
        "market": {
            "median": df["Market_Size_Bn"].median(),
            "mean": df["Market_Size_Bn"].mean(),
            "p20": df["Market_Size_Bn"].quantile(0.2),
            "p80": df["Market_Size_Bn"].quantile(0.8)
        }
    }

    # Benchmark funding
    funding = features.get('funding_usd_m', 3.0)
    funding_percentile = (df["Funding_USD_M"] < funding).mean() * 100
    results["metrics"]["funding"] = {
        "value": funding,
        "percentile": funding_percentile,
        "vs_median": funding / stats["funding"]["median"] if stats["funding"]["median"] > 0 else 0,
        "vs_mean": funding / stats["funding"]["mean"] if stats["funding"]["mean"] > 0 else 0
    }

    # Benchmark burn rate
    burn = features.get('burn_rate_months', 9.0)
    burn_percentile = (df["Burn_Rate_Months"] < burn).mean() * 100
    results["metrics"]["burn_rate"] = {
        "value": burn,
        "percentile": burn_percentile,
        "vs_median": burn / stats["burn_rate"]["median"] if stats["burn_rate"]["median"] > 0 else 0,
        "vs_mean": burn / stats["burn_rate"]["mean"] if stats["burn_rate"]["mean"] > 0 else 0
    }

    # Benchmark experience
    experience = features.get('team_experience_years', 3.0)
    exp_percentile = (df["Founders_Experience_Yrs"] < experience).mean() * 100
    results["metrics"]["experience"] = {
        "value": experience,
        "percentile": exp_percentile,
        "vs_median": experience / stats["experience"]["median"] if stats["experience"]["median"] > 0 else 0,
        "vs_mean": experience / stats["experience"]["mean"] if stats["experience"]["mean"] > 0 else 0
    }

    # Benchmark market size
    market = features.get('market_size_bn', 10.0)
    market_percentile = (df["Market_Size_Bn"] < market).mean() * 100
    results["metrics"]["market_size"] = {
        "value": market,
        "percentile": market_percentile,
        "vs_median": market / stats["market"]["median"] if stats["market"]["median"] > 0 else 0,
        "vs_mean": market / stats["market"]["mean"] if stats["market"]["mean"] > 0 else 0
    }

    # Generate insights
    if burn_percentile < 30:
        results["insights"].append(f"⚠️ Your burn rate ({burn} months) is in the **bottom 30%** - HIGH RISK! Most startups have longer runways.")
        results["recommendations"].append("Reduce burn rate or secure additional funding urgently")

    if burn_percentile > 70:
        results["insights"].append(f"✅ Your burn rate ({burn} months) is in the **top 30%** - well-managed cash flow")

    if funding_percentile < 30:
        results["insights"].append(f"⚠️ Your funding (${funding}M) is in the **bottom 30%** - may need more capital")
        results["recommendations"].append("Consider raising a larger round to extend runway")

    if funding_percentile > 70:
        results["insights"].append(f"✅ Your funding (${funding}M) is in the **top 30%** - strong financial position")

    if exp_percentile < 30:
        results["insights"].append(f"⚠️ Your team experience ({experience} years) is **below dataset median** - consider adding senior advisors")
        results["recommendations"].append("Add experienced advisors or co-founders to the team")

    if exp_percentile > 70:
        results["insights"].append(f"✅ Your team experience ({experience} years) is in the **top 30%** - strong foundation")

    if results["metrics"]["market_size"]["vs_median"] > 2:
        results["insights"].append(f"🚀 Your market size (${market}B) is **{results['metrics']['market_size']['vs_median']:.1f}× bigger** than average!")

    # Calculate risk level
    risk_score = 0
    if burn_percentile < 30: risk_score += 2
    if funding_percentile < 30: risk_score += 2
    if exp_percentile < 30: risk_score += 1

    if risk_score >= 3:
        results["risk_level"] = "HIGH RISK 🔴"
    elif risk_score >= 1:
        results["risk_level"] = "MODERATE RISK 🟡"
    else:
        results["risk_level"] = "LOW RISK 🟢"

    return results


def create_portfolio_heatmap(portfolio_df: pd.DataFrame) -> bytes:
    """
    Create a heatmap visualization of multiple startups' viability scores.

    Args:
        portfolio_df (pd.DataFrame): DataFrame with startup names and viability metrics

    Returns:
        bytes: PNG image of heatmap
    """
    # Calculate viability scores for each startup
    scores_data = []

    for _, row in portfolio_df.iterrows():
        # Calculate viability score for this startup
        features = {
            'funding_usd_m': row.get('Funding_USD_M', 3.0),
            'burn_rate_months': row.get('Burn_Rate_Months', 9.0),
            'team_experience_years': row.get('Founders_Experience_Yrs', 3.0),
            'market_size_bn': row.get('Market_Size_Bn', 10.0),
            'business_model_strength_1_5': row.get('Business_Model', 3),
            'moat_1_5': row.get('Moat', 3),
            'traction_mrr_k': row.get('Traction_MRR_K', 10),
            'growth_rate_pct': row.get('Growth_Rate_Pct', 5),
            'competition_intensity_1_5': row.get('Competition', 3)
        }

        result = viability_score(features)

        scores_data.append({
            'Startup': row['Startup'],
            'Overall Score': result['score'],
            'Runway': result['components']['runway'] * 100,
            'Experience': result['components']['experience'] * 100,
            'Market': result['components']['market'] * 100,
            'Traction': result['components']['traction'] * 100,
            'Growth': result['components']['growth'] * 100
        })

    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(scores_data)
    heatmap_matrix = heatmap_df.set_index('Startup')[['Overall Score', 'Runway', 'Experience', 'Market', 'Traction', 'Growth']]

    # Create heatmap with mobile-optimized size
    default_height = max(6, len(heatmap_df) * 0.5)
    figsize = get_mobile_optimized_figsize(10, default_height)
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap with custom colormap
    sns.heatmap(
        heatmap_matrix.T,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Score (0-100)'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_title('Portfolio Viability Heatmap\n🔴 Poor (0-40) | 🟡 Moderate (40-60) | 🟢 Strong (60-100)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Startups', fontsize=12)
    ax.set_ylabel('Metrics', fontsize=12)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig_to_bytes(fig)


# =============================
# THREAD MANAGEMENT FOR LANGSMITH
# =============================

def get_thread_history(thread_id: str, project_name: str) -> List[Dict[str, Any]]:
    """
    Retrieve conversation history from LangSmith for a specific thread.

    Args:
        thread_id: Unique identifier for the conversation thread
        project_name: LangSmith project name

    Returns:
        List of message dictionaries representing the conversation history
    """
    if not langsmith_client:
        return []

    try:
        # Filter runs by the specific thread and project
        filter_string = f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "{thread_id}"))'

        # Only grab the LLM runs
        runs = list(langsmith_client.list_runs(
            project_name=project_name,
            filter=filter_string,
            run_type="llm"
        ))

        if not runs:
            return []

        # Sort by start time to get chronological order
        runs = sorted(runs, key=lambda run: run.start_time)

        # Extract messages from runs
        messages = []
        for run in runs:
            if run.inputs and 'messages' in run.inputs:
                messages.extend(run.inputs['messages'])
            if run.outputs and 'choices' in run.outputs:
                if run.outputs['choices'] and run.outputs['choices'][0].get('message'):
                    messages.append(run.outputs['choices'][0]['message'])

        return messages
    except Exception as e:
        print(f"Error retrieving thread history: {e}")
        return []

@traceable(name="NAVADA Chat Pipeline")
def process_with_thread_context(
    question: str,
    session_id: str,
    get_chat_history: bool = True,
    persona: Optional[Dict[str, str]] = None
) -> str:
    """
    Process user message with thread context for continuity.

    Args:
        question: User's current question
        session_id: Thread/session identifier
        get_chat_history: Whether to retrieve and use conversation history
        persona: Current persona configuration

    Returns:
        AI response as string
    """
    langsmith_extra = {
        "project_name": LANGSMITH_PROJECT,
        "metadata": {"session_id": session_id}
    }

    messages = []

    # Retrieve conversation history if requested
    if get_chat_history and langsmith_client:
        try:
            historical_messages = get_thread_history(session_id, LANGSMITH_PROJECT)
            if historical_messages:
                messages.extend(historical_messages)
        except Exception as e:
            print(f"Could not retrieve history: {e}")

    # Add system prompt based on persona
    if persona:
        messages.insert(0, {
            "role": "system",
            "content": persona.get('system_prompt', '')
        })

    # Add current user question
    messages.append({"role": "user", "content": question})

    # Make API call with thread metadata
    try:
        if langsmith_client:
            # Use LangSmith thread pattern from documentation
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=800,
                temperature=0.7,
            )
        else:
            # Standard OpenAI call without LangSmith
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing message: {str(e)}"

# =============================
# CHAINLIT EVENT HANDLERS
# =============================

# Action handlers are not needed for this implementation
# Downloads will be handled through direct file sending

@cl.on_chat_start
async def start():
    """
    Initialize the chat session when a user first connects.

    This function runs once at the start of each new chat session and:
    1. Sets up chat settings with About section and quick actions
    2. Initializes thread/session tracking for LangSmith
    3. Sends a brief welcome message

    The settings panel (burger menu) contains detailed information
    about NAVADA's capabilities.

    Chainlit decorator: @cl.on_chat_start
    - Automatically called when a new chat begins
    - Async function for non-blocking UI operations
    """
    # -------------------------
    # INITIALIZE THREAD/SESSION TRACKING
    # -------------------------
    # Generate a unique session ID for this conversation thread
    session_id = str(uuid.uuid4())

    # Store session info in Chainlit user session for persistence
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("thread_metadata", {
        "session_id": session_id,
        "project_name": LANGSMITH_PROJECT,
        "start_time": pd.Timestamp.now().isoformat()
    })

    # Store in global thread sessions mapping
    THREAD_SESSIONS[session_id] = {
        "start_time": pd.Timestamp.now(),
        "messages": [],
        "persona": "founder"  # Default persona
    }

    # Log thread initialization
    if langsmith_client:
        print(f"🧵 Thread initialized: {session_id[:8]}...")

    # -------------------------
    # SETUP CHAT SETTINGS WITH ABOUT SECTION
    # -------------------------
    # Create chat settings with an "About" section accessible via burger menu
    about_content = (
        "# 👋 Welcome to NAVADA\n\n"
        "**NAVADA** (Startup Viability Agent) helps you analyze startup risk, funding, and failure patterns "
        "with **interactive charts and AI analysis**.\n\n"
        "## 🎭 Analysis Modes:\n\n"
        "💼 **Investor Mode** - VC perspective focused on ROI and exit strategies\n"
        "🚀 **Founder Mode** - Entrepreneur perspective focused on execution and growth\n"
        "🇬🇧 **UK Economist Mode** - Economic analysis perspective for UK markets\n\n"
        "## 📊 Advanced Charts:\n\n"
        "🔹 **Growth Trajectory** - MRR growth patterns vs company age\n"
        "🔹 **Team Performance** - Team size vs founder experience matrix\n"
        "🔹 **Market Opportunity** - Market size vs competition analysis\n"
        "🔹 **Funding Efficiency** - Capital efficiency and ROI analysis\n"
        "🔹 **Stage Progression** - Funding stages vs failure rates\n"
        "🔹 **Risk Assessment** - Comprehensive risk radar chart\n"
        "🔹 **UK Economic Dashboard** - Macroeconomic indicators and regional analysis\n\n"
        "## 📈 Interactive Tools:\n\n"
        "🔹 **Interactive Scatter** - Dynamic correlations and filtering\n"
        "🔹 **Sector Dashboard** - Multi-dimensional sector analysis\n"
        "🔹 **Interactive Timeline** - Failure patterns over time\n\n"
        "## 🤖 AI-Powered Features:\n\n"
        "🔹 **assess idea** - Interactive viability scoring with 24 data points\n"
        "🔹 **benchmark** - Compare your startup against 24 successful companies\n"
        "🔹 **portfolio** - Analyze multiple startups with heatmap visualization\n"
        "🔹 **insights** - AI-powered risk assessment and opportunities\n"
        "🔹 **questions** - Guided questions based on your current mode\n"
        "🔹 **macro analysis** - UK macroeconomic impact assessment\n\n"
        "## 🔍 Internet Search:\n\n"
        "🔹 **search [query]** - Get up-to-date market intelligence and trends\n"
        "🔹 **latest news** - Current developments in startup ecosystem\n"
        "🔹 **current trends** - Market shifts and opportunities\n"
        "🔹 Auto-triggered for questions about recent events or market updates\n\n"
        "## 📥 Download & Export:\n\n"
        "🔹 **Charts** - Download any generated chart as PNG with built-in download buttons\n"
        "🔹 **export data** / **download csv** - Export complete dataset as CSV\n"
        "🔹 **export json** / **download json** - Export complete dataset as JSON\n"
        "🔹 **Data Tables** - Download chart data as CSV alongside visualizations\n\n"
        "## 💬 Get Started:\n\n"
        "• Type **'investor mode'**, **'founder mode'**, or **'economist mode'** to set your perspective\n"
        "• Type **'questions'** to get guided analysis questions\n"
        "• Ask: \"Which chart should I look at first?\"\n"
        "• Try: \"Show me funding efficiency\" or \"Risk assessment\"\n\n"
        "---\n\n"
        "**Ready to start?** Choose your mode and dive into comprehensive startup analysis!"
    )

    # Store TTS setting in user session (default off)
    cl.user_session.set("tts_enabled", False)

    # -------------------------
    # SEND BRIEF WELCOME MESSAGE
    # -------------------------
    # Send clear welcome message with instructions
    welcome = """**🚀 NAVADA - Startup Viability Agent**

**Quick Start:**
• Type **'investor mode'**, **'founder mode'**, or **'economist mode'** to begin
• Try **'voice on'** to enable text-to-speech
• Ask **'help'** for available commands

**Analysis Modes:**
💼 **Investor Mode** - VC perspective, ROI analysis, portfolio optimization
🚀 **Founder Mode** - Entrepreneur focus, execution strategies, growth tactics
🇬🇧 **Economist Mode** - UK macroeconomic analysis, regional factors, policy impacts

**Popular Commands:**
• **'dashboard'** - Interactive analytics dashboard
• **'interactive scatter'** - Dynamic scatter plots with hover details
• **'correlation heatmap'** - Multi-dimensional relationship analysis
• **'compare startups'** - Side-by-side startup comparisons
• **'filter dashboard'** - Apply filters to drill down into data
• **'benchmark'** - Compare against successful startups
• **'macro analysis'** - UK economic impact assessment
• **'search [query]'** - Get real-time market intelligence

Ready to analyze your startup? Choose your mode to start!"""

    # Send the welcome message asynchronously to the UI
    await cl.Message(content=welcome).send()

    # -------------------------
    # ADD ELEVENLABS CONVAI WIDGET
    # -------------------------
    # Add ElevenLabs voice agent widget for voice interactions
    elevenlabs_widget = """
    <elevenlabs-convai agent-id="agent_6501k5q5hn4zf9eteg70jwra0ekp"></elevenlabs-convai>
    <script src="https://unpkg.com/@elevenlabs/convai-widget-embed" async type="text/javascript"></script>
    """

    # Send the ElevenLabs widget information
    await cl.Message(
        content="🎙️ **Voice Agent Available** - ElevenLabs ConvAI widget configured with agent ID: `agent_6501k5q5hn4zf9eteg70jwra0ekp`"
    ).send()

    # Try to inject the widget via Chainlit's HTML element
    try:
        await cl.Html(content=elevenlabs_widget, display="page").send()
    except Exception as e:
        print(f"⚠️ Could not inject ElevenLabs widget: {e}")
        # Alternative: provide instructions to user
        await cl.Message(
            content="**Manual Setup:** To enable voice chat, add this HTML to your page:\n```html\n" + elevenlabs_widget + "\n```"
        ).send()


async def generate_speech(text: str) -> cl.Audio:
    """Generate speech from text using OpenAI TTS."""
    try:
        # Generate speech using OpenAI TTS
        response = base_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text[:4000]  # Limit text length
        )

        # Save audio to bytes
        audio_bytes = response.content

        # Create Chainlit audio element
        audio_element = cl.Audio(
            content=audio_bytes,
            mime="audio/mpeg",
            display="inline"
        )

        return audio_element
    except Exception as e:
        print(f"⚠️ TTS Error: {e}")
        return None

# Voice commands are handled in the main message handler


# =============================
# INPUT HELPER FUNCTIONS
# =============================

async def ask_float(prompt: str, default: float) -> float:
    """
    Prompt user for a floating-point number input with a default value.

    This helper function handles the complexity of asking for numeric input
    in Chainlit, including error handling and default values.

    Args:
        prompt (str): Question to ask the user
        default (float): Default value if user presses Enter or input is invalid

    Returns:
        float: The user's input as a float, or the default value

    Error handling:
        - Empty input → returns default
        - Invalid input (non-numeric) → returns default
        - Timeout (600 seconds) → returns default

    Example:
        >>> funding = await ask_float("Enter funding amount:", 3.0)
        # User sees: "Enter funding amount: (default 3.0):"
        # If user enters "5.5" → returns 5.5
        # If user presses Enter → returns 3.0
    """
    # Send question to user with timeout of 600 seconds (10 minutes)
    msg = await cl.AskUserMessage(
        content=f"{prompt} (default {default}):",
        timeout=600
    ).send()

    # Try to parse the response as a float
    try:
        # Check if message exists and has 'output' field
        if msg and msg.get('output'):
            return float(msg['output'])  # Convert string to float
        return float(default)  # No input provided, use default
    except Exception:
        # If any error occurs (ValueError, AttributeError, etc.), use default
        return float(default)


async def ask_int(prompt: str, default: int, mi: int = 1, ma: int = 5) -> int:
    """
    Prompt user for an integer input within a specified range.

    This helper function asks for integer input and enforces min/max bounds.
    Commonly used for rating scales (1-5).

    Args:
        prompt (str): Question to ask the user
        default (int): Default value if input is invalid or empty
        mi (int): Minimum allowed value (default: 1)
        ma (int): Maximum allowed value (default: 5)

    Returns:
        int: The user's input clamped to [mi, ma], or default value

    Behavior:
        - Input is automatically clamped to the min/max range
        - Example: If range is 1-5 and user enters 7, returns 5
        - Example: If range is 1-5 and user enters 0, returns 1

    Example:
        >>> rating = await ask_int("Rate your moat:", 3, mi=1, ma=5)
        # User sees: "Rate your moat: [1..5] (default 3):"
        # If user enters "4" → returns 4
        # If user enters "10" → returns 5 (clamped to max)
        # If user enters "0" → returns 1 (clamped to min)
    """
    # Send question to user showing the valid range
    msg = await cl.AskUserMessage(
        content=f"{prompt} [{mi}..{ma}] (default {default}):",
        timeout=600
    ).send()

    # Try to parse the response as an integer and clamp to range
    try:
        if msg and msg.get('output'):
            val = int(msg['output'])  # Convert string to int
        else:
            val = default  # No input provided, use default

        # Clamp value to [mi, ma] range
        # max(mi, val) ensures val >= mi
        # min(ma, ...) ensures result <= ma
        return max(mi, min(ma, val))
    except Exception:
        # If any error occurs, use default value
        return default

# =============================
# CSV UPLOAD HANDLER
# =============================

async def handle_csv_upload():
    """
    Handle CSV file upload from user to replace the current dataset.

    This function manages the entire CSV upload workflow:
    1. Prompts user to upload a CSV file
    2. Validates file type and size
    3. Parses CSV into DataFrame
    4. Calculates derived columns if possible
    5. Returns the new DataFrame or None on failure

    Returns:
        pd.DataFrame or None: New dataset if upload succeeds, None otherwise

    File requirements:
        - Format: CSV (comma-separated values)
        - Max size: 5 MB
        - Recommended columns: Startup, Funding_USD_M, Burn_Rate_Months,
          Failed, Country, Sector, Founders_Experience_Yrs, Market_Size_Bn

    Automatic processing:
        - If Funding_USD_M and Burn_Rate_Months exist, Est_Failure_Year
          is automatically calculated
        - Missing columns are allowed but may affect functionality

    User feedback:
        - Confirmation message showing filename and row count
        - Error message if upload fails or is cancelled
    """
    # -------------------------
    # PROMPT FOR FILE UPLOAD
    # -------------------------
    # Request CSV file from user with constraints
    files = await cl.AskFileMessage(
        content="Upload a CSV with columns like Startup, Funding_USD_M, Burn_Rate_Months, Failed, Country, Sector",
        accept=["text/csv"],  # Only allow CSV files
        max_size_mb=5,  # Limit file size to 5 MB
        timeout=600  # 10 minute timeout
    ).send()

    # -------------------------
    # HANDLE CANCELLATION
    # -------------------------
    # If user cancels or no file is provided
    if not files:
        await cl.Message(content="No file received. Keeping current dataset.").send()
        return None  # Return None to indicate no change

    # -------------------------
    # PARSE CSV FILE
    # -------------------------
    # Extract first file from the list (usually only one file uploaded)
    f = files[0]

    # Read CSV from bytes into pandas DataFrame
    # f.content is bytes, so wrap in BytesIO for pandas
    new_df = pd.read_csv(io.BytesIO(f.content))

    # -------------------------
    # CALCULATE DERIVED COLUMNS
    # -------------------------
    # Check if required columns exist for failure year calculation
    # Using set intersection to check if both columns are present
    if {"Funding_USD_M", "Burn_Rate_Months"}.issubset(new_df.columns):
        # Calculate estimated failure year using same formula as original data
        new_df["Est_Failure_Year"] = FUNDING_YEAR + (
            new_df["Funding_USD_M"] / new_df["Burn_Rate_Months"]
        )

    # -------------------------
    # CONFIRM UPLOAD SUCCESS
    # -------------------------
    # Send confirmation message showing filename and row count
    await cl.Message(
        content=f"Loaded `{f.name}` with {len(new_df)} rows."
    ).send()

    # Return the new DataFrame to replace the global dataset
    return new_df

# =============================
# MAIN MESSAGE HANDLER
# =============================

@cl.on_message
@traceable(
    name="NAVADA Message Handler",
    run_type="chain",
    tags=["navada", "message-handler", "conversation-entry"],
    metadata={"handler_type": "chainlit_message", "app_version": "2.0.0"}
)
async def main(message: cl.Message):
    """
    Process every user message and route to appropriate handler.

    This is the main message processing function that runs whenever a user
    sends a message. It implements a routing pattern to handle different
    types of requests:

    1. Chart generation commands (timeline, funding vs burn)
    2. CSV upload requests
    3. Viability assessment requests
    4. General Q&A using GPT-4

    Message routing logic:
    ----------------------
    - Show thinking indicator for user feedback
    - Lowercase the input for case-insensitive matching
    - Check for specific keywords to determine intent
    - Execute corresponding handler and return
    - If no specific pattern matches, fallback to AI chat

    Args:
        message (cl.Message): Chainlit message object containing:
            - content: User's text input
            - author: User identifier
            - Other metadata

    Global variables:
        df (pd.DataFrame): Can be modified by CSV upload handler

    Chainlit decorator: @cl.on_message
    - Automatically called for every user message
    - Async function for non-blocking operations
    """
    global df  # Allow modification of global dataset

    # -------------------------
    # AUTHENTICATION CHECK (DISABLED FOR TESTING)
    # -------------------------
    # TESTING MODE: Skip authentication for faster testing
    # auth_status = check_user_authentication()

    # Set default test user for development
    auth_status = {
        "authenticated": True,
        "username": "test_user",
        "user_id": "test_123",
        "email": "test@navada.ai",
        "subscription_tier": "free"
    }

    # Allow login and register commands even when not authenticated
    user_input_raw = message.content.strip()
    user_input = user_input_raw.lower()

    # Handle authentication commands (commented out for testing)
    # if user_input.startswith("login "):
    #     await handle_login_command(user_input_raw)
    #     return

    # if user_input.startswith("register "):
    #     await handle_register_command(user_input_raw)
    #     return

    # If not authenticated and not using auth commands, show login form (disabled)
    # if not auth_status["authenticated"]:
    #     await show_login_form()
    #     return

    # -------------------------
    # THINKING INDICATOR & TIMESTAMP
    # -------------------------
    # Show thinking indicator to provide immediate user feedback
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # Show brief thinking message (will be removed after processing)
    thinking_msg = cl.Message(content=f"🤔 Thinking... ({timestamp})")
    await thinking_msg.send()

    # -------------------------
    # NORMALIZE INPUT
    # -------------------------
    # Convert to lowercase and strip whitespace for consistent matching
    user_input = message.content.strip().lower()

    # =============================
    # ROUTE 0: VOICE COMMANDS
    # =============================
    if user_input in ["voice on", "voice enable", "tts on", "speech on", "audio on"]:
        cl.user_session.set("tts_enabled", True)
        await cl.Message(content="🔊 **Voice enabled!** AI responses will now include audio.").send()
        return

    if user_input in ["voice off", "voice disable", "tts off", "speech off", "audio off"]:
        cl.user_session.set("tts_enabled", False)
        await cl.Message(content="🔇 **Voice disabled.** AI responses will be text only.").send()
        return

    # =============================
    # ROUTE 0.5: MATHEMATICAL ANALYSIS MODE
    # =============================
    if "math mode" in user_input or "analysis mode" in user_input:
        await thinking_msg.remove()

        # Create math-enabled analysis environment
        math_context = {
            'df': df,
            'np': np,
            'stats': stats,
            'current_startup': cl.user_session.get('selected_startup')
        }

        await cl.Message(
            content="## 🧮 Mathematical Analysis Mode\n\n"
                    "You can now perform complex calculations:\n\n"
                    "**Examples:**\n"
                    "• `calculate IRR for 5x return in 7 years`\n"
                    "• `project revenue with 20% monthly growth`\n"
                    "• `simulate 1000 scenarios for exit`\n"
                    "• `optimize burn rate for 18 month runway`\n\n"
                    "Type your calculation or 'exit math mode' to return."
        ).send()

        cl.user_session.set("math_mode", True)
        return

    # Check if user is in math mode
    if cl.user_session.get("math_mode", False):
        await thinking_msg.remove()

        if user_input in ["exit math mode", "exit", "return", "back"]:
            cl.user_session.set("math_mode", False)
            await cl.Message(content="🔄 **Exited math mode.** Back to regular analysis.").send()
            return

        # Process math command
        math_context = {
            'df': df,
            'np': np,
            'stats': stats,
            'current_startup': cl.user_session.get('selected_startup')
        }

        result = await process_math_command(user_input, math_context)
        await cl.Message(content=result).send()
        return

    # =============================
    # ROUTE 0.7: IMAGE GENERATION
    # =============================
    if detect_image_request(user_input):
        await thinking_msg.remove()

        # Extract the image prompt from user input
        image_prompt = extract_image_prompt(user_input_raw)

        # Show image generation message
        generation_msg = cl.Message(content=f"🎨 **Generating image:** {image_prompt}\n\nThis may take a few moments...")
        await generation_msg.send()

        # Generate the image
        result = await generate_image(image_prompt)

        if result["success"]:
            # Create image element from URL
            image_element = cl.Image(
                url=result["image_url"],
                name=f"generated_image_{timestamp.replace(':', '')}.png",
                display="inline"
            )

            # Update message with success
            success_content = f"✅ **Image generated successfully!**\n\n"
            success_content += f"**Original prompt:** {result['original_prompt']}\n"
            if result.get('revised_prompt') and result['revised_prompt'] != result['original_prompt']:
                success_content += f"**DALL-E revised prompt:** {result['revised_prompt']}\n"
            success_content += f"**Size:** {result['size']} | **Quality:** {result['quality']}"

            await cl.Message(content=success_content, elements=[image_element]).send()

            # Log for conversation tracking
            auth_manager.save_conversation(
                user_id=auth_status.get("user_id", "test_123"),
                chainlit_session_id=cl.user_session.get("session_id", "unknown"),
                role="user",
                content=f"Image generation request: {image_prompt}",
                metadata={"action_type": "image_generation", "prompt": image_prompt}
            )

            auth_manager.save_conversation(
                user_id=auth_status.get("user_id", "test_123"),
                chainlit_session_id=cl.user_session.get("session_id", "unknown"),
                role="assistant",
                content=f"Generated image: {result['image_url']}",
                metadata={"action_type": "image_generated", "image_url": result["image_url"]}
            )

        else:
            # Handle error
            error_content = f"❌ **Image generation failed**\n\n"
            error_content += f"**Error:** {result['error']}\n"
            error_content += f"**Prompt:** {result['original_prompt']}\n\n"
            error_content += "Please try again with a different prompt or check your API configuration."

            await cl.Message(content=error_content).send()

        return

    # =============================
    # ROUTE 0.8: SWOT ANALYSIS
    # =============================
    if "swot" in user_input or ("swot mode" in user_input) or ("analyze swot" in user_input):
        await thinking_msg.remove()

        # Show SWOT analysis message
        analysis_msg = cl.Message(content="📊 **Generating SWOT Analysis...**\n\nAnalyzing strengths, weaknesses, opportunities, and threats...")
        await analysis_msg.send()

        # Gather startup context from user session or create generic context
        startup_context = {
            "persona": cl.user_session.get("persona", "founder"),
            "selected_startup": cl.user_session.get("selected_startup"),
            "session_id": cl.user_session.get("session_id", "unknown")
        }

        # Add any available startup data from the dataframe
        startup_data = {}
        if df is not None and not df.empty:
            # Get a sample of startups for context
            sample_data = df.head(3).to_dict('records')
            startup_data["sample_startups"] = sample_data
            startup_data["total_startups"] = len(df)
            startup_data["sectors"] = df['Sector'].value_counts().head(5).to_dict() if 'Sector' in df.columns else {}

        # Generate SWOT analysis using AI
        swot_analysis = await generate_swot_analysis(startup_data, user_input_raw)

        # Create SWOT visualization
        try:
            swot_chart_png = plot_swot_matrix(swot_analysis)

            # Create SWOT DataFrame for download
            swot_df = swot_analysis.to_dataframe()

            # Send the comprehensive SWOT analysis
            swot_summary = swot_analysis.summary()

            await cl.Message(content=swot_summary).send()

            # Send the visual SWOT matrix
            await send_chart_with_download(
                png_data=swot_chart_png,
                filename=f"swot_analysis_{timestamp.replace(':', '')}.png",
                description="📊 **SWOT Analysis Matrix** - Visual representation of your startup analysis",
                csv_data=swot_df
            )

            # Log for conversation tracking
            auth_manager.save_conversation(
                user_id=auth_status.get("user_id", "test_123"),
                chainlit_session_id=cl.user_session.get("session_id", "unknown"),
                role="user",
                content=f"SWOT analysis request: {user_input_raw}",
                metadata={"action_type": "swot_analysis", "context": startup_context}
            )

            auth_manager.save_conversation(
                user_id=auth_status.get("user_id", "test_123"),
                chainlit_session_id=cl.user_session.get("session_id", "unknown"),
                role="assistant",
                content="Generated comprehensive SWOT analysis with matrix visualization",
                metadata={
                    "action_type": "swot_generated",
                    "strengths_count": len(swot_analysis.strengths),
                    "weaknesses_count": len(swot_analysis.weaknesses),
                    "opportunities_count": len(swot_analysis.opportunities),
                    "threats_count": len(swot_analysis.threats)
                }
            )

        except Exception as e:
            # Fallback to text-only SWOT if visualization fails
            error_msg = f"⚠️ **SWOT Analysis Generated** (Chart error: {str(e)})\n\n"
            error_msg += swot_analysis.summary()
            await cl.Message(content=error_msg).send()

        return

    # =============================
    # ROUTE 1: FAILURE TIMELINE CHART
    # =============================
    if "timeline" in user_input:
        # Show loading message while generating chart
        msg = cl.Message(content="📊 Generating failure timeline chart...")
        await msg.send()

        # Generate the chart PNG
        png = plot_failure_timeline(df)

        # Remove loading message
        await msg.remove()

        # Send chart with download options
        # Select available columns for CSV export
        preferred_cols = ['Company', 'Total Funding', 'Monthly Burn Rate', 'Estimated Runway (Months)']
        available_cols = [col for col in preferred_cols if col in df.columns]

        # If no preferred columns exist, use first 4 columns or all if less than 4
        if not available_cols:
            available_cols = df.columns.tolist()[:4]

        await send_chart_with_download(
            png_data=png,
            filename="failure_timeline.png",
            description=(
                "### 📈 Estimated Failure Timeline\n\n"
                "This chart shows when each startup is projected to fail "
                "based on their funding and burn rate."
            ),
            csv_data=df[available_cols] if available_cols else df
        )

        return  # Exit handler, don't process further

    # =============================
    # ROUTE 2: FUNDING VS BURN CHART
    # =============================
    # Check for multiple possible phrasings
    if "funding vs burn" in user_input or (
        "funding" in user_input and "burn" in user_input and "vs" in user_input
    ):
        # Show loading message
        msg = cl.Message(content="📊 Generating funding vs burn chart...")
        await msg.send()

        # Generate scatter plot
        png = plot_funding_vs_burn(df)

        # Remove loading message
        await msg.remove()

        # Send descriptive text with legend explanation
        text_msg = cl.Message(
            content=(
                "### 💰 Funding vs Burn Rate Analysis\n\n"
                "**Green** = Successful | **Red** = Failed\n"
                "Each shape represents a different sector."
            )
        )
        await text_msg.send()

        # Attach chart image
        image = cl.Image(content=png, name="funding_vs_burn.png", display="inline")
        await image.send(for_id=text_msg.id)

        return  # Exit handler

    # =============================
    # ROUTE 3: ADDITIONAL CHART COMMANDS
    # =============================
    # Sector comparison chart
    if "sector" in user_input and ("compare" in user_input or "comparison" in user_input or "chart" in user_input):
        msg = cl.Message(content="📊 Generating sector comparison chart...")
        await msg.send()
        png = plot_sector_comparison(df)
        await msg.remove()
        text_msg = cl.Message(content="### 🏭 Sector Comparison\n\nAverage funding by industry sector.")
        await text_msg.send()
        image = cl.Image(content=png, name="sector_comparison.png", display="inline")
        await image.send(for_id=text_msg.id)
        return

    # Failure rate by country
    if ("failure" in user_input or "fail" in user_input) and "country" in user_input:
        msg = cl.Message(content="📊 Generating failure rate by country chart...")
        await msg.send()
        png = plot_failure_rate_by_country(df)
        await msg.remove()
        text_msg = cl.Message(content="### 🌍 Failure Rate by Country\n\nPercentage of failed startups per country.")
        await text_msg.send()
        image = cl.Image(content=png, name="failure_rate_country.png", display="inline")
        await image.send(for_id=text_msg.id)
        return

    # Experience vs success
    if "experience" in user_input and ("success" in user_input or "funding" in user_input or "chart" in user_input):
        msg = cl.Message(content="📊 Generating experience vs success chart...")
        await msg.send()
        png = plot_experience_vs_success(df)
        await msg.remove()
        text_msg = cl.Message(content="### 👥 Experience vs Success\n\nRelationship between founder experience, funding, and outcome.")
        await text_msg.send()
        image = cl.Image(content=png, name="experience_success.png", display="inline")
        await image.send(for_id=text_msg.id)
        return

    # =============================
    # ROUTE 4: CSV UPLOAD
    # =============================
    if "upload csv" in user_input or "load csv" in user_input:
        # Call upload handler and get new DataFrame (or None)
        new_df = await handle_csv_upload()

        # If upload succeeded, replace global dataset
        if new_df is not None:
            df = new_df

        return  # Exit handler

    # =============================
    # ROUTE 4: VIABILITY ASSESSMENT
    # =============================
    # Check for various phrasings of assessment request
    if "assess idea" in user_input or "new idea" in user_input or "viability" in user_input:
        # -------------------------
        # INTRODUCTION
        # -------------------------
        # Explain the assessment process
        await cl.Message(
            content=(
                "## 🎯 Startup Viability Assessment\n\n"
                "Great! Let's evaluate your startup idea. "
                "I'll ask you **9 quick questions** to calculate a "
                "comprehensive viability score.\n\n"
                "*Press Enter to use default values, or type your answer.*"
            )
        ).send()

        # -------------------------
        # COLLECT INPUT DATA
        # -------------------------
        # Ask 9 questions to gather all required features
        # Each question has a sensible default value

        funding = await ask_float("Funding (USD Millions)", 3.0)
        burn = await ask_float(
            "Burn rate (months of runway if spending 1M/year ≈ 83k/month)", 9.0
        )
        expy = await ask_float("Team experience (years, average)", 3.0)
        market = await ask_float("Market size (Billions USD)", 15.0)
        bm = await ask_int("Business model strength (1=weak..5=excellent)", 3)
        moat = await ask_int("Moat/defensibility (1..5)", 3)
        mrrk = await ask_float("Current MRR (in $k)", 10.0)
        growth = await ask_float("Monthly growth rate (%)", 6.0)
        comp = await ask_int("Competition intensity (1=low..5=very high)", 3)

        # -------------------------
        # PACKAGE FEATURES
        # -------------------------
        # Create dictionary matching the viability_score function signature
        feats = {
            "funding_usd_m": funding,
            "burn_rate_months": burn,
            "team_experience_years": expy,
            "market_size_bn": market,
            "business_model_strength_1_5": bm,
            "moat_1_5": moat,
            "traction_mrr_k": mrrk,
            "growth_rate_pct": growth,
            "competition_intensity_1_5": comp
        }

        # -------------------------
        # CALCULATE SCORE
        # -------------------------
        # Call viability scoring model
        result = viability_score(feats)

        # -------------------------
        # DISPLAY GAUGE CHART
        # -------------------------
        # Generate and display visual score gauge
        gauge_png = plot_viability_gauge(result["score"])
        gauge_msg = cl.Message(content="\n## 📊 Your Viability Score")
        await gauge_msg.send()

        # Attach gauge image to message
        gauge_image = cl.Image(
            content=gauge_png, name="viability_score.png", display="inline"
        )
        await gauge_image.send(for_id=gauge_msg.id)

        # -------------------------
        # DISPLAY DETAILED RESULTS
        # -------------------------
        # Interpret score with color-coded assessment
        score_interpretation = (
            "🟢 Strong" if result['score'] >= 60
            else "🟡 Moderate" if result['score'] >= 40
            else "🔴 Weak"
        )

        # Format comprehensive summary with all metrics
        summary = (
            f"### Overall Assessment: {score_interpretation}\n\n"
            f"**Final Score:** {result['score']:.1f}/100\n\n"
            f"#### 📈 Key Metrics:\n"
            f"• **Estimated Runway:** ~{result['survival_months']:.1f} months\n"
            f"• **Projected Failure Year:** {result['est_failure_year']:.2f} "
            f"(funded {FUNDING_YEAR})\n\n"
            f"#### 🔍 Score Breakdown (0-1 scale):\n"
            f"• Runway: {result['components']['runway']:.2f} | "
            f"Experience: {result['components']['experience']:.2f}\n"
            f"• Market: {result['components']['market']:.2f} | "
            f"Business Model: {result['components']['business_model']:.2f}\n"
            f"• Moat: {result['components']['moat']:.2f} | "
            f"Traction: {result['components']['traction']:.2f}\n"
            f"• Growth: {result['components']['growth']:.2f} | "
            f"Competition: {result['components']['competition']:.2f}\n"
        )
        await cl.Message(content=summary).send()

        # -------------------------
        # DISPLAY RECOMMENDATIONS
        # -------------------------
        # Show actionable tips based on weaknesses
        if result["tips"]:
            # Format tips as bullet list
            tips_text = (
                "### 💡 Recommended Actions\n\n" +
                "\n".join([f"• {tip}" for tip in result["tips"]])
            )
            await cl.Message(content=tips_text).send()
        else:
            # If no tips, startup looks strong
            await cl.Message(
                content=(
                    "### ✅ Looking Good!\n\n"
                    "• Keep executing—your foundations look solid.\n"
                    "• Focus on consistent growth and customer acquisition."
                )
            ).send()

        return  # Exit handler

    # =============================
    # ROUTE 6: BENCHMARK IDEA
    # =============================
    if "benchmark idea" in user_input or "benchmark my idea" in user_input or "compare my idea" in user_input:
        await cl.Message(content="## 🎯 Benchmark Your Startup Idea\n\nI'll compare your metrics against our dataset to see how you stack up!").send()

        # Collect startup metrics from user
        funding = await ask_float("Your funding amount (USD millions)", 3.0)
        burn = await ask_float("Your burn rate (months)", 9.0)
        expy = await ask_float("Your team's average experience (years)", 3.0)
        market = await ask_float("Your target market size (billions USD)", 10.0)

        # Package features for benchmarking
        features = {
            "funding_usd_m": funding,
            "burn_rate_months": burn,
            "team_experience_years": expy,
            "market_size_bn": market
        }

        # Run benchmarking analysis
        benchmark_results = benchmark_founder_idea(features, df)

        # Display benchmarking results
        results_text = f"## 📊 Benchmarking Results\n\n"
        results_text += f"**Risk Level:** {benchmark_results['risk_level']}\n\n"

        results_text += "### 📈 How You Compare:\n\n"

        # Display key insights
        for insight in benchmark_results['insights']:
            results_text += f"• {insight}\n"

        results_text += "\n### 📊 Detailed Metrics:\n\n"

        # Funding percentile
        funding_data = benchmark_results['metrics']['funding']
        results_text += f"**Funding:** ${funding_data['value']}M (percentile: {funding_data['percentile']:.0f}%)\n"
        results_text += f"• {funding_data['vs_median']:.1f}× the median startup\n\n"

        # Burn rate percentile
        burn_data = benchmark_results['metrics']['burn_rate']
        results_text += f"**Burn Rate:** {burn_data['value']} months (percentile: {burn_data['percentile']:.0f}%)\n"
        results_text += f"• {burn_data['vs_median']:.1f}× the median startup\n\n"

        # Experience percentile
        exp_data = benchmark_results['metrics']['experience']
        results_text += f"**Experience:** {exp_data['value']} years (percentile: {exp_data['percentile']:.0f}%)\n"
        results_text += f"• {exp_data['vs_median']:.1f}× the median startup\n\n"

        # Market size percentile
        market_data = benchmark_results['metrics']['market_size']
        results_text += f"**Market Size:** ${market_data['value']}B (percentile: {market_data['percentile']:.0f}%)\n"
        results_text += f"• {market_data['vs_median']:.1f}× the median startup\n\n"

        # Recommendations
        if benchmark_results['recommendations']:
            results_text += "### 💡 Recommendations:\n\n"
            for rec in benchmark_results['recommendations']:
                results_text += f"• {rec}\n"

        await cl.Message(content=results_text).send()

        return  # Exit handler

    # =============================
    # ROUTE 7: PORTFOLIO MODE
    # =============================
    if "portfolio" in user_input and ("mode" in user_input or "analysis" in user_input or "analyze" in user_input or user_input.strip() == "portfolio"):
        # Remove thinking indicator
        await thinking_msg.remove()

        await cl.Message(content="## 📊 Portfolio Analysis Mode\n\nI'll create a comprehensive heatmap of all startups in your dataset!").send()

        # Show loading message
        msg = cl.Message(content="🔥 Generating portfolio heatmap... Calculating viability scores for all startups.")
        await msg.send()

        try:
            # Generate portfolio heatmap
            heatmap_bytes = create_portfolio_heatmap(df)

            # Remove loading message
            await msg.remove()

            # Send heatmap description
            desc_msg = cl.Message(
                content="### 🔥 Portfolio Viability Heatmap\n\n"
                        f"**Analysis of {len(df)} startups across 6 key metrics:**\n"
                        "• **Overall Score** - Combined viability (0-100)\n"
                        "• **Runway** - Financial sustainability\n"
                        "• **Experience** - Team expertise\n"
                        "• **Market** - Market opportunity\n"
                        "• **Traction** - Current momentum\n"
                        "• **Growth** - Growth trajectory\n\n"
                        "**Color Guide:** 🔴 Poor (0-40) | 🟡 Moderate (40-60) | 🟢 Strong (60-100)"
            )
            await desc_msg.send()

            # Attach heatmap image
            heatmap_image = cl.Image(
                content=heatmap_bytes, name="portfolio_heatmap.png", display="inline"
            )
            await heatmap_image.send(for_id=desc_msg.id)

            # Calculate and show investment recommendations
            portfolio_scores = []
            for _, row in df.iterrows():
                features = {
                    'funding_usd_m': row['Funding_USD_M'],
                    'burn_rate_months': row['Burn_Rate_Months'],
                    'team_experience_years': row['Founders_Experience_Yrs'],
                    'market_size_bn': row['Market_Size_Bn'],
                    'business_model_strength_1_5': 3,
                    'moat_1_5': 3,
                    'traction_mrr_k': 10,
                    'growth_rate_pct': 5,
                    'competition_intensity_1_5': 3
                }
                score = viability_score(features)['score']
                portfolio_scores.append((row['Startup'], score))

            # Sort by score
            portfolio_scores.sort(key=lambda x: x[1], reverse=True)

            # Create recommendations
            reco_text = "### 🎯 Investment Recommendations:\n\n"
            reco_text += "**🟢 INVEST (Score ≥ 60):**\n"
            invest_list = [f"• {name} ({score:.1f})" for name, score in portfolio_scores if score >= 60]
            if invest_list:
                reco_text += "\n".join(invest_list) + "\n\n"
            else:
                reco_text += "• None in this category\n\n"

            reco_text += "**🟡 MONITOR (Score 40-59):**\n"
            monitor_list = [f"• {name} ({score:.1f})" for name, score in portfolio_scores if 40 <= score < 60]
            if monitor_list:
                reco_text += "\n".join(monitor_list) + "\n\n"
            else:
                reco_text += "• None in this category\n\n"

            reco_text += "**🔴 PASS (Score < 40):**\n"
            pass_list = [f"• {name} ({score:.1f})" for name, score in portfolio_scores if score < 40]
            if pass_list:
                reco_text += "\n".join(pass_list) + "\n\n"
            else:
                reco_text += "• None in this category\n\n"

            # Portfolio statistics
            avg_score = sum(score for _, score in portfolio_scores) / len(portfolio_scores)
            high_performers = len([s for _, s in portfolio_scores if s >= 60])

            reco_text += f"**📊 Portfolio Stats:**\n"
            reco_text += f"• Average Score: {avg_score:.1f}/100\n"
            reco_text += f"• High Performers: {high_performers}/{len(df)} ({high_performers/len(df)*100:.1f}%)\n"
            reco_text += f"• Success Rate: {((df['Failed'] == 0).sum()/len(df)*100):.1f}%"

            await cl.Message(content=reco_text).send()

        except Exception as e:
            await msg.remove()
            await cl.Message(
                content=f"❌ Error generating portfolio analysis: {str(e)}\n\n"
                        f"Please try again or contact support if the issue persists."
            ).send()

        return  # Exit handler

    # =============================
    # ROUTE: UK MACRO ANALYSIS
    # =============================
    if "macro analysis" in user_input or "uk analysis" in user_input:
        await thinking_msg.remove()

        analyzer = UKEconomicsAnalyzer()

        await cl.Message(content="## 🇬🇧 UK Macroeconomic Impact Analysis\n\nI'll analyze how UK economic conditions affect your startup.").send()

        # Collect startup data
        funding = await ask_float("Funding (£ millions)", 4.0)
        location = await cl.AskUserMessage(content="Location (London/Manchester/Edinburgh/Birmingham/Bristol/Cambridge):").send()
        sector = await cl.AskUserMessage(content="Sector (FinTech/HealthTech/GreenTech/EdTech/RetailTech):").send()
        team_size = await ask_int("Team size", 10, mi=1, ma=500)

        startup_data = {
            'funding_usd_m': funding * 1.27,  # Convert to USD
            'location': location.get('output', 'London'),
            'sector': sector.get('output', 'Tech'),
            'team_size': team_size,
            'burn_rate_months': 12,  # Default
            'debt_ratio': 0.3,  # Default 30% debt
            'is_b2b': True
        }

        # Run analysis
        macro_impacts = analyzer.analyze_macro_impact(startup_data)

        # Generate chart
        chart = plot_uk_economic_indicators(df)

        # Display results
        content = f"""
## 🇬🇧 UK Economic Impact Assessment

### Interest Rate Environment
- **Cost of Capital:** {macro_impacts['interest_rate_impact']['cost_of_capital']:.1f}%
- **Annual Interest Cost:** £{macro_impacts['interest_rate_impact']['annual_interest_cost']*0.79:.0f}k
- **Impact Level:** {macro_impacts['interest_rate_impact']['impact_level']}
- **Recommendation:** {macro_impacts['interest_rate_impact']['recommendation']}

### Inflation Impact
- **Current CPI:** {macro_impacts['inflation_impact']['current_inflation']}%
- **Annual Cost Increase:** £{macro_impacts['inflation_impact']['real_cost_increase_annual']*0.79:.0f}
- **Pricing Power:** {macro_impacts['inflation_impact']['pricing_power']}
- **Wage Pressure:** {macro_impacts['inflation_impact']['wage_pressure']}

### Labour Market Conditions
- **UK Unemployment:** {macro_impacts['labour_market_impact']['unemployment_rate']}%
- **Labour Cost Index:** {macro_impacts['labour_market_impact']['labour_cost_index']:.0f}
- **Talent Availability:** {macro_impacts['labour_market_impact']['talent_availability']}
- **Wage Growth Pressure:** {macro_impacts['labour_market_impact']['wage_growth_pressure']}

### Regional Factors ({startup_data['location']})
- **Regional Growth:** {macro_impacts['regional_factors']['regional_growth']}%
- **Cost Index:** {macro_impacts['regional_factors']['cost_index']} (100 = UK average)
- **Talent Pool:** {macro_impacts['regional_factors']['talent_pool']}
- **Competitiveness:** {macro_impacts['regional_factors']['competitiveness']}

### Sector Outlook ({startup_data['sector']})
- **Expected Growth:** {macro_impacts['sector_outlook']['growth']}%
- **Regulatory Burden:** {macro_impacts['sector_outlook']['regulation']}
- **Key Opportunity:** {macro_impacts['sector_outlook']['opportunity']}

### Strategic Recommendations:
1. {'Consider debt financing while rates stabilize' if macro_impacts['interest_rate_impact']['cost_of_capital'] < 10 else 'Focus on equity financing'}
2. {'Build inflation adjustments into contracts' if macro_impacts['inflation_impact']['current_inflation'] > 2 else 'Lock in current pricing'}
3. {'Invest in talent retention' if macro_impacts['labour_market_impact']['talent_availability'] == 'Tight' else 'Opportunity to hire quality talent'}
"""

        text_msg = await cl.Message(content=content).send()

        # Send chart
        image = cl.Image(content=chart, name="uk_economic_dashboard.png", display="inline")
        await image.send(for_id=text_msg.id)

        return

    # =============================
    # ROUTE: INTERACTIVE DASHBOARD
    # =============================
    if "dashboard" in user_input or "interactive dashboard" in user_input:
        await thinking_msg.remove()

        dashboard = InteractiveDashboard(df)

        await cl.Message(content="## 📊 Interactive Dashboard Mode\n\nGenerating real-time analytics dashboard with interactive features...").send()

        # Create executive summary
        summary_text = create_dashboard_summary(dashboard)
        summary_msg = await cl.Message(content=summary_text).send()

        # Generate and send real-time metrics dashboard
        dashboard_chart = dashboard.create_real_time_metrics_dashboard()
        dashboard_image = cl.Image(content=dashboard_chart, name="interactive_dashboard.png", display="inline")
        await dashboard_image.send(for_id=summary_msg.id)

        await cl.Message(
            content="## 🎯 Dashboard Commands Available:\n\n"
                   "• **'interactive scatter'** - Dynamic scatter plot with hover details\n"
                   "• **'correlation heatmap'** - Multi-dimensional relationship analysis\n"
                   "• **'filter dashboard'** - Apply filters (sector, country, funding)\n"
                   "• **'compare startups'** - Side-by-side startup analysis\n"
                   "• **'export dashboard'** - Generate professional PDF report\n\n"
                   "🔍 **Pro Tip:** Use filters to drill down into specific segments!"
        ).send()

        return

    # =============================
    # ROUTE: INTERACTIVE SCATTER PLOT
    # =============================
    if "interactive scatter" in user_input:
        await thinking_msg.remove()

        dashboard = InteractiveDashboard(df)

        await cl.Message(content="## 📊 Interactive Scatter Plot Analysis\n\nGenerating dynamic visualization with hover details and selection capabilities...").send()

        # Get axis preferences from user input
        x_axis = 'Funding_USD_M'
        y_axis = 'Burn_Rate_Months'

        if 'funding' in user_input and 'experience' in user_input:
            x_axis, y_axis = 'Funding_USD_M', 'Founders_Experience_Yrs'
        elif 'market' in user_input and 'funding' in user_input:
            x_axis, y_axis = 'Market_Size_Bn', 'Funding_USD_M'
        elif 'mrr' in user_input and 'growth' in user_input:
            x_axis, y_axis = 'MRR_K', 'Monthly_Growth_Rate'

        # Generate interactive scatter plot
        scatter_chart = dashboard.create_interactive_scatter(x_axis, y_axis)

        content = f"""
## 📊 Interactive Scatter Analysis: {x_axis.replace('_', ' ')} vs {y_axis.replace('_', ' ')}

### 🎯 **Key Insights:**
- **Green dots** = Successful startups
- **Red dots** = Failed startups
- **Dot size** = Market size (larger = bigger market)

### 🔍 **Interactive Features:**
- **Hover** over points for detailed information
- **Click** points to select for comparison
- **Drag** to zoom into specific areas
- **Double-click** to reset zoom

### 📈 **Analysis Options:**
Try these variations:
• "interactive scatter funding vs experience"
• "interactive scatter market vs funding"
• "interactive scatter mrr vs growth"
"""

        text_msg = await cl.Message(content=content).send()
        scatter_image = cl.Image(content=scatter_chart, name="interactive_scatter.png", display="inline")
        await scatter_image.send(for_id=text_msg.id)

        return

    # =============================
    # ROUTE: CORRELATION HEATMAP
    # =============================
    if "correlation heatmap" in user_input or "heatmap" in user_input:
        await thinking_msg.remove()

        dashboard = InteractiveDashboard(df)

        await cl.Message(content="## 📊 Multi-Dimensional Correlation Analysis\n\nGenerating interactive correlation heatmap...").send()

        # Generate correlation heatmap
        heatmap_chart = dashboard.create_multi_dimensional_heatmap()

        content = """
## 📊 Interactive Correlation Heatmap

### 🎯 **How to Read:**
- **Blue** = Positive correlation (variables move together)
- **Red** = Negative correlation (variables move opposite)
- **White** = No correlation
- **Numbers** = Correlation strength (-1 to +1)

### 🔍 **Key Relationships to Explore:**
- Funding vs Market Size
- Experience vs Success Rate
- MRR vs Growth Rate
- Competition vs Moat Strength

### 📈 **Insights:**
Strong correlations (>0.5 or <-0.5) indicate important relationships for investment decisions.
"""

        text_msg = await cl.Message(content=content).send()
        heatmap_image = cl.Image(content=heatmap_chart, name="correlation_heatmap.png", display="inline")
        await heatmap_image.send(for_id=text_msg.id)

        return

    # =============================
    # ROUTE: STARTUP COMPARISON
    # =============================
    if "compare startups" in user_input or "startup comparison" in user_input:
        await thinking_msg.remove()

        dashboard = InteractiveDashboard(df)

        await cl.Message(content="## 🔍 Startup Comparison Analysis\n\nSelect startups to compare side-by-side...").send()

        # Extract startup names from user input or ask user
        startup_names = []
        for startup in df['Startup'].values:
            if startup.lower() in user_input.lower():
                startup_names.append(startup)

        # If no startups found in input, ask user to specify
        if len(startup_names) < 2:
            available_startups = ", ".join(df['Startup'].head(10).values)
            await cl.Message(
                content=f"Please specify 2-4 startup names to compare.\n\n"
                       f"**Available startups:** {available_startups}...\n\n"
                       f"**Example:** \"Compare TechFlow and DataCorp and AIStart\""
            ).send()
            return

        # Generate comparison radar chart
        comparison_chart = dashboard.compare_startups(startup_names[:4])  # Limit to 4 for readability

        if comparison_chart is None:
            await cl.Message(content="❌ No valid startups found for comparison. Please check the names.").send()
            return

        content = f"""
## 🔍 Startup Comparison: {', '.join(startup_names[:4])}

### 📊 **Radar Chart Analysis:**
- **Green lines** = Successful startups
- **Red lines** = Failed startups
- **Outer edge** = Better performance (scale 0-5)

### 🎯 **Comparison Dimensions:**
- **Funding** (scaled to 0-5)
- **Founder Experience** (years)
- **Market Size** (billions)
- **Business Model Strength** (1-5)
- **Competitive Moat** (1-5)
- **MRR** (scaled to 0-5)
- **Growth Rate** (scaled to 0-5)

### 💡 **Investment Insights:**
Look for startups with larger radar areas and balanced performance across dimensions.
"""

        text_msg = await cl.Message(content=content).send()
        comparison_image = cl.Image(content=comparison_chart, name="startup_comparison.png", display="inline")
        await comparison_image.send(for_id=text_msg.id)

        return

    # =============================
    # ROUTE: DASHBOARD FILTERING
    # =============================
    if "filter dashboard" in user_input or "apply filters" in user_input:
        await thinking_msg.remove()

        await cl.Message(content="## 🔍 Dashboard Filtering Options\n\nApply filters to focus your analysis...").send()

        # Parse filters from user input
        filters = {}

        # Sector filtering
        if 'fintech' in user_input.lower():
            filters['sectors'] = ['FinTech']
        elif 'healthtech' in user_input.lower():
            filters['sectors'] = ['HealthTech']
        elif 'ai' in user_input.lower() or 'artificial intelligence' in user_input.lower():
            filters['sectors'] = ['AI']

        # Country filtering
        if 'uk' in user_input.lower() or 'united kingdom' in user_input.lower():
            filters['countries'] = ['UK']
        elif 'us' in user_input.lower() or 'usa' in user_input.lower():
            filters['countries'] = ['US']

        # Success filtering
        if 'successful' in user_input.lower() or 'success only' in user_input.lower():
            filters['success_only'] = True

        # Apply filters and generate filtered dashboard
        dashboard = InteractiveDashboard(df)
        if filters:
            dashboard.filter_data(filters)

        # Generate filtered dashboard
        filtered_summary = create_dashboard_summary(dashboard)
        filtered_chart = dashboard.create_real_time_metrics_dashboard()

        filter_description = ""
        if 'sectors' in filters:
            filter_description += f"**Sectors:** {', '.join(filters['sectors'])}\n"
        if 'countries' in filters:
            filter_description += f"**Countries:** {', '.join(filters['countries'])}\n"
        if 'success_only' in filters:
            filter_description += "**Filter:** Successful startups only\n"

        content = f"""
## 🔍 Filtered Dashboard Analysis

### 📊 **Applied Filters:**
{filter_description if filter_description else "**No specific filters detected.** Try: 'filter dashboard fintech uk successful'"}

{filtered_summary}

### 🎯 **Available Filter Commands:**
• "filter dashboard fintech" - FinTech startups only
• "filter dashboard uk successful" - Successful UK startups
• "filter dashboard healthtech us" - US HealthTech companies
• "filter dashboard ai" - AI/ML startups
"""

        text_msg = await cl.Message(content=content).send()
        filtered_image = cl.Image(content=filtered_chart, name="filtered_dashboard.png", display="inline")
        await filtered_image.send(for_id=text_msg.id)

        return

    # =============================
    # ROUTE: EXPORT DASHBOARD
    # =============================
    if "export dashboard" in user_input or "dashboard report" in user_input:
        await thinking_msg.remove()

        await cl.Message(content="## 📄 Exporting Interactive Dashboard Report\n\nGenerating comprehensive PDF with all dashboard analytics...").send()

        dashboard = InteractiveDashboard(df)

        # Create a comprehensive dashboard export
        export_content = f"""
# 📊 NAVADA Interactive Dashboard Report

## Executive Summary
{create_dashboard_summary(dashboard)}

## 📈 Dashboard Analytics

### Key Insights:
- **Real-time Metrics:** Multi-dimensional analysis across 4 key areas
- **Interactive Features:** Hover details, drill-down capabilities, filtering
- **Comparison Tools:** Side-by-side startup analysis with radar charts
- **Correlation Analysis:** Relationship mapping between key variables

### Available Commands:
1. **dashboard** - Launch main interactive dashboard
2. **interactive scatter** - Dynamic scatter plots with selection
3. **correlation heatmap** - Multi-dimensional correlation matrix
4. **compare startups** - Radar chart comparisons
5. **filter dashboard [criteria]** - Apply smart filters

### Professional Features:
- ✅ Real-time data visualization
- ✅ Interactive hover details
- ✅ Custom filtering and drill-down
- ✅ Multi-startup comparisons
- ✅ Export capabilities
- ✅ Mobile-responsive design

---
**Generated by NAVADA Interactive Dashboard Suite**
*Next-generation startup analytics with real-time intelligence*
"""

        # Send the export summary
        await cl.Message(content=export_content).send()

        await cl.Message(
            content="## 🎯 Dashboard Export Complete!\n\n"
                   "**What's Included:**\n"
                   "• Executive summary with key metrics\n"
                   "• Interactive feature documentation\n"
                   "• Command reference guide\n"
                   "• Professional formatting\n\n"
                   "**Next Steps:**\n"
                   "• Use 'dashboard' to launch interactive mode\n"
                   "• Try 'interactive scatter' for dynamic analysis\n"
                   "• Explore 'compare startups [names]' for detailed comparisons"
        ).send()

        return

    # =============================
    # ROUTE 8: GENERATE PDF REPORT
    # =============================
    if "generate report" in user_input or "create report" in user_input or "investment report" in user_input:
        # Show loading message
        msg = cl.Message(content="📄 Generating comprehensive PDF report... This may take 10-15 seconds.")
        await msg.send()

        # Check if user specified a startup name
        startup_name = None
        for startup in df['Startup'].values:
            if startup.lower() in user_input:
                startup_name = startup
                break

        try:
            # Generate PDF report
            pdf_bytes = generate_investment_report(df, startup_name)

            # Remove loading message
            await msg.remove()

            # Send description message
            if startup_name:
                desc_msg = cl.Message(
                    content=f"### 📊 Investment Analysis Report: {startup_name}\n\n"
                            f"**Report Contents:**\n"
                            f"- Executive Summary\n"
                            f"- Key Metrics & Financials\n"
                            f"- 4 Data Visualizations\n"
                            f"- Risk Analysis\n"
                            f"- Investment Recommendations\n\n"
                            f"**Viability score and actionable insights included.**"
                )
            else:
                desc_msg = cl.Message(
                    content=f"### 📊 Portfolio Analysis Report\n\n"
                            f"**Report Contents:**\n"
                            f"- Portfolio Overview ({len(df)} startups)\n"
                            f"- Success Rate Analysis\n"
                            f"- 4 Data Visualizations\n"
                            f"- Risk Assessment\n"
                            f"- Strategic Recommendations\n\n"
                            f"**Download the PDF below for the complete analysis.**"
                )
            await desc_msg.send()

            # Send PDF file as downloadable attachment
            report_filename = f"{startup_name}_Analysis.pdf" if startup_name else "Portfolio_Analysis.pdf"
            pdf_element = cl.File(
                name=report_filename,
                content=pdf_bytes,
                display="inline"
            )
            await pdf_element.send()

        except Exception as e:
            await msg.remove()
            await cl.Message(
                content=f"❌ Error generating report: {str(e)}\n\n"
                        f"Please try again or contact support if the issue persists."
            ).send()

        return  # Exit handler

    # =============================
    # ROUTE 7: INTELLIGENT CHART DETECTION
    # =============================
    # Check if user is asking for a chart/graph/visualization
    chart_keywords = ["chart", "graph", "plot", "visualize", "visualization", "show me", "display"]
    is_chart_request = any(keyword in user_input for keyword in chart_keywords)

    if is_chart_request:
        # -------------------------
        # DETECT CHART INTENT WITH AI
        # -------------------------
        df_str = df.to_string(index=False)
        available_columns = list(df.columns)

        # Get session ID for LangSmith tracking
        session_id = cl.user_session.get("session_id", get_session_id())

        # Include LangSmith metadata if available
        if langsmith_client and session_id:
            intent_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data visualization assistant. Analyze the user's request and determine:\n"
                            "1. What type of chart they want (bar, scatter, line, pie, or 'analysis' for text response)\n"
                            "2. Which columns to use (x-axis and y-axis)\n"
                            "3. A descriptive title\n\n"
                            f"Available columns: {', '.join(available_columns)}\n\n"
                            "Respond ONLY in this JSON format:\n"
                            '{"chart_type": "bar/scatter/line/pie/analysis", "x_col": "column_name", "y_col": "column_name", "title": "Chart Title"}\n\n'
                            "If the request doesn't make sense for a chart, use chart_type: 'analysis'."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Dataset columns: {available_columns}\n\nUser request: {message.content}"
                    }
                ],
                max_tokens=150,
            )
        else:
            intent_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data visualization assistant. Analyze the user's request and determine:\n"
                            "1. What type of chart they want (bar, scatter, line, pie, or 'analysis' for text response)\n"
                            "2. Which columns to use (x-axis and y-axis)\n"
                            "3. A descriptive title\n\n"
                            f"Available columns: {', '.join(available_columns)}\n\n"
                            "Respond ONLY in this JSON format:\n"
                            '{"chart_type": "bar/scatter/line/pie/analysis", "x_col": "column_name", "y_col": "column_name", "title": "Chart Title"}\n\n'
                            "If the request doesn't make sense for a chart, use chart_type: 'analysis'."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Dataset columns: {available_columns}\n\nUser request: {message.content}"
                    }
                ],
                max_tokens=150
            )

        try:
            # Parse AI response to get chart parameters
            import json
            intent_text = intent_response.choices[0].message.content.strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```" in intent_text:
                intent_text = intent_text.split("```")[1]
                if intent_text.startswith("json"):
                    intent_text = intent_text[4:]
            intent = json.loads(intent_text.strip())

            # If AI says to do analysis instead of chart, fall through to Q&A
            if intent.get("chart_type") == "analysis":
                raise ValueError("Analysis requested instead of chart")

            # Generate the requested chart
            msg = cl.Message(content=f"📊 Generating {intent['chart_type']} chart...")
            await msg.send()

            png = plot_custom_chart(
                df,
                intent.get("chart_type", "bar"),
                intent.get("x_col", "Startup"),
                intent.get("y_col", "Funding_USD_M"),
                intent.get("title")
            )

            await msg.remove()
            text_msg = cl.Message(content=f"### 📊 {intent.get('title', 'Custom Chart')}")
            await text_msg.send()
            image = cl.Image(content=png, name="custom_chart.png", display="inline")
            await image.send(for_id=text_msg.id)
            return

        except Exception as e:
            # If chart generation fails, fall through to regular Q&A
            print(f"Chart generation failed: {e}")
            pass

    # =============================
    # ROUTE 8: INTERACTIVE DASHBOARDS
    # =============================
    # Handle general "interactive" request - default to scatter plot dashboard
    if user_input.strip() == "interactive" or (user_input.strip() == "proceed" and thinking_msg.content and "interactive" in thinking_msg.content):
        msg = cl.Message(content="🎯 Creating interactive scatter plot dashboard...")
        await msg.send()

        html_content = create_interactive_scatter(df, "Interactive Startup Analysis")
        await msg.remove()

        # Send the interactive chart as HTML
        await cl.Message(
            content="## 🎯 Interactive Startup Dashboard\n\n"
                   "**Features:**\n"
                   "- 🖱️ Hover for detailed startup information\n"
                   "- 🔍 Zoom and pan to explore data\n"
                   "- 📊 Size indicates market size, color shows success/failure\n\n"
        ).send()

        # Create an HTML file and send it
        with open("interactive_dashboard.html", "w") as f:
            f.write(html_content)

        dashboard_file = cl.File(
            path="interactive_dashboard.html",
            name="Interactive Dashboard",
            display="inline"
        )
        await dashboard_file.send()
        return

    if "interactive" in user_input and ("dashboard" in user_input or "scatter" in user_input):
        msg = cl.Message(content="🎯 Creating interactive scatter plot dashboard...")
        await msg.send()

        html_content = create_interactive_scatter(df, "Interactive Startup Analysis")
        await msg.remove()

        # Send the interactive chart as HTML
        await cl.Message(
            content="## 🎯 Interactive Startup Dashboard\n\n"
                   "**Features:**\n"
                   "- 🖱️ Hover for detailed startup information\n"
                   "- 🔍 Zoom and pan to explore data\n"
                   "- 📊 Size indicates market size, color shows success/failure\n\n"
        ).send()

        # Create an HTML file and send it
        with open("interactive_dashboard.html", "w") as f:
            f.write(html_content)

        dashboard_file = cl.File(
            path="interactive_dashboard.html",
            name="Interactive Dashboard",
            display="inline"
        )
        await dashboard_file.send()
        return

    if "interactive timeline" in user_input:
        msg = cl.Message(content="📈 Creating interactive failure timeline...")
        await msg.send()

        html_content = create_interactive_timeline(df)
        await msg.remove()

        await cl.Message(
            content="## 📈 Interactive Failure Timeline\n\n"
                   "**Features:**\n"
                   "- 🖱️ Hover to see startup details\n"
                   "- 📊 Interactive bars with sector and funding info\n"
                   "- 🔍 Zoom to focus on specific time ranges\n\n"
        ).send()

        with open("interactive_timeline.html", "w") as f:
            f.write(html_content)

        timeline_file = cl.File(
            path="interactive_timeline.html",
            name="Interactive Timeline",
            display="inline"
        )
        await timeline_file.send()
        return

    if "sector dashboard" in user_input or ("interactive" in user_input and "sector" in user_input):
        msg = cl.Message(content="🏭 Creating interactive sector dashboard...")
        await msg.send()

        html_content = create_sector_dashboard(df)
        await msg.remove()

        await cl.Message(
            content="## 🏭 Interactive Sector Dashboard\n\n"
                   "**Features:**\n"
                   "- 📊 Four interconnected charts\n"
                   "- 🖱️ Hover and zoom on each panel\n"
                   "- 💡 Compare sectors across multiple dimensions\n\n"
        ).send()

        with open("sector_dashboard.html", "w") as f:
            f.write(html_content)

        sector_file = cl.File(
            path="sector_dashboard.html",
            name="Sector Dashboard",
            display="inline"
        )
        await sector_file.send()
        return

    # =============================
    # ROUTE 9: NEW ADVANCED CHARTS
    # =============================
    if "growth trajectory" in user_input or ("growth" in user_input and "chart" in user_input):
        msg = cl.Message(content="📈 Generating growth trajectory analysis...")
        await msg.send()

        png = plot_growth_trajectory(df)
        await msg.remove()

        text_msg = cl.Message(
            content="### 📈 Growth Trajectory Analysis\n\n"
                   "This chart shows **MRR growth vs company age** with bubble sizes representing growth rates.\n"
                   "- **Green dots**: Successful companies\n"
                   "- **Red dots**: Failed companies\n"
                   "- **Bubble size**: Monthly growth rate percentage"
        )
        await text_msg.send()

        image = cl.Image(content=png, name="growth_trajectory.png", display="inline")
        await image.send(for_id=text_msg.id)
        return

    if "team performance" in user_input or ("team" in user_input and "matrix" in user_input):
        msg = cl.Message(content="👥 Generating team performance matrix...")
        await msg.send()

        png = plot_team_performance(df)
        await msg.remove()

        text_msg = cl.Message(
            content="### 👥 Team Performance Matrix\n\n"
                   "This chart analyzes **team size vs founder experience** correlation.\n"
                   "- **Bubble size**: Total funding raised\n"
                   "- **Red trend line**: Shows correlation between team size and experience\n"
                   "- **Colors**: Green = successful, Red = failed"
        )
        await text_msg.send()

        image = cl.Image(content=png, name="team_performance.png", display="inline")
        await image.send(for_id=text_msg.id)
        return

    if "market opportunity" in user_input or ("market" in user_input and "competition" in user_input):
        msg = cl.Message(content="🎯 Generating market opportunity matrix...")
        await msg.send()

        png = plot_market_opportunity(df)
        await msg.remove()

        text_msg = cl.Message(
            content="### 🎯 Market Opportunity Matrix\n\n"
                   "This chart identifies **sweet spots** in market size vs competition landscape.\n"
                   "- **X-axis**: Market size (bigger = better)\n"
                   "- **Y-axis**: Market opportunity (higher = less competition)\n"
                   "- **Bubble size**: Current traction (MRR)\n"
                   "- **Sweet Spot**: Large market + low competition"
        )
        await text_msg.send()

        image = cl.Image(content=png, name="market_opportunity.png", display="inline")
        await image.send(for_id=text_msg.id)
        return

    if "funding efficiency" in user_input or ("capital" in user_input and "efficiency" in user_input):
        msg = cl.Message(content="💰 Generating capital efficiency analysis...")
        await msg.send()

        png = plot_funding_efficiency(df)
        await msg.remove()

        text_msg = cl.Message(
            content="### 💰 Capital Efficiency Analysis\n\n"
                   "This chart shows **revenue generated per dollar invested**.\n"
                   "- **X-axis**: Total funding raised\n"
                   "- **Y-axis**: Annual revenue per dollar of funding\n"
                   "- **Bubble size**: Efficiency score (revenue × growth rate)\n"
                   "- **Orange line**: Median efficiency benchmark"
        )
        await text_msg.send()

        image = cl.Image(content=png, name="funding_efficiency.png", display="inline")
        await image.send(for_id=text_msg.id)
        return

    if "stage progression" in user_input or ("funding" in user_input and "stage" in user_input):
        msg = cl.Message(content="🚀 Generating funding stage analysis...")
        await msg.send()

        png = plot_stage_progression(df)
        await msg.remove()

        text_msg = cl.Message(
            content="### 🚀 Funding Stage Analysis\n\n"
                   "This chart tracks **funding amounts and failure rates by stage**.\n"
                   "- **Blue bars**: Average funding amount per stage\n"
                   "- **Red line**: Failure rate percentage\n"
                   "- **Labels**: Show funding amount and company count\n"
                   "- **Insight**: Later stages typically have lower failure rates"
        )
        await text_msg.send()

        image = cl.Image(content=png, name="stage_progression.png", display="inline")
        await image.send(for_id=text_msg.id)
        return

    if "risk assessment" in user_input or ("risk" in user_input and "radar" in user_input):
        msg = cl.Message(content="🎯 Generating risk assessment radar...")
        await msg.send()

        png = plot_risk_assessment(df)
        await msg.remove()

        text_msg = cl.Message(
            content="### 🎯 Risk Assessment Profile\n\n"
                   "This **radar chart** compares risk profiles between successful and failed companies.\n"
                   "- **Green area**: Successful companies' average risk profile\n"
                   "- **Red area**: Failed companies' average risk profile\n"
                   "- **Scale**: 0 = low risk, 10 = high risk\n"
                   "- **Categories**: Financial, Market, Team, Competition, and Traction risks"
        )
        await text_msg.send()

        image = cl.Image(content=png, name="risk_assessment.png", display="inline")
        await image.send(for_id=text_msg.id)
        return

    # =============================
    # ROUTE 10: GUIDED QUESTIONS
    # =============================
    if "questions" in user_input or "guide me" in user_input or "what should i ask" in user_input:
        await thinking_msg.remove()

        persona = get_current_persona()
        questions = persona.get("questions", [])

        content = f"### 🎯 {persona['name']} - Guided Questions\n\n"
        content += "Here are key questions to explore based on your current mode:\n\n"

        for i, question in enumerate(questions, 1):
            content += f"**{i}.** {question}\n"

        content += "\n📊 **Recommended Charts:**\n"
        for chart in persona.get("charts", []):
            chart_name = chart.replace("_", " ").title()
            content += f"- Type **'{chart_name}'** for {chart_name} analysis\n"

        content += "\n💡 **Pro Tip:** Copy any question above and I'll provide detailed analysis!"

        await cl.Message(content=content).send()
        return

    # =============================
    # ROUTE 11: INTERNET SEARCH
    # =============================
    if user_input.startswith("search ") or \
       ("search" in user_input and ("internet" in user_input or "web" in user_input)) or \
       ("search" in user_input and any(word in user_input for word in ["latest", "recent", "current", "news", "trends", "2024", "2025"])) or \
       ("what's happening" in user_input) or \
       ("latest news" in user_input) or \
       ("current trends" in user_input) or \
       ("find" in user_input and any(word in user_input for word in ["latest", "recent", "current", "new"])):

        await thinking_msg.remove()

        # Extract search query
        if user_input.startswith("search "):
            query = user_input[7:].strip()
        elif "search" in user_input:
            # Extract query after "search"
            parts = user_input.split("search", 1)
            if len(parts) > 1:
                query = parts[1].strip()
            else:
                query = user_input
        else:
            # For phrases like "what's happening with AI startups"
            query = user_input

        if not query:
            await cl.Message(
                content="Please provide a search query. For example:\n"
                       "- **search AI startups 2024**\n"
                       "- **latest trends in fintech**\n"
                       "- **current venture capital news**"
            ).send()
            return

        msg = cl.Message(content=f"🔍 Searching for: **{query}**...")
        await msg.send()

        # Perform search
        search_results = search_internet(query, count=5)

        if not search_results["success"]:
            # Provide fallback response with helpful guidance
            fallback_message = f"❌ Search failed: {search_results['error']}\n\n"

            if search_results.get("fallback_available"):
                fallback_message += "💡 **Alternative approach**: I can still help you with:\n"
                fallback_message += "• General startup advice and best practices\n"
                fallback_message += "• Analysis based on my training data\n"
                fallback_message += "• Startup failure pattern analysis\n"
                fallback_message += "• Business model evaluation\n\n"
                fallback_message += "🔧 **To enable web search**: Please configure a valid Brave Search API key in your .env file.\n"
                fallback_message += "Visit https://brave.com/search/api/ to get an API key."

            msg.content = fallback_message
            await msg.update()
            return

        # Get current persona for analysis
        persona = get_current_persona()

        # Get session ID for LangSmith tracking
        session_id = cl.user_session.get("session_id", get_session_id())

        # Analyze results with persona context
        analysis = analyze_search_results(search_results, persona, "startup and investment context", session_id)

        # Format and send results
        content = f"## 🔍 Search Results: {query}\n\n"
        content += f"**Found {search_results['total_results']} results:**\n\n"

        for i, result in enumerate(search_results['results'], 1):
            content += f"**{i}. {result['title']}**\n"
            content += f"🔗 {result['url']}\n"
            content += f"📝 {result['description'][:200]}{'...' if len(result['description']) > 200 else ''}\n"
            if result.get('age'):
                content += f"⏰ {result['age']}\n"
            content += "\n"

        content += f"---\n\n## 🤖 {persona['name']} Analysis:\n\n{analysis}"

        msg.content = content
        await msg.update()
        return

    # =============================
    # ROUTE 12: TEXT-TO-SPEECH
    # =============================
    if user_input.startswith("speak ") or user_input.startswith("say ") or \
       ("audio" in user_input and "response" in user_input) or \
       ("read aloud" in user_input) or ("voice" in user_input):

        await thinking_msg.remove()

        # Extract text to speak
        if user_input.startswith("speak "):
            text_to_speak = user_input[6:].strip()
        elif user_input.startswith("say "):
            text_to_speak = user_input[4:].strip()
        else:
            text_to_speak = "Welcome to NAVADA, your AI-powered startup viability agent. I can analyze startup risks, generate charts, and provide investment insights in both investor and founder modes."

        if not text_to_speak:
            await cl.Message(
                content="Please provide text to convert to speech:\n"
                       "- **speak [your text]**\n"
                       "- **say [your text]**\n"
                       "- **read aloud [your text]**"
            ).send()
            return

        # Generate audio
        msg = cl.Message(content=f"🔊 Generating speech: **{text_to_speak[:100]}{'...' if len(text_to_speak) > 100 else ''}**")
        await msg.send()

        try:
            # Create audio message
            audio = create_audio_message(text_to_speak, voice="alloy")

            if audio:
                # Send text message with audio
                content = f"🔊 **Audio Response:**\n\n{text_to_speak}"
                text_msg = cl.Message(content=content)
                await text_msg.send()

                # Send audio
                await audio.send(for_id=text_msg.id)
            else:
                msg.content = "❌ Failed to generate audio. Please try again."
                await msg.update()

        except Exception as e:
            msg.content = f"❌ Audio generation error: {str(e)}"
            await msg.update()

        return

    # =============================
    # ROUTE 13: PERSONA MANAGEMENT
    # =============================
    if "investor mode" in user_input or "switch to investor" in user_input:
        # Remove thinking indicator
        await thinking_msg.remove()

        cl.user_session.set("persona", "investor")
        persona = get_current_persona()
        recommendations = format_persona_recommendations("investor")
        await cl.Message(
            content=f"{persona['style']}\n\n"
                   "I'm now analyzing from a **venture capitalist perspective**. "
                   "I'll focus on ROI, market size, competitive analysis, and exit strategies.\n\n"
                   f"{recommendations}"
                   "**What would you like to analyze today?**\n\n"
                   "💰 **Quick Analysis:**\n"
                   "• Type **'portfolio'** - Investment recommendations across all startups\n"
                   "• Type **'insights'** - AI-powered risk assessment and opportunities\n"
                   "• Type **'benchmark'** - Compare new startup ideas against our dataset\n"
                   "• Type **'questions'** - Get guided investor-focused questions\n\n"
                   "📊 **Advanced Charts:**\n"
                   "• **'Funding Efficiency'** - Capital efficiency and ROI analysis\n"
                   "• **'Stage Progression'** - Funding stages vs failure rates\n"
                   "• **'Market Opportunity'** - Market size vs competition matrix\n"
                   "• **'Risk Assessment'** - Comprehensive risk radar chart\n\n"
                   "📈 **Interactive Tools:**\n"
                   "• **'Sector Dashboard'** - Multi-dimensional sector analysis\n"
                   "• **'Interactive'** - Dynamic scatter plots and correlations\n\n"
                   "🔍 **Internet Search:**\n"
                   "• **'search latest VC trends'** - Get up-to-date market intelligence\n"
                   "• **'current startup news'** - Recent developments in startup ecosystem\n"
                   "• **'search [company name] funding'** - Research specific companies\n\n"
                   "🎯 **Ask me directly:**\n"
                   "• \"Which startups have the best ROI potential?\"\n"
                   "• \"What are the red flags in our portfolio?\"\n"
                   "• \"Search for latest AI startup trends\""
        ).send()
        return

    if "founder mode" in user_input or "switch to founder" in user_input:
        # Remove thinking indicator
        await thinking_msg.remove()

        cl.user_session.set("persona", "founder")
        persona = get_current_persona()
        recommendations = format_persona_recommendations("founder")
        await cl.Message(
            content=f"{persona['style']}\n\n"
                   "I'm now analyzing from an **experienced founder perspective**. "
                   "I'll focus on practical execution, team building, product development, and tactical advice.\n\n"
                   f"{recommendations}"
                   "**What challenges can I help you tackle today?**\n\n"
                   "🚀 **Quick Assessment:**\n"
                   "• Type **'assess idea'** - Get viability score for your startup concept\n"
                   "• Type **'benchmark'** - Compare your metrics to successful startups\n"
                   "• Type **'insights'** - Get tactical recommendations to reduce risk\n"
                   "• Type **'questions'** - Get guided founder-focused questions\n\n"
                   "📊 **Growth Analysis:**\n"
                   "• **'Growth Trajectory'** - MRR growth patterns and success factors\n"
                   "• **'Team Performance'** - Team size vs experience optimization\n"
                   "• **'Market Opportunity'** - Find your competitive sweet spot\n"
                   "• **'Stage Progression'** - Funding stage benchmarks and expectations\n\n"
                   "📈 **Tactical Tools:**\n"
                   "• **'Timeline'** - Failure patterns to avoid common pitfalls\n"
                   "• **'Interactive'** - Explore data patterns affecting your sector\n"
                   "• **'Portfolio'** - Study successful companies in your space\n\n"
                   "🔍 **Market Intelligence:**\n"
                   "• **'search competitor analysis'** - Research competitive landscape\n"
                   "• **'latest startup challenges'** - Current industry challenges\n"
                   "• **'search [your sector] trends'** - Stay ahead of market shifts\n\n"
                   "💡 **Ask me directly:**\n"
                   "• \"How can I extend my runway and reduce burn?\"\n"
                   "• \"What team size is optimal for my stage?\"\n"
                   "• \"Search for current SaaS pricing trends\""
        ).send()
        return

    # =============================
    # ROUTE: DISPLAY RECOMMENDATIONS
    # =============================
    if "recommendations" in user_input or "best practices" in user_input:
        await thinking_msg.remove()

        current_persona_name = cl.user_session.get("persona", "founder")
        current_persona = PERSONAS[current_persona_name]
        recommendations = format_persona_recommendations(current_persona_name)

        await cl.Message(
            content=f"{current_persona['style']}\n\n"
                   f"Here are the key recommendations for {current_persona['name']}:"
                   f"{recommendations}"
                   "💡 **Want more specific advice?** Ask me about any of these areas, or switch to a different mode:\n"
                   "• **'investor mode'** - VC investment criteria\n"
                   "• **'founder mode'** - Tactical execution advice\n"
                   "• **'economist mode'** - UK economic analysis\n"
                   "• **'company analyst mode'** - Financial performance focus"
        ).send()
        return

    # =============================
    # ROUTE: USER DASHBOARD & HISTORY
    # =============================
    if "dashboard" in user_input or "my conversations" in user_input or "history" in user_input:
        await thinking_msg.remove()

        if not AUTH_AVAILABLE or not auth_status["authenticated"]:
            await cl.Message(content="❌ Please log in to view your dashboard.").send()
            return

        # Get user's conversation history
        conversations = auth_manager.get_user_conversations(auth_status["user_id"], limit=10)

        if not conversations:
            await cl.Message(
                content=f"📊 **Welcome to your NAVADA Dashboard, {auth_status['username']}!**\n\n"
                       "🎯 **Account Info:**\n"
                       f"• Username: {auth_status['username']}\n"
                       f"• Email: {auth_status.get('email', 'Not provided')}\n"
                       f"• Subscription: {auth_status.get('subscription_tier', 'free').title()}\n\n"
                       "📝 **Conversation History:** No conversations yet\n\n"
                       "Start chatting to build your conversation history!"
            ).send()
        else:
            dashboard_content = f"📊 **Welcome to your NAVADA Dashboard, {auth_status['username']}!**\n\n"
            dashboard_content += "🎯 **Account Info:**\n"
            dashboard_content += f"• Username: {auth_status['username']}\n"
            dashboard_content += f"• Email: {auth_status.get('email', 'Not provided')}\n"
            dashboard_content += f"• Subscription: {auth_status.get('subscription_tier', 'free').title()}\n\n"
            dashboard_content += f"📝 **Recent Conversations ({len(conversations)}):**\n\n"

            for i, conv in enumerate(conversations, 1):
                dashboard_content += f"**{i}. {conv['title']}**\n"
                dashboard_content += f"• Mode: {conv['persona_mode'].title()}\n"
                dashboard_content += f"• Messages: {conv['message_count']}\n"
                dashboard_content += f"• Updated: {conv['updated_at'][:19].replace('T', ' ')}\n"
                dashboard_content += f"• Session ID: `{conv['session_id'][:8]}...`\n\n"

            dashboard_content += "💡 **Tip:** Type 'logout' to end your session securely."

            await cl.Message(content=dashboard_content).send()
        return

    if user_input == "logout":
        await thinking_msg.remove()

        if not AUTH_AVAILABLE or not auth_status["authenticated"]:
            await cl.Message(content="❌ You are not logged in.").send()
            return

        # Logout user
        session_token = cl.user_session.get("session_token")
        if session_token:
            auth_manager.logout_user(session_token)

        # Clear session data
        cl.user_session.set("auth_token", None)
        cl.user_session.set("session_token", None)
        cl.user_session.set("user_id", None)
        cl.user_session.set("username", None)
        cl.user_session.set("user_email", None)
        cl.user_session.set("subscription_tier", None)

        await cl.Message(
            content=f"👋 **Goodbye, {auth_status['username']}!**\n\n"
                   "You have been logged out successfully.\n"
                   "Type `login username password` to log back in."
        ).send()
        return

    # =============================
    # ROUTE: UK ECONOMIST MODE
    # =============================
    if "economist mode" in user_input or "economics mode" in user_input or "uk economy" in user_input:
        await thinking_msg.remove()

        cl.user_session.set("persona", "economist")
        persona = get_current_persona()
        recommendations = format_persona_recommendations("economist")

        await cl.Message(
            content=f"{persona['style']}\n\n"
                    "I'm now analyzing from a **UK economics perspective**, combining macroeconomic trends with startup viability.\n\n"
                    f"{recommendations}"
                    "**Current UK Economic Context:**\n"
                    "• Bank Rate: 4.75% (affecting cost of capital)\n"
                    "• CPI Inflation: 2.3% (near BoE target)\n"
                    "• Unemployment: 4.2% (tight labour market)\n"
                    "• GDP Growth: 0.3% quarterly (sluggish growth)\n"
                    "• GBP/USD: 1.27 (currency impacts)\n\n"
                    "**Economic Analysis Tools:**\n"
                    "• Type **'macro analysis'** - UK macroeconomic impact assessment\n"
                    "• Type **'sector outlook'** - UK sector-specific opportunities\n"
                    "• Type **'regional analysis'** - Location-based economic factors\n"
                    "• Type **'policy impact'** - Government policy effects\n\n"
                    "**Key Questions I Can Answer:**\n"
                    "• How do interest rates affect your funding strategy?\n"
                    "• What's the inflation impact on your cost structure?\n"
                    "• How does UK productivity affect your scaling plans?\n"
                    "• Which UK regions offer the best opportunities?\n"
                    "• How do fiscal policies impact your sector?\n\n"
                    "**Ask me about:**\n"
                    "• Brexit impacts on your market\n"
                    "• London vs regional economics\n"
                    "• UK labour market conditions\n"
                    "• Sector-specific regulations\n"
                    "• Currency exposure and hedging"
        ).send()
        return

    if "persona" in user_input or "mode" in user_input:
        current_persona = get_current_persona()
        await cl.Message(
            content=f"## 🎭 Current Mode: {current_persona['style']}\n\n"
                   "**Available modes:**\n\n"
                   "💼 **Investor Mode** - VC perspective focused on ROI and exit strategies\n"
                   "• Best for: Portfolio analysis, due diligence, investment decisions\n"
                   "• Commands: portfolio, insights, sector dashboard\n\n"
                   "🚀 **Founder Mode** - Entrepreneur perspective focused on execution\n"
                   "• Best for: Startup assessment, risk reduction, tactical advice\n"
                   "• Commands: assess idea, benchmark, timeline\n\n"
                   "🇬🇧 **UK Economist Mode** - Economic analysis perspective for UK markets\n"
                   "• Best for: Macroeconomic impacts, regional analysis, policy effects\n"
                   "• Commands: macro analysis, sector outlook, regional analysis\n\n"
                   "**Ready to switch?**\n"
                   "• Type **'investor mode'** for VC analysis\n"
                   "• Type **'founder mode'** for founder guidance\n"
                   "• Type **'economist mode'** for UK economic analysis\n\n"
                   "**Or continue in current mode - what would you like to analyze?**"
        ).send()
        return

    # =============================
    # ROUTE: COMPANY ANALYST MODE
    # =============================
    if "company analyst" in user_input or "company analysis" in user_input or "financial analysis" in user_input:
        await thinking_msg.remove()

        cl.user_session.set("persona", "company_analyst")
        persona = get_current_persona()

        await cl.Message(
            content=f"{persona['style']}\n\n"
                    "I'm now analyzing from a **company financial health perspective**, focusing on profitability, unit economics, and financial sustainability.\n\n"
                    "**Financial Analysis Focus:**\n"
                    "• Profitability: Gross, operating, and net margins\n"
                    "• Unit Economics: LTV/CAC ratios, payback periods\n"
                    "• Cash Flow: Runway analysis, working capital\n"
                    "• Break-even: Path to profitability analysis\n"
                    "• Benchmarking: Industry performance comparisons\n\n"
                    "**Analysis Tools:**\n"
                    "• Type **'analyze company'** - Comprehensive financial analysis\n"
                    "• Type **'profitability analysis'** - Full margin and profitability assessment\n"
                    "• Type **'unit economics'** - Customer economics and LTV/CAC analysis\n"
                    "• Type **'cash flow analysis'** - Runway and cash management assessment\n"
                    "• Type **'break even analysis'** - Path to profitability calculation\n\n"
                    "**Ready to dive deep into financial health?**"
        ).send()
        return

    # =============================
    # ROUTE: COMPANY ANALYSIS EXECUTION
    # =============================
    if "analyze company" in user_input or "profitability analysis" in user_input or "financial analysis" in user_input:
        await thinking_msg.remove()

        analyzer = CompanyAnalyzer()

        await cl.Message(
            content="## 💼 Company Financial Analysis\n\n"
                    "I'll perform a comprehensive profitability and financial health analysis.\n\n"
                    "Choose analysis type:\n"
                    "1. **Quick Analysis** - Key metrics only\n"
                    "2. **Full Analysis** - Complete financial deep dive\n"
                    "3. **Upload Financials** - Analyze from CSV/Excel"
        ).send()

        analysis_type = await cl.AskUserMessage(content="Enter choice (1/2/3):").send()

        if "2" in analysis_type.get('output', '') or "full" in analysis_type.get('output', '').lower():
            # Full Analysis
            await cl.Message(content="### 📊 Full Company Analysis\n\nI'll need detailed financial information.").send()

            # Collect comprehensive data
            company_name = await cl.AskUserMessage(content="Company name:").send()
            industry = await cl.AskUserMessage(content="Industry (SaaS/E-commerce/Marketplace/FinTech/Services):").send()

            # Revenue metrics
            revenue = await ask_float("Annual revenue ($ millions)", 10.0)
            growth_rate = await ask_float("Revenue growth rate (%)", 20.0)

            # Cost structure
            cogs_pct = await ask_float("COGS as % of revenue", 40.0)
            opex_pct = await ask_float("Operating expenses as % of revenue", 35.0)
            sales_marketing_pct = await ask_float("Sales & Marketing as % of revenue", 15.0)

            # Unit economics
            cac = await ask_float("Customer Acquisition Cost ($)", 100.0)
            ltv = await ask_float("Customer Lifetime Value ($)", 400.0)
            monthly_churn = await ask_float("Monthly churn rate (%)", 5.0)

            # Cash metrics
            cash_balance = await ask_float("Current cash balance ($ millions)", 5.0)
            monthly_burn = await ask_float("Monthly burn rate ($ thousands)", 200.0)

            # Prepare financial data
            revenue_amount = revenue * 1_000_000
            financials = {
                'revenue': revenue_amount,
                'cogs': revenue_amount * (cogs_pct / 100),
                'opex': revenue_amount * (opex_pct / 100),
                'sales_marketing': revenue_amount * (sales_marketing_pct / 100),
                'fixed_costs': revenue_amount * 0.2,
                'variable_cost_ratio': cogs_pct / 100,
                'price_per_unit': 100,
                'current_revenue': revenue_amount / 12  # Monthly
            }

            metrics = {
                'customer_acquisition_cost': cac,
                'lifetime_value': ltv,
                'monthly_revenue': ltv / 24,  # Assume 24-month lifetime
                'revenue_per_unit': 100,
                'variable_cost_per_unit': 40
            }

            cash_data = {
                'cash_from_operations': -monthly_burn * 1000 * 12,
                'cash_from_investing': -revenue_amount * 0.05,
                'cash_from_financing': 0,
                'cash_balance': cash_balance * 1_000_000,
                'days_sales_outstanding': 45,
                'days_inventory_outstanding': 0 if industry.get('output', '') == 'SaaS' else 30,
                'days_payables_outstanding': 30
            }

            # Run analyses
            msg = cl.Message(content="🔍 Analyzing financial health...")
            await msg.send()

            profitability = analyzer.analyze_profitability(financials)
            unit_economics = analyzer.analyze_unit_economics(metrics)
            cash_flow = analyzer.analyze_cash_flow(cash_data)
            break_even = analyzer.calculate_break_even(financials)

            # Benchmark analysis
            company_metrics = {
                'gross_margin': profitability['gross_margin'],
                'operating_margin': profitability['operating_margin'],
                'ltv_cac_ratio': unit_economics['ltv_cac_ratio']
            }
            benchmarks = analyzer.benchmark_performance(
                company_metrics,
                industry.get('output', 'Services')
            )

            # Generate visualization
            analysis_data = {
                'ltv_cac_ratio': unit_economics['ltv_cac_ratio']
            }
            chart = plot_profitability_analysis(analysis_data)

            await msg.remove()

            # Display comprehensive results
            content = f"""
## 📊 Financial Analysis: {company_name.get('output', 'Company')}

### 💰 Profitability Analysis
- **Gross Margin:** {profitability['gross_margin']:.1f}% {'✅' if profitability['gross_margin'] > 50 else '⚠️' if profitability['gross_margin'] > 30 else '❌'}
- **Operating Margin:** {profitability['operating_margin']:.1f}% {'✅' if profitability['operating_margin'] > 15 else '⚠️' if profitability['operating_margin'] > 0 else '❌'}
- **EBITDA Margin:** {profitability['ebitda_margin']:.1f}%
- **Net Margin:** {profitability['net_margin']:.1f}%
- **Overall Health:** {profitability['profit_health']}

### 📈 Unit Economics
- **LTV/CAC Ratio:** {unit_economics['ltv_cac_ratio']:.2f} {'✅ Healthy' if unit_economics['ltv_cac_ratio'] > 3 else '⚠️ Concerning' if unit_economics['ltv_cac_ratio'] > 1 else '❌ Unsustainable'}
- **Payback Period:** {unit_economics['payback_months']:.1f} months
- **Contribution Margin:** {unit_economics['contribution_margin_pct']:.1f}%
- **Unit Economics:** {unit_economics['unit_economics_health']}

### 💵 Cash Flow & Runway
- **Monthly Burn:** ${cash_flow['monthly_burn']:,.0f}
- **Runway:** {cash_flow['runway_months']:.1f} months {'✅' if cash_flow['runway_months'] > 18 else '⚠️' if cash_flow['runway_months'] > 12 else '❌'}
- **Cash Conversion Cycle:** {cash_flow['cash_conversion_cycle']:.0f} days
- **Cash Efficiency:** {cash_flow['cash_efficiency']}

### 🎯 Break-even Analysis
- **Break-even Revenue:** ${break_even['break_even_revenue']:,.0f}
- **Margin of Safety:** {break_even['margin_of_safety']:.1f}%
- **Months to Break-even:** {break_even['months_to_break_even'] if break_even['months_to_break_even'] < 60 else '60+'}

### 📊 Industry Benchmarks ({industry.get('output', 'Services')})
- **Gross Margin:** Company {benchmarks['comparisons']['gross_margin']['company']:.1f}% vs Industry {benchmarks['comparisons']['gross_margin']['industry']:.1f}% ({benchmarks['comparisons']['gross_margin']['performance']})
- **Operating Margin:** Company {benchmarks['comparisons']['operating_margin']['company']:.1f}% vs Industry {benchmarks['comparisons']['operating_margin']['industry']:.1f}% ({benchmarks['comparisons']['operating_margin']['performance']})
- **Overall Rating:** {benchmarks['overall_rating']}

### 💡 Key Recommendations
1. {'✅ Maintain strong margins' if profitability['gross_margin'] > 50 else '⚠️ Improve gross margins through pricing or cost reduction'}
2. {'✅ Unit economics are healthy' if unit_economics['ltv_cac_ratio'] > 3 else '⚠️ Optimize CAC or increase LTV'}
3. {'✅ Adequate runway' if cash_flow['runway_months'] > 18 else '❌ Consider fundraising or reducing burn'}
4. {'✅ Near profitability' if break_even['margin_of_safety'] > 0 else '⚠️ Focus on path to profitability'}

### 🎬 Action Items
{chr(10).join(['• ' + rec for rec in benchmarks['recommendations']])}
"""

            text_msg = cl.Message(content=content)
            await text_msg.send()

            # Send chart
            image = cl.Image(content=chart, name="company_analysis.png", display="inline")
            await image.send(for_id=text_msg.id)

        else:  # Quick Analysis
            await cl.Message(content="### ⚡ Quick Profitability Check\n\nProvide key metrics for rapid assessment.").send()

            revenue = await ask_float("Monthly revenue ($ thousands)", 100.0)
            costs = await ask_float("Monthly costs ($ thousands)", 120.0)
            customers = await ask_int("Number of customers", 100, mi=1, ma=100000)

            # Quick calculations
            profit = revenue - costs
            margin = (profit / revenue * 100) if revenue > 0 else -100
            revenue_per_customer = (revenue * 1000) / customers if customers > 0 else 0

            status = "📈 Profitable" if profit > 0 else "📉 Not Yet Profitable"
            health = "Strong" if margin > 20 else "Moderate" if margin > 0 else "Needs Improvement"

            content = f"""
### ⚡ Quick Analysis Results

**Status:** {status}
**Net Margin:** {margin:.1f}%
**Monthly Profit/Loss:** ${profit*1000:,.0f}
**Revenue per Customer:** ${revenue_per_customer:.2f}
**Financial Health:** {health}

**Quick Insights:**
- {'Focus on achieving profitability' if profit < 0 else 'Maintain positive trajectory'}
- {'Reduce costs or increase pricing' if margin < 0 else 'Consider scaling'}
- {'Improve customer monetization' if revenue_per_customer < 100 else 'Good customer value'}
"""

            await cl.Message(content=content).send()

        return

    # =============================
    # ROUTE 10: WEB SCRAPING
    # =============================
    if user_input.startswith("scrape "):
        # Remove thinking indicator
        await thinking_msg.remove()

        # Parse scrape command: "scrape <url>" or "scrape <url> <selector>"
        parts = message.content.strip().split()

        if len(parts) < 2:
            await cl.Message(
                content="⚠️ **Invalid scrape command**\n\n"
                       "**Usage:** `scrape <url>` or `scrape <url> <selector>`\n\n"
                       "**Examples:**\n"
                       "• `scrape https://example.com` - Scrape paragraphs\n"
                       "• `scrape https://news.site h1,h2` - Scrape headlines\n"
                       "• `scrape https://blog.com .article` - Scrape articles by class"
            ).send()
            return

        url = parts[1]
        selector = parts[2] if len(parts) > 2 else "p"  # Default to paragraphs

        # Show loading message with URL
        loading_msg = cl.Message(content=f"🔍 Scraping {url}...")
        await loading_msg.send()

        # Perform scraping
        scrape_result = scrape_site(url, selector)

        # Remove loading message
        await loading_msg.remove()

        if not scrape_result["success"]:
            # Handle scraping failure
            await cl.Message(
                content=f"❌ **Scraping failed**\n\n"
                       f"**URL:** {url}\n"
                       f"**Error:** {scrape_result['error']}\n\n"
                       f"**Suggestions:**\n"
                       f"• Check if the URL is accessible in your browser\n"
                       f"• Try a different CSS selector (h1, div, span)\n"
                       f"• Some sites block automated requests"
            ).send()
            return

        # Scraping successful - get results
        scraped_data = scrape_result["data"]
        count = scrape_result["count"]
        size_mb = scrape_result["size_mb"]

        # Show scraping summary
        summary_msg = cl.Message(
            content=f"✅ **Scraping successful!**\n\n"
                   f"**URL:** {url}\n"
                   f"**Selector:** `{selector}`\n"
                   f"**Items scraped:** {count}\n"
                   f"**Content size:** {size_mb}MB\n\n"
                   f"**Preview (first 3 items):**"
        )
        await summary_msg.send()

        # Show preview of scraped content
        preview_items = scraped_data.head(3)
        preview_text = ""
        for i, row in preview_items.iterrows():
            content = row["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            preview_text += f"**{i+1}.** {content}\n\n"

        preview_msg = cl.Message(content=preview_text)
        await preview_msg.send()

        # Store scraped data in session memory for later reference
        session_id = get_session_id()
        add_to_memory(session_id, "scraped_data", f"Scraped {count} items from {url}")

        # Store the actual data in session for follow-up questions
        cl.user_session.set("last_scraped_data", scraped_data)
        cl.user_session.set("last_scraped_url", url)

        # Generate AI analysis of scraped content
        analysis_msg = cl.Message(content="🤖 **Analyzing scraped content...**")
        await analysis_msg.send()

        # Get current persona for contextual analysis
        persona = get_current_persona()
        analysis = analyze_scraped_content(scraped_data, url, persona)

        # Remove analysis loading message
        await analysis_msg.remove()

        # Send AI analysis with persona indicator
        analysis_response = f"**Website Analysis**\n\n{analysis}"
        await cl.Message(content=analysis_response).send()

        return

    # =============================
    # ROUTE 11: AUTO-GENERATED INSIGHTS
    # =============================
    if "insights" in user_input or "auto insights" in user_input or "generate insights" in user_input:
        msg = cl.Message(content="🤖 Analyzing data and generating insights...")
        await msg.send()

        insights = generate_insights(df, "general")
        insights_message = format_insights_message(insights)

        await msg.remove()
        await cl.Message(content=insights_message).send()
        return

    # =============================
    # ROUTE 12: DATA EXPORT & DOWNLOAD
    # =============================
    if any(keyword in user_input for keyword in ["export data", "download data", "export csv", "download csv", "export json", "download json"]):
        await thinking_msg.remove()

        # Determine export format
        if "json" in user_input:
            format_type = "json"
        else:
            format_type = "csv"  # Default to CSV

        # Send data export
        await send_data_export(df, "startup_dataset", format_type)
        return

    # =============================
    # ROUTE 13: AI-POWERED Q&A WITH MEMORY & PERSONA (DEFAULT)
    # =============================
    # If no specific pattern matched, use GPT-4 for natural language response

    # -------------------------
    # SESSION MEMORY & PERSONA INTEGRATION
    # -------------------------
    session_id = cl.user_session.get("session_id", get_session_id())
    persona = get_current_persona()
    memory_context = get_memory_context(session_id)

    # Add user message to memory
    add_to_memory(session_id, "user", message.content)

    # Save user message to database if authenticated
    if AUTH_AVAILABLE and auth_status["authenticated"]:
        auth_manager.save_conversation(
            user_id=auth_status["user_id"],
            chainlit_session_id=session_id,
            role="user",
            content=message.content,
            persona_mode=get_current_persona()["name"].lower().replace(" mode", ""),
            metadata={"timestamp": timestamp, "raw_input": user_input_raw}
        )

    # -------------------------
    # PREPARE ENHANCED CONTEXT
    # -------------------------
    # Convert DataFrame to string for inclusion in prompt
    df_str = df.to_string(index=False)

    # Auto-search enhancement for relevant queries
    search_results = None
    search_context = ""
    current_persona_name = cl.user_session.get("persona", "founder")

    # -------------------------
    # USE THREAD-AWARE PROCESSING FOR LANGSMITH
    # -------------------------
    # Check if LangSmith is enabled and use thread context
    if langsmith_client:
        # Use the thread-aware processing function with LangSmith tracing
        enhanced_question = f"Dataset:\n{df_str}\n\nUser question: {message.content}"

        ai_response = process_with_thread_context(
            question=enhanced_question,
            session_id=session_id,
            get_chat_history=True,  # Always use history for continuity
            persona=persona
        )
    else:
        # Fallback to standard processing without LangSmith tracing
        enhanced_system_prompt = (
            f"{persona['system_prompt']}\n\n"
            "Available commands you can suggest:\n"
            "- 'timeline' - failure timeline\n"
            "- 'funding vs burn' - funding vs burn rate\n"
            "- 'interactive dashboard' - interactive scatter plot\n"
            "- 'interactive timeline' - interactive failure timeline\n"
            "- 'sector dashboard' - multi-chart sector analysis\n"
            "- 'benchmark' - compare founder idea to dataset\n"
            "- 'portfolio' - analyze multiple startups with heatmap\n"
            "- 'insights' - auto-generate risks and recommendations\n"
            "- 'investor mode' / 'founder mode' - switch analysis perspective\n\n"
            "Remember conversation history when relevant. "
            "If the user asks a question that would be better answered with a visualization, "
            "suggest they try one of these commands."
        )

        # Use LangSmith thread management for conversation continuity
        current_persona = PERSONAS[current_persona_name]

        # Check if we should use conversation history (after first message in session)
        session_metadata = get_session_metadata(session_id)
        use_history = session_metadata["conversation_count"] > 0

        # Determine if this query would benefit from real-time search
        search_triggers = [
            "market", "competition", "trends", "latest", "recent", "current", "2024", "2025",
            "startup", "funding", "investment", "industry", "valuation", "growth", "exit"
        ]

        should_search = any(trigger in message.content.lower() for trigger in search_triggers)

        if should_search and search_api_key:
            # Generate persona-specific search query
            search_query = generate_search_query(message.content, current_persona_name)

            if search_query:
                # Show search indicator
                search_msg = await cl.Message(content="🔍 Searching for latest market intelligence...").send()

                # Perform search
                search_results = search_internet(search_query, count=3)

                # Remove search indicator
                await search_msg.remove()

                if search_results["success"]:
                    search_context = f"\n\nRECENT MARKET INTELLIGENCE:\n"
                    for i, result in enumerate(search_results["results"], 1):
                        search_context += f"\n{i}. **{result['title']}**\n"
                        search_context += f"   {result['description']}\n"
                        search_context += f"   Source: {result['url']}\n"

        # Enhance user question with dataset and search context
        enhanced_question = f"Dataset:\n{df_str}\n\nUser question: {message.content}"
        if search_context:
            enhanced_question += search_context

        # Use the LangSmith chat pipeline for thread-aware responses
        ai_response = navada_chat_pipeline(
            question=enhanced_question,
            session_id=session_id,
            persona=current_persona_name,
            get_chat_history=use_history
        )

    # -------------------------
    # SEND AI RESPONSE WITH PERSONA INDICATOR
    # -------------------------
    # Add AI response to memory
    add_to_memory(session_id, "assistant", ai_response)

    # Save AI response to database if authenticated
    if AUTH_AVAILABLE and auth_status["authenticated"]:
        auth_manager.save_conversation(
            user_id=auth_status["user_id"],
            chainlit_session_id=session_id,
            role="assistant",
            content=ai_response,
            persona_mode=get_current_persona()["name"].lower().replace(" mode", ""),
            metadata={
                "search_used": bool(search_context),
                "search_results_count": len(search_results.get("results", [])) if search_results else 0,
                "persona": current_persona_name,
                "enhanced_with_search": bool(search_context and search_results and search_results["success"])
            }
        )

    # Add persona indicator to response (with safety check)
    if not ai_response or ai_response.strip() == "":
        ai_response = "I apologize, but I encountered an issue generating a response. Please try again."

    response_with_persona = ai_response

    # Add search intelligence indicator if search was used
    if search_context and search_results and search_results["success"]:
        response_with_persona += f"\n\n---\n\n*🔍 Enhanced with real-time market intelligence from {len(search_results['results'])} sources*"

    # Generate and append auto-insights for analysis-type responses
    if any(keyword in user_input for keyword in ["analyze", "analysis", "compare", "evaluate"]):
        insights = generate_insights(df, "analysis")
        if insights["risks"] or insights["opportunities"] or insights["recommendations"]:
            response_with_persona += "\n\n" + format_insights_message(insights)

    # Remove thinking indicator before sending final response
    await thinking_msg.remove()

    # Send the text response
    message = await cl.Message(content=response_with_persona).send()

    # -------------------------
    # AUTO TEXT-TO-SPEECH
    # -------------------------
    # Check if TTS is enabled in user settings
    tts_enabled = cl.user_session.get("tts_enabled", False)
    if tts_enabled and response_with_persona:
        try:
            # Clean response text for TTS (remove markdown, emojis, etc.)
            clean_text = clean_text_for_tts(response_with_persona)
            if clean_text.strip():
                # Generate and send audio
                audio_element = await generate_speech(clean_text)
                if audio_element:
                    await cl.Message(
                        content="🔊 Audio version:",
                        elements=[audio_element]
                    ).send()
        except Exception as e:
            print(f"⚠️ Auto-TTS failed: {e}")

def clean_text_for_tts(text: str) -> str:
    """Clean text for text-to-speech by removing markdown and special characters."""
    import re

    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'`(.*?)`', r'\1', text)        # Code
    text = re.sub(r'#+ ', '', text)               # Headers
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)    # Links
    text = re.sub(r'---+', '', text)              # Horizontal rules

    # Remove emojis and special characters
    text = re.sub(r'[🔍🚀💼📊🔹⚡📈🎯💡🔧📚⏰📝🔗✅⚠️🎙️🔊📢]', '', text)

    # Clean up multiple spaces and newlines
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)

    # Limit length for TTS
    return text.strip()[:1000]

# =============================
# LANGSMITH PLATFORM OPTIMIZATION
# =============================

@traceable
def initialize_knowledge_base():
    """Initialize vector store with external LangChain database or fallback to local."""
    global vector_store

    if not vector_store:
        # Try to connect to external LangChain database first
        if langchain_database_id and CHROMA_AVAILABLE:
            try:
                # Connect to external LangChain database using the provided ID
                vector_store = Chroma(
                    embedding_function=embeddings,
                    persist_directory=f"./langchain_db_{langchain_database_id}"
                )
                print(f"✅ Connected to LangChain database: {langchain_database_id[:8]}...")
            except Exception as e:
                print(f"⚠️ Failed to connect to LangChain database: {e}")
                # Fallback to local knowledge base
                vector_store = _create_local_knowledge_base()
        else:
            # Fallback to local knowledge base
            vector_store = _create_local_knowledge_base()

    return vector_store

def _create_local_knowledge_base():
    """Create local knowledge base as fallback."""
    # Startup knowledge optimized for LangSmith platform
    startup_knowledge = [
        "Successful startups show product-market fit within 18-24 months",
        "SaaS startups should aim for 20% month-over-month growth",
        "B2B startups need longer sales cycles but higher LTV",
        "Consumer apps require viral growth and strong engagement",
        "Hardware startups need more capital and longer dev cycles",
        "Fintech faces regulatory challenges but high market opportunity",
        "AI/ML startups need strong technical teams and data advantages",
        "E-commerce should focus on unit economics and CAC",
        "Database ID: " + (langchain_database_id or "local-fallback")
    ]

    documents = [Document(page_content=text) for text in startup_knowledge]

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

@traceable
def enhanced_rag_response(user_query: str, context: str) -> str:
    """Generate enhanced response using RAG system optimized for LangSmith."""
    if not vector_store:
        initialize_knowledge_base()

    # Query knowledge base
    docs = vector_store.similarity_search(user_query, k=3)
    relevant_knowledge = [doc.page_content for doc in docs]

    # Enhanced prompt with RAG context
    rag_prompt = f"""
    Based on startup knowledge and context, provide actionable insights:

    Relevant Knowledge:
    {chr(10).join(relevant_knowledge)}

    Context: {context}
    Query: {user_query}

    Provide detailed, actionable response with specific recommendations.
    """

    response = llm.invoke(rag_prompt)
    return response.content

# Health check endpoint for LangSmith platform
@cl.on_settings_update
async def health_check():
    """Health check endpoint for LangSmith monitoring."""
    return {"status": "healthy", "app": "NAVADA", "version": "1.0.0"}

@cl.on_chat_start
async def init_navada_langsmith():
    """Initialize NAVADA for LangSmith platform deployment."""
    try:
        # Initialize knowledge base
        initialize_knowledge_base()

        # Send simple welcome message
        await cl.Message(content="**NAVADA**").send()

    except Exception as e:
        await cl.Message(content=f"⚠️ Initialization issue: {str(e)}. Using standard mode.").send()

# =============================
# LANGGRAPH AGENT EXPORT
# =============================
# Export agent for LangGraph deployment
# agent = cl  # Commented out - incomplete line