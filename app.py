# app.py
# Startup Viability Agent - A Chainlit-powered chatbot for analyzing startup risk and failure patterns

# =============================
# IMPORTS
# =============================
import io  # For in-memory file operations (byte streams)
import math  # Mathematical operations (currently unused but available)
import json  # JSON parsing (currently unused but available)
import asyncio  # Async/await support for concurrent operations
import chainlit as cl  # Chainlit framework for building conversational AI interfaces
import pandas as pd  # Data manipulation and analysis with DataFrames
import matplotlib.pyplot as plt  # Core plotting library for creating visualizations
import seaborn as sns  # Statistical data visualization built on matplotlib
import plotly.express as px  # Interactive plotting library for dynamic visualizations
import plotly.graph_objects as go  # Low-level plotly interface for custom charts
import plotly.io as pio  # Plotly I/O utilities for saving/converting charts
import requests  # HTTP library for making web requests and scraping
from bs4 import BeautifulSoup  # HTML/XML parser for web scraping
from urllib.parse import urlparse  # URL validation and parsing utilities
import re  # Regular expressions for text processing and validation

from typing import Dict, Any, List  # Type hints for better code documentation
from openai import OpenAI  # OpenAI API client for GPT model interactions
from dotenv import load_dotenv  # Load environment variables from .env file
from IPython.display import display  # IPython display utilities (not actively used)
from sklearn.model_selection import train_test_split  # Split data for ML training
from sklearn.ensemble import RandomForestClassifier  # Random Forest model for predictions

# LangChain & LangSmith imports for hosting
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langsmith import traceable
import os

# =============================
# INITIAL SETUP & CONFIGURATION
# =============================
# Load environment variables (OPENAI_API_KEY) from .env file
# This keeps sensitive API keys out of the source code
load_dotenv()

# Initialize OpenAI client for making API calls to GPT models
# The API key is automatically pulled from the OPENAI_API_KEY environment variable
# If not found in env, OpenAI SDK will look for it in default locations
client = OpenAI()

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
# Store conversation history and persona settings per session
SESSION_MEMORY = {}  # Stores conversation history per session ID
PERSONAS = {
    "investor": {
        "name": "Investor Mode",
        "system_prompt": (
            "You are a seasoned venture capitalist with 15+ years experience. "
            "Focus on ROI, market size, competitive analysis, and exit strategies. "
            "Be direct, data-driven, and ask tough financial questions. "
            "Suggest concrete metrics to track and milestones for funding rounds."
        ),
        "style": "💼 **INVESTOR MODE** - VC perspective",
        "color": "🔵"
    },
    "founder": {
        "name": "Founder Mode",
        "system_prompt": (
            "You are an experienced startup founder who's built multiple companies. "
            "Focus on practical execution, team building, product development, "
            "and overcoming early-stage challenges. Be supportive, strategic, "
            "and share tactical advice based on real founder experiences."
        ),
        "style": "🚀 **FOUNDER MODE** - Entrepreneur perspective",
        "color": "🟢"
    }
}

# =============================
# SAMPLE DATASET - STARTUP DATA
# =============================
# Create a comprehensive fake dataset with 12 startups across various sectors
# This dataset includes multiple dimensions of startup metrics for analysis:
# - Financial: Funding amount, burn rate
# - Team: Founder experience
# - Market: Market size, sector, geography
# - Outcome: Success/failure status
data = {
    "Startup": [
        "TechX", "Foodly", "EcoGo", "EduSmart", "MediAI", "FinSolve", "Healthify",
        "GreenCore", "LogistiChain", "RoboAssist", "NeuroStream", "ByteCart"
    ],
    # Funding amounts in millions USD - represents total funding raised
    "Funding_USD_M":       [5.0, 1.2, 0.8, 3.0, 12.0, 7.5, 4.2, 9.8, 15.0, 6.6, 18.0, 2.5],

    # Burn rate in months - how many months the funding lasts at current spending
    # Lower burn rate = burning through money faster
    "Burn_Rate_Months":    [12, 6, 3, 9, 24, 18, 10, 15, 30, 8, 26, 7],

    # Average years of experience across founding team members
    "Founders_Experience_Yrs":[2, 1, 0, 3, 8, 5, 6, 4, 10, 2, 7, 1],

    # Total addressable market size in billions USD
    "Market_Size_Bn":      [50, 5, 2, 15, 80, 60, 25, 40, 100, 20, 120, 8],

    # Binary outcome: 1 = failed, 0 = still operating/successful
    "Failed":              [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],

    # Country codes (UK, DE=Germany, FR=France, US=United States)
    "Country":             ["UK", "UK", "UK", "UK", "DE", "FR", "US", "UK", "US", "UK", "US", "UK"],

    # Industry sector classification
    "Sector":              ["Tech","Food","Transport","EdTech","HealthTech","FinTech",
                            "HealthTech","Energy","Logistics","Robotics","HealthTech","Retail"]
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
# UTILITY FUNCTIONS - CHART GENERATION
# =============================

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
    # Create figure with specified size (9 inches wide, 5 inches tall)
    fig, ax = plt.subplots(figsize=(9, 5))

    # Create bar plot using seaborn for better styling
    # Coolwarm palette: cooler colors for later failure, warmer for sooner
    sns.barplot(data=df_in, x="Startup", y="Est_Failure_Year", palette="coolwarm", ax=ax)

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
    fig, ax = plt.subplots(figsize=(9, 5))

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
    # Create compact figure for gauge display
    fig, ax = plt.subplots(figsize=(6, 1.2))

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
    fig, ax = plt.subplots(figsize=(10, 6))

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
    fig, ax = plt.subplots(figsize=(10, 6))

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
    fig, ax = plt.subplots(figsize=(10, 6))

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
    fig, ax = plt.subplots(figsize=(10, 6))

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
    • {len(high_risk)} startups with burn rate under 10 months (high risk)<br/>
    • {len(low_funding)} startups with funding under $3M (undercapitalized)<br/>
    • {len(inexperienced)} startups with founders having less than 3 years experience<br/><br/>

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
        • Runway of {runway_months:.1f} months provides {'adequate' if runway_months > 18 else 'limited'} time to achieve milestones<br/>
        • Market size of ${market}B offers {'strong' if market > 50 else 'moderate'} growth potential<br/>
        • Team experience of {experience} years is {'above' if experience >= 5 else 'below'} industry average<br/>
        • Current status: {status}
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
            rec_text += f"• {row['Startup']} ({row['Sector']}) - ${row['Funding_USD_M']}M funding<br/>"

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
        margin=dict(                       # Add margins for better spacing
            t=50,                          # Top margin for title
            b=50,                          # Bottom margin for annotations
            l=50,                          # Left margin for y-axis labels
            r=50                           # Right margin for legend
        )
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
        width=800,
        height=500
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
        height=800,
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

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(heatmap_df) * 0.5)))

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
# CHAINLIT EVENT HANDLERS
# =============================

@cl.on_chat_start
async def start():
    """
    Initialize the chat session when a user first connects.

    This function runs once at the start of each new chat session and:
    1. Sets up chat settings with About section and quick actions
    2. Sends a brief welcome message

    The settings panel (burger menu) contains detailed information
    about NAVADA's capabilities.

    Chainlit decorator: @cl.on_chat_start
    - Automatically called when a new chat begins
    - Async function for non-blocking UI operations
    """
    # -------------------------
    # SETUP CHAT SETTINGS WITH ABOUT SECTION
    # -------------------------
    # Create chat settings with an "About" section accessible via burger menu
    about_content = (
        "# 👋 Welcome to NAVADA\n\n"
        "**NAVADA** (Startup Viability Agent) helps you analyze startup risk, funding, and failure patterns "
        "with **interactive charts and AI analysis**.\n\n"
        "## 📊 Built-in Charts:\n\n"
        "🔹 **timeline** - Failure timeline for all startups\n"
        "🔹 **funding vs burn** - Funding vs burn rate scatter plot\n"
        "🔹 **sector comparison chart** - Average funding by sector\n"
        "🔹 **failure rate by country** - Failure rates per country\n"
        "🔹 **experience vs success chart** - Founder experience analysis\n\n"
        "## 🤖 AI-Powered Features:\n\n"
        "🔹 **assess idea** - Interactive viability scoring\n"
        "🔹 **benchmark idea** - Compare your startup against dataset averages\n"
        "🔹 **portfolio mode** - Analyze multiple startups with heatmap visualization\n"
        "🔹 **generate report** - Create comprehensive PDF investment analysis\n"
        "🔹 **Natural language charts** - Ask me to 'show a bar chart of...' and I'll generate it!\n"
        "🔹 **upload csv** - Replace dataset with your own data\n\n"
        "## 💬 Sample Questions:\n\n"
        "• Which sector looks riskiest?\n"
        "• Show me a pie chart of funding by sector\n"
        "• Visualize the relationship between market size and failure\n"
        "• Compare UK vs US startup survival rates\n\n"
        "---\n\n"
        "**Ready to start?** Type a command, ask for a chart, or ask me anything!"
    )

    # Create settings panel with About section and quick actions
    settings = await cl.ChatSettings(
        [
            cl.input_widget.TextInput(
                id="about",
                label="📚 About NAVADA",
                initial=about_content,
                description="Learn about NAVADA's capabilities and features",
            ),
            cl.input_widget.Select(
                id="quick_actions",
                label="⚡ Quick Actions",
                values=["timeline", "funding vs burn", "sector comparison chart", "assess idea", "upload csv"],
                initial_index=0,
                description="Select a command to execute quickly"
            ),
        ]
    ).send()

    # -------------------------
    # SEND BRIEF WELCOME MESSAGE
    # -------------------------
    # Send agent name only - no intro text per user request
    welcome = "**NAVADA**"

    # Send the welcome message asynchronously to the UI
    await cl.Message(content=welcome).send()


@cl.on_settings_update
async def settings_update(settings):
    """
    Handle settings panel updates.

    This function is called when users interact with the settings panel
    (burger menu). If they select a quick action, execute it.

    Args:
        settings: Dictionary containing updated settings values
    """
    # Check if user selected a quick action
    if settings.get("quick_actions"):
        action = settings["quick_actions"]
        # Send the action as if the user typed it
        await cl.Message(content=f"Executing: {action}").send()
        # Note: The actual command will be processed by the @cl.on_message handler


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

        # Send descriptive text message
        text_msg = cl.Message(
            content=(
                "### 📈 Estimated Failure Timeline\n\n"
                "This chart shows when each startup is projected to fail "
                "based on their funding and burn rate."
            )
        )
        await text_msg.send()

        # Attach chart image to the text message
        # for_id links the image to the parent message
        image = cl.Image(content=png, name="failure_timeline.png", display="inline")
        await image.send(for_id=text_msg.id)

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
    # ROUTE 9: PERSONA MANAGEMENT
    # =============================
    if "investor mode" in user_input or "switch to investor" in user_input:
        # Remove thinking indicator
        await thinking_msg.remove()

        cl.user_session.set("persona", "investor")
        persona = get_current_persona()
        await cl.Message(
            content=f"{persona['color']} {persona['style']}\n\n"
                   "I'm now analyzing from a **venture capitalist perspective**. "
                   "I'll focus on ROI, market size, competitive analysis, and exit strategies.\n\n"
                   "**What would you like to analyze today?**\n\n"
                   "💰 **Quick Options:**\n"
                   "• Type **'portfolio'** - Analyze all startups with investment recommendations\n"
                   "• Type **'insights'** - Get AI-powered risk assessment and opportunities\n"
                   "• Type **'benchmark'** - Compare a new startup idea against our data\n"
                   "• Type **'sector dashboard'** - Interactive sector performance analysis\n"
                   "• Type **'scrape <url>'** - Analyze competitor websites for investment insights\n\n"
                   "📊 **Or ask me directly:**\n"
                   "• \"Which startups have the best ROI potential?\"\n"
                   "• \"What sectors should I avoid investing in?\"\n"
                   "• \"Show me failure rates by market size\""
        ).send()
        return

    if "founder mode" in user_input or "switch to founder" in user_input:
        # Remove thinking indicator
        await thinking_msg.remove()

        cl.user_session.set("persona", "founder")
        persona = get_current_persona()
        await cl.Message(
            content=f"{persona['color']} {persona['style']}\n\n"
                   "I'm now analyzing from an **experienced founder perspective**. "
                   "I'll focus on practical execution, team building, and tactical advice.\n\n"
                   "**What challenges can I help you tackle today?**\n\n"
                   "🚀 **Quick Options:**\n"
                   "• Type **'assess idea'** - Get a viability score for your startup concept\n"
                   "• Type **'benchmark'** - See how your metrics compare to successful startups\n"
                   "• Type **'insights'** - Get tactical recommendations to reduce risk\n"
                   "• Type **'timeline'** - See failure patterns to avoid common pitfalls\n"
                   "• Type **'scrape <url>'** - Learn from competitor websites and strategies\n\n"
                   "💡 **Or ask me directly:**\n"
                   "• \"How can I extend my runway?\"\n"
                   "• \"What are the biggest risks I should watch for?\"\n"
                   "• \"Which successful startups are similar to mine?\""
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
                   "**Ready to switch?**\n"
                   "• Type **'investor mode'** for VC analysis\n"
                   "• Type **'founder mode'** for founder guidance\n\n"
                   "**Or continue in current mode - what would you like to analyze?**"
        ).send()
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
        analysis_response = f"{persona['color']} **Website Analysis**\n\n{analysis}"
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
    # ROUTE 11: AI-POWERED Q&A WITH MEMORY & PERSONA (DEFAULT)
    # =============================
    # If no specific pattern matched, use GPT-4 for natural language response

    # -------------------------
    # SESSION MEMORY & PERSONA INTEGRATION
    # -------------------------
    session_id = get_session_id()
    persona = get_current_persona()
    memory_context = get_memory_context(session_id)

    # Add user message to memory
    add_to_memory(session_id, "user", message.content)

    # -------------------------
    # PREPARE ENHANCED CONTEXT
    # -------------------------
    # Convert DataFrame to string for inclusion in prompt
    df_str = df.to_string(index=False)

    # -------------------------
    # CALL OPENAI API WITH PERSONA & MEMORY
    # -------------------------
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

    messages = [
        {
            "role": "system",
            "content": enhanced_system_prompt
        }
    ]

    # Add memory context if available
    if memory_context:
        messages.append({
            "role": "system",
            "content": f"Conversation context:\n{memory_context}"
        })

    messages.append({
        "role": "user",
        "content": f"Dataset:\n{df_str}\n\nUser question: {message.content}"
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500
    )

    # -------------------------
    # SEND AI RESPONSE WITH PERSONA INDICATOR
    # -------------------------
    ai_response = response.choices[0].message.content

    # Add AI response to memory
    add_to_memory(session_id, "assistant", ai_response)

    # Add persona indicator to response
    response_with_persona = f"{persona['color']} {ai_response}"

    # Generate and append auto-insights for analysis-type responses
    if any(keyword in user_input for keyword in ["analyze", "analysis", "compare", "evaluate"]):
        insights = generate_insights(df, "analysis")
        if insights["risks"] or insights["opportunities"] or insights["recommendations"]:
            response_with_persona += "\n\n" + format_insights_message(insights)

    # Remove thinking indicator before sending final response
    await thinking_msg.remove()

    await cl.Message(content=response_with_persona).send()

# =============================
# LANGSMITH PLATFORM OPTIMIZATION
# =============================

@traceable
def initialize_knowledge_base():
    """Initialize vector store with startup knowledge for LangSmith hosting."""
    global vector_store

    # Startup knowledge optimized for LangSmith platform
    startup_knowledge = [
        "Successful startups show product-market fit within 18-24 months",
        "SaaS startups should aim for 20% month-over-month growth",
        "B2B startups need longer sales cycles but higher LTV",
        "Consumer apps require viral growth and strong engagement",
        "Hardware startups need more capital and longer dev cycles",
        "Fintech faces regulatory challenges but high market opportunity",
        "AI/ML startups need strong technical teams and data advantages",
        "E-commerce should focus on unit economics and CAC"
    ]

    documents = [Document(page_content=text) for text in startup_knowledge]

    if not vector_store:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

    return vector_store

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

        # LangSmith-specific welcome message
        welcome_msg = """
🚀 **NAVADA - Startup Viability Agent**
*Powered by LangSmith Platform*

🎯 **Advanced Features Available:**
- **RAG-Enhanced Analysis** - Intelligent startup knowledge base
- **Real-time Tracing** - Every interaction monitored
- **Conversation Memory** - Persistent across sessions
- **Performance Analytics** - Optimized for scale

Type any command to get started, or try:
- `investor mode` - Switch to VC perspective
- `founder mode` - Switch to entrepreneur view
- `help` - See all available commands

*Ready to analyze your startup's viability! 📊*
        """

        await cl.Message(content=welcome_msg).send()

    except Exception as e:
        await cl.Message(content=f"⚠️ Initialization issue: {str(e)}. Using standard mode.").send()

# =============================
# LANGGRAPH AGENT EXPORT
# =============================
# Export agent for LangGraph deployment
agent = cl