# Vercel-optimized FastAPI app for NAVADA
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="NAVADA - Startup Analysis API",
    description="AI-Powered Startup Viability Analysis (Vercel Optimized)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================
# SIMPLIFIED MODELS
# =============================

class ViabilityRequest(BaseModel):
    funding_usd_m: float = Field(default=5.0, ge=0.1, le=1000)
    burn_rate_months: float = Field(default=12.0, ge=1, le=120)
    team_experience_years: float = Field(default=3.0, ge=0, le=50)
    market_size_bn: float = Field(default=10.0, ge=0.1, le=10000)
    business_model_strength_1_5: int = Field(default=3, ge=1, le=5)
    moat_1_5: int = Field(default=3, ge=1, le=5)
    traction_mrr_k: float = Field(default=10.0, ge=0, le=10000)
    growth_rate_pct: float = Field(default=10.0, ge=-50, le=100)
    competition_intensity_1_5: int = Field(default=3, ge=1, le=5)

# =============================
# SIMPLIFIED ANALYSIS ENGINE
# =============================

def calculate_viability_score(features: Dict[str, Any]) -> Dict[str, Any]:
    """Simplified viability scoring for Vercel deployment"""

    # Extract features with defaults
    funding = features.get('funding_usd_m', 5.0)
    burn_rate = features.get('burn_rate_months', 12.0)
    experience = features.get('team_experience_years', 3.0)
    market_size = features.get('market_size_bn', 10.0)
    business_model = features.get('business_model_strength_1_5', 3)
    moat = features.get('moat_1_5', 3)
    traction = features.get('traction_mrr_k', 10.0)
    growth_rate = features.get('growth_rate_pct', 10.0)
    competition = features.get('competition_intensity_1_5', 3)

    # Component scoring (0-1 scale)
    runway_score = min(1.0, (funding * 12 / burn_rate) / 24)  # 24 months = perfect
    market_score = min(1.0, market_size / 50)  # $50B = perfect market
    team_score = min(1.0, experience / 10)  # 10 years = perfect experience
    model_score = business_model / 5  # 5 = perfect business model
    moat_score = moat / 5  # 5 = perfect moat
    traction_score = min(1.0, traction / 100)  # $100k MRR = perfect traction
    growth_score = min(1.0, max(0, growth_rate) / 20)  # 20% = perfect growth
    competition_score = (6 - competition) / 5  # Low competition = higher score

    # Weighted composite score
    weights = {
        'runway': 0.15,
        'market': 0.15,
        'team': 0.15,
        'model': 0.15,
        'moat': 0.10,
        'traction': 0.15,
        'growth': 0.10,
        'competition': 0.05
    }

    composite_score = (
        runway_score * weights['runway'] +
        market_score * weights['market'] +
        team_score * weights['team'] +
        model_score * weights['model'] +
        moat_score * weights['moat'] +
        traction_score * weights['traction'] +
        growth_score * weights['growth'] +
        competition_score * weights['competition']
    ) * 100

    # Calculate survival months
    monthly_burn = funding * 1_000_000 / burn_rate
    monthly_revenue = traction * 1000
    net_burn = max(0, monthly_burn - monthly_revenue)
    survival_months = (funding * 1_000_000) / net_burn if net_burn > 0 else 60

    # Generate recommendations
    tips = []
    if runway_score < 0.5:
        tips.append("Extend runway: Reduce burn rate or raise additional funding")
    if traction_score < 0.3:
        tips.append("Boost traction: Focus on customer acquisition and retention")
    if market_score < 0.4:
        tips.append("Market validation: Ensure sufficient market size and demand")
    if team_score < 0.4:
        tips.append("Team strengthening: Add experienced team members or advisors")

    return {
        "score": round(composite_score, 1),
        "survival_months": round(survival_months, 1),
        "components": {
            "runway": round(runway_score * 100, 1),
            "market": round(market_score * 100, 1),
            "team": round(team_score * 100, 1),
            "business_model": round(model_score * 100, 1),
            "moat": round(moat_score * 100, 1),
            "traction": round(traction_score * 100, 1),
            "growth": round(growth_score * 100, 1),
            "competition": round(competition_score * 100, 1)
        },
        "tips": tips
    }

# =============================
# API ENDPOINTS
# =============================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "NAVADA - Startup Analysis API",
        "version": "2.0.0 (Vercel Optimized)",
        "status": "active",
        "endpoints": ["/viability", "/health", "/docs"],
        "description": "Lightweight startup viability analysis API"
    }

@app.post("/viability")
def analyze_viability(request: ViabilityRequest):
    """Analyze startup viability"""
    try:
        features = request.dict()
        result = calculate_viability_score(features)

        return {
            "success": True,
            "analysis": {
                "overall_score": result["score"],
                "interpretation": get_score_interpretation(result["score"]),
                "survival_months": result["survival_months"],
                "component_scores": result["components"],
                "recommendations": result["tips"]
            },
            "risk_level": get_risk_level(result["score"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NAVADA API (Vercel)",
        "version": "2.0.0"
    }

# =============================
# HELPER FUNCTIONS
# =============================

def get_score_interpretation(score: float) -> str:
    """Interpret viability score"""
    if score >= 80:
        return "Excellent - Strong viability with competitive advantages"
    elif score >= 65:
        return "Good - Solid fundamentals with manageable risks"
    elif score >= 50:
        return "Moderate - Mixed signals, improvements needed"
    elif score >= 35:
        return "Below Average - Significant challenges ahead"
    else:
        return "Critical - High failure risk, major changes required"

def get_risk_level(score: float) -> str:
    """Get risk level from score"""
    if score >= 70:
        return "Low Risk"
    elif score >= 50:
        return "Moderate Risk"
    elif score >= 30:
        return "High Risk"
    else:
        return "Critical Risk"

# Vercel serverless function
app_instance = app