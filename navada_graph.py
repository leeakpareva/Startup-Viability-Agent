from typing import TypedDict, Dict, Any, List
from langgraph.graph import Graph, END
from langgraph.graph.graph import CompiledGraph

class StartupState(TypedDict):
    """State for startup analysis"""
    request: str
    startup_data: Dict[str, Any]
    analysis_type: str
    results: Dict[str, Any]
    response: str

def parse_request(state: StartupState) -> StartupState:
    """Parse and route the request"""
    request = state.get("request", "").lower()

    if "viability" in request:
        state["analysis_type"] = "viability"
    elif "risk" in request:
        state["analysis_type"] = "risk"
    elif "financial" in request or "projection" in request:
        state["analysis_type"] = "financial"
    else:
        state["analysis_type"] = "general"

    return state

def analyze_viability(state: StartupState) -> StartupState:
    """Simple viability analysis"""
    data = state.get("startup_data", {})

    # Simple scoring logic
    funding = data.get("funding_usd_m", 5)
    burn = data.get("burn_rate_months", 12)
    runway = (funding * 12) / max(burn, 1)

    score = min(100, runway * 5)  # Simple score based on runway

    state["results"]["viability"] = {
        "score": score,
        "runway_months": runway,
        "risk_level": "High" if score < 40 else "Medium" if score < 60 else "Low"
    }

    return state

def analyze_risk(state: StartupState) -> StartupState:
    """Simple risk analysis"""
    data = state.get("startup_data", {})

    # Simple risk calculation
    funding = data.get("funding_usd_m", 5)
    market = data.get("market_size_bn", 10)

    risk_score = 100 - (min(funding * 10, 50) + min(market * 5, 50))

    state["results"]["risk"] = {
        "composite_risk": risk_score,
        "level": "Critical" if risk_score > 70 else "High" if risk_score > 50 else "Moderate"
    }

    return state

def analyze_financial(state: StartupState) -> StartupState:
    """Simple financial projection"""
    data = state.get("startup_data", {})

    mrr = data.get("mrr_k", 10) * 1000
    growth = data.get("growth_rate", 10) / 100

    # Project 12 months
    projected_mrr = mrr * ((1 + growth) ** 12)

    state["results"]["financial"] = {
        "current_mrr": mrr,
        "projected_mrr_12m": projected_mrr,
        "growth_rate": growth * 100
    }

    return state

def format_response(state: StartupState) -> StartupState:
    """Format the final response"""
    results = state.get("results", {})
    analysis_type = state.get("analysis_type", "general")

    response = f"Analysis Type: {analysis_type}\n\n"

    if "viability" in results:
        v = results["viability"]
        response += f"Viability Score: {v['score']:.1f}/100\n"
        response += f"Runway: {v['runway_months']:.1f} months\n"
        response += f"Risk Level: {v['risk_level']}\n"

    if "risk" in results:
        r = results["risk"]
        response += f"Risk Score: {r['composite_risk']:.1f}/100\n"
        response += f"Risk Level: {r['level']}\n"

    if "financial" in results:
        f = results["financial"]
        response += f"Current MRR: ${f['current_mrr']:,.0f}\n"
        response += f"Projected MRR (12m): ${f['projected_mrr_12m']:,.0f}\n"
        response += f"Growth Rate: {f['growth_rate']:.1f}%\n"

    state["response"] = response
    return state

def create_navada_graph() -> CompiledGraph:
    """Create the NAVADA analysis graph"""
    workflow = Graph()

    # Initialize results dict in state
    def initialize_state(state: StartupState) -> StartupState:
        if "results" not in state:
            state["results"] = {}
        return state

    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("parse_request", parse_request)
    workflow.add_node("analyze_viability", analyze_viability)
    workflow.add_node("analyze_risk", analyze_risk)
    workflow.add_node("analyze_financial", analyze_financial)
    workflow.add_node("format_response", format_response)

    # Add edges
    workflow.add_edge("initialize", "parse_request")

    # Conditional routing based on analysis type
    def route_analysis(state: StartupState):
        analysis_type = state.get("analysis_type", "general")
        if analysis_type == "viability":
            return "analyze_viability"
        elif analysis_type == "risk":
            return "analyze_risk"
        elif analysis_type == "financial":
            return "analyze_financial"
        else:
            return "analyze_viability"  # Default

    workflow.add_conditional_edges(
        "parse_request",
        route_analysis,
        {
            "analyze_viability": "analyze_viability",
            "analyze_risk": "analyze_risk",
            "analyze_financial": "analyze_financial"
        }
    )

    # All analysis nodes lead to formatting
    workflow.add_edge("analyze_viability", "format_response")
    workflow.add_edge("analyze_risk", "format_response")
    workflow.add_edge("analyze_financial", "format_response")

    # Format leads to end
    workflow.add_edge("format_response", END)

    # Set entry point
    workflow.set_entry_point("initialize")

    return workflow.compile()

# Create the graph instance that LangGraph expects
graph = create_navada_graph()