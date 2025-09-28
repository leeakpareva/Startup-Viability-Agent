# Test script for new NAVADA features
import pandas as pd
from app import benchmark_founder_idea, create_portfolio_heatmap, df

print("Testing New NAVADA Features...")
print(f"Dataset has {len(df)} startups")

# Test 1: Founder Idea Benchmarking
print("\n1. Testing Founder Idea Benchmarking...")
try:
    test_features = {
        'funding_usd_m': 8.0,
        'burn_rate_months': 6.0,
        'team_experience_years': 2.0,
        'market_size_bn': 80.0
    }

    benchmark_results = benchmark_founder_idea(test_features, df)

    print(f"[OK] Benchmarking completed successfully")
    print(f"    Risk Level: {benchmark_results['risk_level']}")
    print(f"    Insights: {len(benchmark_results['insights'])} generated")
    print(f"    Recommendations: {len(benchmark_results['recommendations'])} provided")

    # Show sample insight
    if benchmark_results['insights']:
        print(f"    Sample insight: {benchmark_results['insights'][0][:50]}...")

except Exception as e:
    print(f"[ERROR] Benchmarking failed: {e}")

# Test 2: Portfolio Heatmap
print("\n2. Testing Portfolio Heatmap Generation...")
try:
    heatmap_bytes = create_portfolio_heatmap(df)

    with open("Portfolio_Heatmap_Test.png", "wb") as f:
        f.write(heatmap_bytes)

    print(f"[OK] Heatmap generated successfully ({len(heatmap_bytes)} bytes)")
    print(f"    Saved as Portfolio_Heatmap_Test.png")

except Exception as e:
    print(f"[ERROR] Heatmap generation failed: {e}")

print("\n[OK] All new feature tests completed!")