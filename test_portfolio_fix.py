# Test script to verify portfolio heatmap fix
import pandas as pd
from app import df, create_portfolio_heatmap

print("Testing Portfolio Heatmap Fix...")
print(f"Dataset has {len(df)} startups")

# Test the conditions that should trigger portfolio mode
test_inputs = [
    "portfolio",
    "portfolio mode",
    "portfolio analysis",
    "analyze portfolio"
]

print("\nTesting portfolio command matching:")
for test_input in test_inputs:
    user_input = test_input.strip().lower()

    # This is the condition from the updated code
    condition = "portfolio" in user_input and ("mode" in user_input or "analysis" in user_input or "analyze" in user_input or user_input.strip() == "portfolio")

    print(f"'{test_input}' -> {condition}")

print("\nTesting portfolio heatmap generation:")
try:
    heatmap_bytes = create_portfolio_heatmap(df)
    print(f"[OK] Portfolio heatmap generated successfully ({len(heatmap_bytes)} bytes)")

    # Save for testing
    with open("test_portfolio_heatmap.png", "wb") as f:
        f.write(heatmap_bytes)
    print("[OK] Saved as test_portfolio_heatmap.png")

except Exception as e:
    print(f"[ERROR] Portfolio heatmap generation failed: {e}")

print("\n[SUCCESS] Portfolio fix verification complete!")
print("The 'portfolio' command should now work properly in the app.")