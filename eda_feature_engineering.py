import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD SIMULATION RESULTS
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING SIMULATION RESULTS")
print("=" * 60)

sim_df      = pd.read_csv("/home/outputs/simulation_results.csv")
baseline_df = pd.read_csv("/home/outputs/baseline_performance.csv")
route_perf  = pd.read_csv("/home/outputs/route_performance.csv")

print(f"Simulation scenarios : {len(sim_df)}")
print(f"Baseline products    : {len(baseline_df)}")
print(f"Route combinations   : {len(route_perf)}")

# ─────────────────────────────────────────────
# 2. SCORING FUNCTION
# Composite score = weighted sum of 3 KPIs
#   - Lead Time Reduction %    (40%)
#   - Distance Reduction       (30%)
#   - Profit Stability         (30%)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: COMPOSITE OPTIMIZATION SCORING")
print("=" * 60)

# Keep only beneficial reassignments (LT_Reduction > 0)
good = sim_df[sim_df["LT_Reduction"] > 0].copy()

# Normalize each component to 0-1
def normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

good["Score_LT"]     = normalize(good["LT_Reduction_Pct"])
good["Score_Dist"]   = normalize(good["Dist_Reduction_km"])
good["Score_Profit"] = normalize(good["New_Est_Profit"])

# Default weights
W_LT, W_DIST, W_PROFIT = 0.40, 0.30, 0.30

good["Composite_Score"] = (
    W_LT     * good["Score_LT"]     +
    W_DIST   * good["Score_Dist"]   +
    W_PROFIT * good["Score_Profit"]
).round(4)

print(f"Scored {len(good)} beneficial scenarios")
print(f"Composite Score range: {good['Composite_Score'].min():.4f} – {good['Composite_Score'].max():.4f}")

# ─────────────────────────────────────────────
# 3. TOP-N FACTORY REASSIGNMENT RECOMMENDATIONS
# Best recommendation per product (globally)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: TOP FACTORY REASSIGNMENT RECOMMENDATIONS")
print("=" * 60)

top_per_product = (
    good.sort_values("Composite_Score", ascending=False)
    .groupby("Product")
    .first()
    .reset_index()
)

rec_cols = [
    "Product", "Current_Factory", "Alt_Factory", "Region", "Ship_Mode",
    "Baseline_LT", "Predicted_LT", "LT_Reduction_Pct",
    "Dist_Reduction_km", "Baseline_Profit", "New_Est_Profit",
    "Composite_Score", "Confidence_Score"
]

print("\n🏆 TOP RECOMMENDATION PER PRODUCT:")
print(top_per_product[rec_cols].sort_values(
    "Composite_Score", ascending=False
).to_string(index=False))

# ─────────────────────────────────────────────
# 4. TOP-5 GLOBAL RECOMMENDATIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: TOP-5 GLOBAL RECOMMENDATIONS")
print("=" * 60)

top5 = good.sort_values("Composite_Score", ascending=False).head(5)
print(top5[rec_cols].to_string(index=False))

# ─────────────────────────────────────────────
# 5. SPEED vs PROFIT PRIORITY SCENARIOS
# Priority Slider = 0 (profit first) to 1 (speed first)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: PRIORITY SCENARIOS — SPEED vs PROFIT")
print("=" * 60)

def score_with_priority(df, speed_weight):
    profit_weight = 1 - speed_weight
    w_lt     = speed_weight  * 0.70
    w_dist   = speed_weight  * 0.30
    w_profit = profit_weight
    df = df.copy()
    df["Priority_Score"] = (
        w_lt     * df["Score_LT"]     +
        w_dist   * df["Score_Dist"]   +
        w_profit * df["Score_Profit"]
    ).round(4)
    return df.sort_values("Priority_Score", ascending=False)

for label, sw in [("Speed Priority (0.9)", 0.9),
                   ("Balanced (0.5)",       0.5),
                   ("Profit Priority (0.1)", 0.1)]:
    ranked = score_with_priority(good, sw)
    top3   = ranked.groupby("Product").first().reset_index().head(3)
    print(f"\n── {label} ──")
    print(top3[["Product", "Alt_Factory", "LT_Reduction_Pct",
                 "New_Est_Profit", "Priority_Score"]].to_string(index=False))

# ─────────────────────────────────────────────
# 6. RISK ASSESSMENT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: RISK ASSESSMENT")
print("=" * 60)

# High risk = profit drops AND low confidence
sim_df["Risk_Level"] = "Low"
sim_df.loc[
    (sim_df["Profit_Risk"] == True) & (sim_df["Confidence_Score"] < 0.2),
    "Risk_Level"
] = "High"
sim_df.loc[
    (sim_df["Profit_Risk"] == True) & (sim_df["Confidence_Score"] >= 0.2),
    "Risk_Level"
] = "Medium"

risk_summary = sim_df.groupby("Risk_Level")["Product"].count().rename("Count")
print("\nRisk Level Distribution:")
print(risk_summary.to_string())

high_risk = sim_df[sim_df["Risk_Level"] == "High"][[
    "Product", "Current_Factory", "Alt_Factory", "Region",
    "LT_Reduction_Pct", "Baseline_Profit", "New_Est_Profit", "Confidence_Score"
]].sort_values("LT_Reduction_Pct", ascending=False)

print(f"\nHigh Risk Scenarios ({len(high_risk)}):")
print(high_risk.head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 7. KPI SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: KPI SUMMARY")
print("=" * 60)

beneficial = sim_df[sim_df["LT_Reduction"] > 0]

kpi = {
    "Total Scenarios Evaluated":       len(sim_df),
    "Beneficial Reassignments":        len(beneficial),
    "Avg Lead Time Reduction (days)":  round(beneficial["LT_Reduction"].mean(), 2),
    "Max Lead Time Reduction (days)":  round(beneficial["LT_Reduction"].max(), 2),
    "Avg LT Reduction (%)":            round(beneficial["LT_Reduction_Pct"].mean(), 2),
    "Max LT Reduction (%)":            round(beneficial["LT_Reduction_Pct"].max(), 2),
    "Profit-Stable Scenarios":         int((sim_df["Profit_Risk"] == False).sum()),
    "High Risk Scenarios":             int((sim_df["Risk_Level"] == "High").sum()),
    "Avg Confidence Score":            round(beneficial["Confidence_Score"].mean(), 4),
    "Products with Recommendations":   beneficial["Product"].nunique(),
    "Recommendation Coverage (%)":     round(beneficial["Product"].nunique() / sim_df["Product"].nunique() * 100, 2),
}

kpi_df = pd.DataFrame(list(kpi.items()), columns=["KPI", "Value"])
print(kpi_df.to_string(index=False))

# ─────────────────────────────────────────────
# 8. FINAL RANKED RECOMMENDATION TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: FINAL RANKED RECOMMENDATIONS (ALL PRODUCTS)")
print("=" * 60)

final_recs = (
    good.sort_values("Composite_Score", ascending=False)
    .groupby(["Product", "Region"])
    .first()
    .reset_index()
    .sort_values("Composite_Score", ascending=False)
)

final_recs["Rank"] = range(1, len(final_recs) + 1)

print(final_recs[[
    "Rank", "Product", "Region", "Current_Factory", "Alt_Factory",
    "LT_Reduction_Pct", "Dist_Reduction_km", "New_Est_Profit",
    "Composite_Score", "Confidence_Score"
]].head(25).to_string(index=False))

# ─────────────────────────────────────────────
# 9. SAVE ALL OUTPUTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: SAVING OUTPUTS")
print("=" * 60)

os.makedirs("/home/outputs", exist_ok=True)

good.to_csv("/home/outputs/scored_scenarios.csv", index=False)
top_per_product.to_csv("/home/outputs/top_recommendation_per_product.csv", index=False)
top5.to_csv("/home/outputs/top5_global_recommendations.csv", index=False)
final_recs.to_csv("/home/outputs/final_ranked_recommendations.csv", index=False)
kpi_df.to_csv("/home/outputs/kpi_summary.csv", index=False)
sim_df.to_csv("/home/outputs/simulation_with_risk.csv", index=False)

print("Saved: scored_scenarios.csv")
print("Saved: top_recommendation_per_product.csv")
print("Saved: top5_global_recommendations.csv")
print("Saved: final_ranked_recommendations.csv")
print("Saved: kpi_summary.csv")
print("Saved: simulation_with_risk.csv")
print("\n✅ Optimization & Recommendation Logic Complete!")