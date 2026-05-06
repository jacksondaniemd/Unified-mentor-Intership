import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy — Factory Optimization",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FACTORIES = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176, -81.088371),
    "Sugar Shack":       (48.11914,  -96.18115),
    "Secret Factory":    (41.446333, -90.565487),
    "The Other Factory": (35.1175,   -89.971107),
}
REGION_COORDS = {
    "Atlantic": (35.0, -78.0),
    "Gulf":     (30.0, -90.0),
    "Interior": (41.0, -95.0),
    "Pacific":  (37.0, -120.0),
}
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}
COLORS = {
    "primary":   "#7C3AED",
    "secondary": "#10B981",
    "warning":   "#F59E0B",
    "danger":    "#EF4444",
    "info":      "#3B82F6",
    "bg":        "#0F172A",
    "card":      "#1E293B",
}

# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────
BASE = "/home/claude/outputs"

@st.cache_data
def load_data():
    df          = pd.read_csv(f"{BASE}/nassau_clustered.csv")
    sim         = pd.read_csv(f"{BASE}/simulation_with_risk.csv")
    recs        = pd.read_csv(f"{BASE}/final_ranked_recommendations.csv")
    top_prod    = pd.read_csv(f"{BASE}/top_recommendation_per_product.csv")
    baseline    = pd.read_csv(f"{BASE}/baseline_performance.csv")
    route_perf  = pd.read_csv(f"{BASE}/route_performance.csv")
    prod_clust  = pd.read_csv(f"{BASE}/product_clusters.csv", index_col=0)
    kpi         = pd.read_csv(f"{BASE}/kpi_summary.csv")
    feat_imp    = pd.read_csv(f"{BASE}/feature_importance.csv")
    reg_comp    = pd.read_csv(f"{BASE}/regression_comparison.csv")
    return df, sim, recs, top_prod, baseline, route_perf, prod_clust, kpi, feat_imp, reg_comp

@st.cache_resource
def load_models():
    with open(f"{BASE}/best_model.pkl",      "rb") as f: model   = pickle.load(f)
    with open(f"{BASE}/scaler.pkl",          "rb") as f: scaler  = pickle.load(f)
    with open(f"{BASE}/feature_cols.pkl",    "rb") as f: fcols   = pickle.load(f)
    with open(f"{BASE}/label_encoders.pkl",  "rb") as f: le_dict = pickle.load(f)
    return model, scaler, fcols, le_dict

df, sim, recs, top_prod, baseline, route_perf, prod_clust, kpi_df, feat_imp, reg_comp = load_data()
model, scaler, fcols, le_dict = load_models()

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def get_distance(factory, region):
    return round(geodesic(FACTORIES[factory], REGION_COORDS[region]).km, 2)

def predict_lt(product, factory, region, ship_mode):
    ref = df[df["Product Name"] == product].iloc[0]
    def enc(le, v):
        return le.transform([v])[0] if v in le.classes_ else 0
    row = {
        "Ship Mode_Enc":    enc(le_dict["Ship Mode"], ship_mode),
        "Region_Enc":       enc(le_dict["Region"], region),
        "Division_Enc":     enc(le_dict["Division"], ref["Division"]),
        "Factory_Enc":      enc(le_dict["Factory"], factory),
        "Product Name_Enc": enc(le_dict["Product Name"], product),
        "Distance_km":      get_distance(factory, region),
        "Sales":            ref["Sales"],
        "Units":            ref["Units"],
        "Cost":             ref["Cost"],
        "Profit Margin %":  ref["Profit Margin %"],
        "Revenue Per Unit": ref["Revenue Per Unit"],
        "Cost Per Unit":    ref["Cost Per Unit"],
        "Profit Per Unit":  ref["Profit Per Unit"],
        "Order Month":      ref["Order Month"],
        "Order DayOfWeek":  ref["Order DayOfWeek"],
    }
    X_row = pd.DataFrame([row])[fcols]
    return round(float(model.predict(scaler.transform(X_row))[0]), 1)

def normalize(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn) if mx != mn else pd.Series(0.5, index=series.index)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F172A; }
    .stApp { background-color: #0F172A; }
    .metric-card {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7C3AED; }
    .metric-label { font-size: 0.85rem; color: #94A3B8; margin-top: 4px; }
    .metric-delta { font-size: 0.8rem; color: #10B981; }
    .section-header {
        background: linear-gradient(90deg, #7C3AED22, transparent);
        border-left: 4px solid #7C3AED;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 10px 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #E2E8F0;
    }
    .rec-card {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .rec-rank { font-size: 1.4rem; font-weight: 800; color: #7C3AED; }
    .rec-product { font-size: 1rem; font-weight: 600; color: #E2E8F0; }
    .rec-detail { font-size: 0.8rem; color: #94A3B8; }
    .badge-green  { background:#10B98133; color:#10B981; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
    .badge-red    { background:#EF444433; color:#EF4444; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
    .badge-yellow { background:#F59E0B33; color:#F59E0B; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
    div[data-testid="stSidebarNav"] { display:none; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; }
    .stTabs [aria-selected="true"] { color: #7C3AED !important; border-bottom-color: #7C3AED !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍬 Nassau Candy")
    st.markdown("**Factory Optimization System**")
    st.divider()

    st.markdown("### 🔧 Global Filters")
    sel_region    = st.multiselect("Region", df["Region"].unique().tolist(),
                                   default=df["Region"].unique().tolist())
    sel_ship_mode = st.multiselect("Ship Mode", df["Ship Mode"].unique().tolist(),
                                   default=df["Ship Mode"].unique().tolist())
    sel_division  = st.multiselect("Division", df["Division"].unique().tolist(),
                                   default=df["Division"].unique().tolist())

    st.divider()
    speed_weight = st.slider("⚡ Optimization Priority",
                              min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                              help="0 = Profit First | 1 = Speed First")
    st.caption(f"Speed weight: {speed_weight:.1f} | Profit weight: {1-speed_weight:.1f}")
    st.divider()
    st.caption("📊 Nassau Candy Distributor")
    st.caption("Factory Reallocation & Shipping Optimization")

# Apply global filter to df
fdf = df[
    df["Region"].isin(sel_region) &
    df["Ship Mode"].isin(sel_ship_mode) &
    df["Division"].isin(sel_division)
]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='color:#7C3AED; font-size:2.2rem; margin-bottom:4px;'>
        🍬 Nassau Candy — Factory Optimization System
    </h1>
    <p style='color:#94A3B8; font-size:1rem;'>
        Predictive Shipping Intelligence · Scenario Simulation · Smart Reassignment Recommendations
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP KPI STRIP
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpi_vals = dict(zip(kpi_df["KPI"], kpi_df["Value"]))

with k1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{int(kpi_vals.get('Total Scenarios Evaluated',560))}</div>
        <div class='metric-label'>Scenarios Evaluated</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{int(kpi_vals.get('Beneficial Reassignments',214))}</div>
        <div class='metric-label'>Beneficial Reassignments</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{kpi_vals.get('Max LT Reduction (%)',94.29):.1f}%</div>
        <div class='metric-label'>Max LT Reduction</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{kpi_vals.get('Recommendation Coverage (%)',80):.0f}%</div>
        <div class='metric-label'>Product Coverage</div>
    </div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{int(kpi_vals.get('Profit-Stable Scenarios',316))}</div>
        <div class='metric-label'>Profit-Stable Moves</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 Factory Optimizer",
    "🔄 What-If Scenario",
    "🏆 Recommendations",
    "⚠️ Risk & Impact"
])

# ══════════════════════════════════════════════
# TAB 1 — FACTORY OPTIMIZER
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>🏭 Factory Performance Simulator</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        sel_product   = st.selectbox("Select Product", sorted(df["Product Name"].unique()))
        sel_region_t1 = st.selectbox("Select Region",  df["Region"].unique().tolist())
        sel_ship_t1   = st.selectbox("Select Ship Mode", df["Ship Mode"].unique().tolist())

    current_factory = PRODUCT_FACTORY[sel_product]

    # Predict lead time for all factories
    factory_preds = []
    for fac in FACTORIES:
        lt   = predict_lt(sel_product, fac, sel_region_t1, sel_ship_t1)
        dist = get_distance(fac, sel_region_t1)
        factory_preds.append({
            "Factory":     fac,
            "Predicted LT (days)": lt,
            "Distance (km)":       dist,
            "Current":     "✅ Current" if fac == current_factory else "🔄 Alternate"
        })
    fpdf = pd.DataFrame(factory_preds).sort_values("Predicted LT (days)")

    with c2:
        fig = px.bar(
            fpdf, x="Factory", y="Predicted LT (days)",
            color="Current",
            color_discrete_map={"✅ Current": "#7C3AED", "🔄 Alternate": "#10B981"},
            title=f"Predicted Lead Time by Factory — {sel_product}",
            text="Predicted LT (days)"
        )
        fig.update_traces(texttemplate="%{text:.0f}d", textposition="outside")
        fig.update_layout(
            paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
            font_color="#E2E8F0", showlegend=True,
            title_font_size=14, height=380
        )
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    # Table + map row
    st.markdown("<div class='section-header'>📊 Factory Comparison Table</div>", unsafe_allow_html=True)
    st.dataframe(fpdf.style.background_gradient(subset=["Predicted LT (days)"], cmap="RdYlGn_r"),
                 use_container_width=True, hide_index=True)

    # Factory map
    st.markdown("<div class='section-header'>🗺️ Factory Locations</div>", unsafe_allow_html=True)
    map_data = pd.DataFrame([
        {"Factory": k, "lat": v[0], "lon": v[1],
         "Type": "Current" if k == current_factory else "Alternate"}
        for k, v in FACTORIES.items()
    ])
    region_pt = pd.DataFrame([{
        "Factory": f"📍 {sel_region_t1} (Destination)",
        "lat": REGION_COORDS[sel_region_t1][0],
        "lon": REGION_COORDS[sel_region_t1][1],
        "Type": "Destination"
    }])
    map_all = pd.concat([map_data, region_pt], ignore_index=True)

    fig_map = px.scatter_mapbox(
        map_all, lat="lat", lon="lon", color="Type", hover_name="Factory",
        color_discrete_map={"Current":"#7C3AED","Alternate":"#10B981","Destination":"#F59E0B"},
        zoom=3, height=400,
        mapbox_style="carto-darkmatter"
    )
    fig_map.update_layout(paper_bgcolor="#0F172A", font_color="#E2E8F0", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # Product-Region heatmap
    st.markdown("<div class='section-header'>🌡️ Product × Region Lead Time Heatmap</div>", unsafe_allow_html=True)
    pivot = df.pivot_table(values="Lead Time Days", index="Product Name",
                           columns="Region", aggfunc="mean").round(1)
    fig_heat = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                         title="Avg Lead Time Days by Product & Region",
                         aspect="auto", height=500)
    fig_heat.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                           font_color="#E2E8F0", title_font_size=14)
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — WHAT-IF SCENARIO
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>🔄 What-If Scenario Analysis</div>", unsafe_allow_html=True)

    wa, wb, wc, wd = st.columns(4)
    with wa: wi_product  = st.selectbox("Product",   sorted(df["Product Name"].unique()), key="wi_prod")
    with wb: wi_region   = st.selectbox("Region",    df["Region"].unique().tolist(),       key="wi_reg")
    with wc: wi_ship     = st.selectbox("Ship Mode", df["Ship Mode"].unique().tolist(),    key="wi_ship")
    with wd: wi_alt_fac  = st.selectbox("Alt Factory", [f for f in FACTORIES if f != PRODUCT_FACTORY[wi_product]], key="wi_fac")

    cur_fac   = PRODUCT_FACTORY[wi_product]
    cur_lt    = predict_lt(wi_product, cur_fac,   wi_region, wi_ship)
    new_lt    = predict_lt(wi_product, wi_alt_fac, wi_region, wi_ship)
    cur_dist  = get_distance(cur_fac,   wi_region)
    new_dist  = get_distance(wi_alt_fac, wi_region)
    lt_delta  = cur_lt - new_lt
    dist_delta= cur_dist - new_dist
    lt_pct    = round(lt_delta / cur_lt * 100, 1) if cur_lt > 0 else 0

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{cur_lt:.0f}d</div>
            <div class='metric-label'>Current Lead Time</div>
            <div class='metric-detail' style='color:#94A3B8;font-size:0.75rem;'>{cur_fac}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        color = "#10B981" if new_lt < cur_lt else "#EF4444"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:{color};'>{new_lt:.0f}d</div>
            <div class='metric-label'>New Lead Time</div>
            <div class='metric-detail' style='color:#94A3B8;font-size:0.75rem;'>{wi_alt_fac}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        arrow = "▼" if lt_delta > 0 else "▲"
        color = "#10B981" if lt_delta > 0 else "#EF4444"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:{color};'>{arrow} {abs(lt_delta):.0f}d</div>
            <div class='metric-label'>Lead Time Change</div>
            <div class='metric-delta'>{lt_pct:+.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        color = "#10B981" if dist_delta > 0 else "#EF4444"
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:{color};'>{dist_delta:+.0f} km</div>
            <div class='metric-label'>Distance Change</div>
            <div class='metric-detail' style='color:#94A3B8;font-size:0.75rem;'>{cur_dist:.0f} → {new_dist:.0f} km</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge chart
    ga, gb = st.columns(2)
    with ga:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=new_lt,
            delta={"reference": cur_lt, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
            title={"text": "Lead Time (days)", "font": {"color": "#E2E8F0"}},
            gauge={
                "axis": {"range": [0, max(cur_lt, new_lt) * 1.2], "tickcolor": "#E2E8F0"},
                "bar":  {"color": "#10B981" if new_lt < cur_lt else "#EF4444"},
                "bgcolor": "#1E293B",
                "steps": [
                    {"range": [0, cur_lt * 0.5],  "color": "#10B98133"},
                    {"range": [cur_lt * 0.5, cur_lt], "color": "#F59E0B33"},
                ],
                "threshold": {"line": {"color": "#7C3AED", "width": 4},
                              "thickness": 0.75, "value": cur_lt}
            },
            number={"font": {"color": "#E2E8F0"}}
        ))
        fig_gauge.update_layout(paper_bgcolor="#0F172A", height=320,
                                font={"color": "#E2E8F0"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with gb:
        # All factories comparison for this scenario
        all_preds = []
        for fac in FACTORIES:
            lt = predict_lt(wi_product, fac, wi_region, wi_ship)
            all_preds.append({"Factory": fac, "LT": lt,
                              "Label": "Current" if fac == cur_fac else
                                       ("Selected Alt" if fac == wi_alt_fac else "Other")})
        apdf = pd.DataFrame(all_preds).sort_values("LT")

        fig_bar = px.bar(apdf, x="LT", y="Factory", orientation="h",
                         color="Label",
                         color_discrete_map={"Current":"#7C3AED","Selected Alt":"#10B981","Other":"#475569"},
                         title="All Factories Compared", text="LT")
        fig_bar.update_traces(texttemplate="%{text:.0f}d", textposition="outside")
        fig_bar.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                              font_color="#E2E8F0", height=320,
                              title_font_size=13, showlegend=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Scenario across all ship modes
    st.markdown("<div class='section-header'>📈 Lead Time Across All Ship Modes</div>", unsafe_allow_html=True)
    modes_data = []
    for sm in df["Ship Mode"].unique():
        modes_data.append({
            "Ship Mode": sm,
            "Current Factory": predict_lt(wi_product, cur_fac,    wi_region, sm),
            "Alt Factory":     predict_lt(wi_product, wi_alt_fac, wi_region, sm),
        })
    mdf = pd.DataFrame(modes_data)
    fig_modes = px.line(mdf, x="Ship Mode", y=["Current Factory", "Alt Factory"],
                        markers=True, title=f"{wi_product} — {wi_region}",
                        color_discrete_map={"Current Factory":"#7C3AED","Alt Factory":"#10B981"})
    fig_modes.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                            font_color="#E2E8F0", height=320)
    st.plotly_chart(fig_modes, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — RECOMMENDATIONS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🏆 Ranked Factory Reassignment Recommendations</div>", unsafe_allow_html=True)

    # Re-score with sidebar priority slider
    scored = sim[sim["LT_Reduction"] > 0].copy()
    def ns(s):
        mn, mx = s.min(), s.max()
        return (s - mn)/(mx - mn) if mx != mn else pd.Series(0.5, index=s.index)

    scored["Score_LT"]     = ns(scored["LT_Reduction_Pct"])
    scored["Score_Dist"]   = ns(scored["Dist_Reduction_km"])
    scored["Score_Profit"] = ns(scored["New_Est_Profit"])
    sw = speed_weight
    scored["Dynamic_Score"] = (
        sw * 0.7 * scored["Score_LT"] +
        sw * 0.3 * scored["Score_Dist"] +
        (1 - sw) * scored["Score_Profit"]
    ).round(4)

    top_dynamic = (
        scored.sort_values("Dynamic_Score", ascending=False)
        .groupby(["Product", "Region"]).first().reset_index()
        .sort_values("Dynamic_Score", ascending=False)
        .head(15)
    )
    top_dynamic["Rank"] = range(1, len(top_dynamic) + 1)

    # Cards for top 5
    st.markdown(f"**Priority: {'⚡ Speed' if speed_weight > 0.6 else ('💰 Profit' if speed_weight < 0.4 else '⚖️ Balanced')}** (slider = {speed_weight})")
    st.markdown("<br>", unsafe_allow_html=True)

    for _, row in top_dynamic.head(5).iterrows():
        badge = f"<span class='badge-green'>↓ {row['LT_Reduction_Pct']:.1f}% LT</span>"
        risk_badge = f"<span class='badge-red'>⚠ Profit Risk</span>" if row.get('Profit_Risk') else \
                     f"<span class='badge-green'>✓ Profit Stable</span>"
        st.markdown(f"""<div class='rec-card'>
            <span class='rec-rank'>#{int(row['Rank'])}</span>&nbsp;&nbsp;
            <span class='rec-product'>{row['Product']}</span>&nbsp;&nbsp;
            {badge}&nbsp;{risk_badge}
            <div class='rec-detail' style='margin-top:8px;'>
                <b style='color:#E2E8F0;'>{row['Current_Factory']}</b>
                <span style='color:#7C3AED;'> → </span>
                <b style='color:#10B981;'>{row['Alt_Factory']}</b>
                &nbsp;|&nbsp; Region: {row['Region']}
                &nbsp;|&nbsp; Score: {row['Dynamic_Score']:.3f}
                &nbsp;|&nbsp; Confidence: {row['Confidence_Score']:.2f}
            </div>
        </div>""", unsafe_allow_html=True)

    # Full table
    st.markdown("<div class='section-header'>📋 Full Recommendation Table</div>", unsafe_allow_html=True)
    display_cols = ["Rank","Product","Region","Current_Factory","Alt_Factory",
                    "LT_Reduction_Pct","Dist_Reduction_km","New_Est_Profit","Dynamic_Score","Confidence_Score"]
    st.dataframe(top_dynamic[display_cols].style.background_gradient(
        subset=["LT_Reduction_Pct","Dynamic_Score"], cmap="Greens"),
        use_container_width=True, hide_index=True)

    # Bubble chart
    st.markdown("<div class='section-header'>🫧 Recommendation Bubble Chart</div>", unsafe_allow_html=True)
    fig_bub = px.scatter(
        top_dynamic,
        x="LT_Reduction_Pct", y="New_Est_Profit",
        size="Confidence_Score", color="Alt_Factory",
        hover_name="Product",
        hover_data={"Region": True, "Dynamic_Score": True},
        title="Lead Time Reduction vs Profit (bubble = confidence)",
        size_max=50, height=450
    )
    fig_bub.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                          font_color="#E2E8F0", title_font_size=13)
    st.plotly_chart(fig_bub, use_container_width=True)

    # Factory-wise LT reduction
    st.markdown("<div class='section-header'>🏭 Avg LT Reduction by Target Factory</div>", unsafe_allow_html=True)
    fac_grp = scored.groupby("Alt_Factory")["LT_Reduction_Pct"].mean().reset_index()
    fig_fac = px.bar(fac_grp.sort_values("LT_Reduction_Pct", ascending=True),
                     x="LT_Reduction_Pct", y="Alt_Factory", orientation="h",
                     color="LT_Reduction_Pct", color_continuous_scale="Viridis",
                     title="Which factory performs best as reassignment target?", text="LT_Reduction_Pct")
    fig_fac.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_fac.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                          font_color="#E2E8F0", height=350, title_font_size=13)
    st.plotly_chart(fig_fac, use_container_width=True)

    # Feature importance
    st.markdown("<div class='section-header'>🔍 Feature Importance (Random Forest)</div>", unsafe_allow_html=True)
    fig_fi = px.bar(feat_imp.sort_values("Importance"),
                    x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Purples",
                    title="What drives Lead Time predictions?", text="Importance")
    fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fi.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                          font_color="#E2E8F0", height=450, title_font_size=13)
    st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — RISK & IMPACT
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>⚠️ Risk Assessment & Impact Panel</div>", unsafe_allow_html=True)

    # Risk distribution
    risk_cnt = sim["Risk_Level"].value_counts().reset_index()
    risk_cnt.columns = ["Risk Level", "Count"]
    color_map = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}

    ra, rb = st.columns(2)
    with ra:
        fig_risk = px.pie(risk_cnt, names="Risk Level", values="Count",
                          color="Risk Level", color_discrete_map=color_map,
                          title="Risk Level Distribution", hole=0.5)
        fig_risk.update_layout(paper_bgcolor="#0F172A", font_color="#E2E8F0",
                               title_font_size=14, height=350)
        st.plotly_chart(fig_risk, use_container_width=True)

    with rb:
        # Model comparison
        fig_model = px.bar(reg_comp, x="Model", y="R2",
                           color="R2", color_continuous_scale="Purples",
                           title="Model Performance (R²)", text="R2")
        fig_model.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_model.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                                font_color="#E2E8F0", height=350, title_font_size=14)
        st.plotly_chart(fig_model, use_container_width=True)

    # High risk alerts
    st.markdown("<div class='section-header'>🚨 High-Risk Reassignment Alerts</div>", unsafe_allow_html=True)
    high_risk = sim[sim["Risk_Level"] == "High"][[
        "Product","Current_Factory","Alt_Factory","Region",
        "LT_Reduction_Pct","Baseline_Profit","New_Est_Profit","Confidence_Score"
    ]].sort_values("LT_Reduction_Pct", ascending=False).head(10)

    for _, row in high_risk.iterrows():
        profit_drop = row["New_Est_Profit"] - row["Baseline_Profit"]
        st.markdown(f"""<div class='rec-card' style='border-color:#EF444466;'>
            <span style='color:#EF4444; font-weight:700;'>⚠ HIGH RISK</span>&nbsp;&nbsp;
            <span class='rec-product'>{row['Product']}</span>
            <div class='rec-detail' style='margin-top:6px;'>
                {row['Current_Factory']} → <span style='color:#EF4444;'>{row['Alt_Factory']}</span>
                &nbsp;|&nbsp; Region: {row['Region']}
                &nbsp;|&nbsp; LT Cut: {row['LT_Reduction_Pct']:.1f}%
                &nbsp;|&nbsp; Profit Δ: <span style='color:#EF4444;'>{profit_drop:+.2f}%</span>
                &nbsp;|&nbsp; Confidence: {row['Confidence_Score']:.3f}
            </div>
        </div>""", unsafe_allow_html=True)

    # Profit impact scatter
    st.markdown("<div class='section-header'>💰 Profit Impact Analysis</div>", unsafe_allow_html=True)
    fig_prof = px.scatter(
        sim, x="LT_Reduction_Pct", y="New_Est_Profit",
        color="Risk_Level", symbol="Risk_Level",
        color_discrete_map=color_map,
        hover_name="Product",
        hover_data={"Alt_Factory": True, "Region": True},
        title="Lead Time Reduction vs Estimated Profit — All Scenarios",
        height=450
    )
    fig_prof.add_hline(y=sim["Baseline_Profit"].mean(),
                       line_dash="dash", line_color="#F59E0B",
                       annotation_text="Avg Baseline Profit")
    fig_prof.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                           font_color="#E2E8F0", title_font_size=13)
    st.plotly_chart(fig_prof, use_container_width=True)

    # Route performance
    st.markdown("<div class='section-header'>🛣️ Route Performance — Factory × Region</div>", unsafe_allow_html=True)
    fig_route = px.density_heatmap(
        route_perf, x="Region", y="Factory",
        z="Avg_LeadTime", color_continuous_scale="RdYlGn_r",
        title="Avg Lead Time by Factory & Region",
        text_auto=True, height=400
    )
    fig_route.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                            font_color="#E2E8F0", title_font_size=13)
    st.plotly_chart(fig_route, use_container_width=True)

    # Product cluster summary
    st.markdown("<div class='section-header'>📦 Product Performance Clusters</div>", unsafe_allow_html=True)
    fig_pc = px.scatter(
        prod_clust.reset_index(), x="Avg_LeadTime", y="Avg_ProfitPct",
        size="Total_Orders", color="Product_Label",
        hover_name="Product Name",
        color_discrete_map={"High Performer":"#10B981","Low Performer":"#EF4444"},
        title="Product Clusters — Lead Time vs Profit Margin",
        size_max=60, height=420
    )
    fig_pc.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                         font_color="#E2E8F0", title_font_size=13)
    st.plotly_chart(fig_pc, use_container_width=True)

    # KPI table
    st.markdown("<div class='section-header'>📊 Full KPI Summary</div>", unsafe_allow_html=True)
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)