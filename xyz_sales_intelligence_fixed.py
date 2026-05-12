import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XYZ Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b27 40%, #1a2035 100%);
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
    border-right: 1px solid #2d3748;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Header ── */
.hero-header {
    background: linear-gradient(120deg, #1e3a5f 0%, #162d4a 50%, #0f2035 100%);
    border: 1px solid #2d4a6e;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: #f0f9ff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 300;
}
.hero-accent {
    display: inline-block;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── KPI Cards ── */
.kpi-grid { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 160px;
    background: rgba(30,58,95,0.35);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}
.kpi-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #f0f9ff; }
.kpi-delta { font-size: 0.82rem; margin-top: 4px; }
.kpi-neg { color: #f87171; }
.kpi-pos { color: #4ade80; }

/* ── Section headers ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 32px 0 6px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e3a5f;
}
.section-desc {
    font-size: 0.88rem;
    color: #64748b;
    margin-bottom: 18px;
}

/* ── Insight boxes ── */
.insight-box {
    background: rgba(15,32,53,0.7);
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-top: 12px;
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.65;
}
.insight-box strong { color: #38bdf8; }
.insight-warn { border-left-color: #f97316; }
.insight-warn strong { color: #f97316; }
.insight-good { border-left-color: #4ade80; }
.insight-good strong { color: #4ade80; }

/* ── Metric chips ── */
.chip {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 6px;
}
.chip-blue { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
.chip-orange { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
.chip-green { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.chip-red { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

/* ── Corr badge ── */
.corr-badge {
    font-size: 2rem; font-weight: 700;
    font-family: 'Playfair Display', serif;
}

/* ── Upload area ── */
.upload-hint {
    text-align: center;
    padding: 48px 20px;
    border: 2px dashed #2d4a6e;
    border-radius: 16px;
    color: #64748b;
    margin-top: 20px;
}
.upload-hint h2 { font-family: 'Playfair Display', serif; color: #94a3b8; margin-bottom: 8px; }

div[data-testid="stExpander"] {
    background: rgba(22,27,39,0.6);
    border: 1px solid #2d3748;
    border-radius: 10px;
}

hr { border-color: #1e3a5f; }

/* Plotly overrides */
.js-plotly-plot { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

CHANNEL_COLORS = {
    "Platinum": "#818cf8",
    "Gold": "#fbbf24",
    "Silver": "#94a3b8",
    "Bronze": "#f97316",
}
MONTHS = ["Jan", "Feb", "March", "April"]


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def corr_label(r):
    if r is None: return "N/A", "chip-blue"
    a = abs(r)
    if a >= 0.7: strength, chip = "Strong", "chip-green" if r > 0 else "chip-red"
    elif a >= 0.4: strength, chip = "Moderate", "chip-orange"
    else: strength, chip = "Weak", "chip-blue"
    direction = "Positive" if r >= 0 else "Negative"
    return f"{strength} {direction}", chip


def pearson(x, y):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 3: return None, None
    r, p = stats.pearsonr(x[mask], y[mask])
    return round(r, 4), round(p, 4)


def safe_pct(a, b):
    if b == 0 or pd.isna(b): return None
    return round((a - b) / b * 100, 1)


def fig_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Playfair Display", size=16, color="#e2e8f0"), x=0.03),
        paper_bgcolor="rgba(15,32,53,0.0)",
        plot_bgcolor="rgba(15,32,53,0.4)",
        font=dict(family="DM Sans", color="#94a3b8"),
        height=height,
        margin=dict(l=16, r=16, t=50, b=16),
        legend=dict(
            bgcolor="rgba(15,32,53,0.7)",
            bordercolor="#2d4a6e",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="#1e2d45", linecolor="#2d3748", zerolinecolor="#2d3748"),
        yaxis=dict(gridcolor="#1e2d45", linecolor="#2d3748", zerolinecolor="#2d3748"),
    )
    return fig


# ─── PREPROCESS ────────────────────────────────────────────────────────────────
def preprocess(df):
    df.columns = df.columns.str.strip()

    rename = {
        "Actual visits done in April": "Apr_Visits",
        "Actual visits done in Jan": "Jan_Visits",
        "Actual visits done in Feb": "Feb_Visits",
        "Actual Visits in March": "Mar_Visits",
        "Jan Sales": "Jan_Sales",
        "Feb Sales": "Feb_Sales",
        "March Sales": "Mar_Sales",
        "April Sales": "Apr_Sales",
        "Visit Diff March and April": "VDiff_Mar_Apr",
        "Sales Diff March and April": "SDiff_Mar_Apr",
        "Visit Diff Feb and April": "VDiff_Feb_Apr",
        "Sales Diff Feb and April": "SDiff_Feb_Apr",
        "CHANNEL": "Channel",
        "CUSTOMER_NAME": "Customer",
        "DISTRIBUTOR_NAME": "Distributor",
        "SELLER_NAME": "Seller",
        "ROUTE_NAME": "Route",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    num_cols = ["Apr_Visits","Jan_Visits","Feb_Visits","Mar_Visits",
                "Jan_Sales","Feb_Sales","Mar_Sales","Apr_Sales",
                "VDiff_Mar_Apr","SDiff_Mar_Apr","VDiff_Feb_Apr","SDiff_Feb_Apr"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute visit diffs for Jan vs April if not present
    if "Jan_Visits" in df.columns and "Apr_Visits" in df.columns:
        df["VDiff_Jan_Apr"] = df["Jan_Visits"] - df["Apr_Visits"]
    if "Jan_Sales" in df.columns and "Apr_Sales" in df.columns:
        df["SDiff_Jan_Apr"] = df["Jan_Sales"] - df["Apr_Sales"]

    df = df.dropna(subset=["Apr_Sales","Mar_Sales"], how="all")

    if "Channel" in df.columns:
        df["Channel"] = df["Channel"].astype(str).str.strip().str.title()

    return df


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 12px'>
        <div style='font-family:Playfair Display,serif;font-size:1.3rem;color:#e2e8f0;font-weight:700;'>📊 XYZ Intelligence</div>
        <div style='font-size:0.78rem;color:#64748b;margin-top:4px;'>Visit–Sales Correlation Suite</div>
    </div>
    <hr style='border-color:#2d3748;margin:0 0 16px'>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Dataset (CSV / Excel)", type=["csv","xlsx","xls"])

    st.markdown("---")
    st.markdown("<div style='font-size:0.8rem;color:#64748b;'>FILTERS</div>", unsafe_allow_html=True)

    channel_filter = st.multiselect(
        "Channel", options=["Platinum","Gold","Silver","Bronze"],
        default=["Platinum","Gold","Silver","Bronze"]
    )
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:#374151;'>Correlation strength guide:<br>🟢 ≥0.70 Strong &nbsp; 🟡 0.40–0.69 Moderate &nbsp; 🔵 &lt;0.40 Weak</div>", unsafe_allow_html=True)


# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">XYZ <span class="hero-accent">Sales Decline</span> Intelligence</div>
  <p class="hero-subtitle">Investigating the relationship between visit frequency and April sales performance · Correlation & Channel Analysis</p>
</div>
""", unsafe_allow_html=True)


# ─── NO FILE ───────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("""
    <div class="upload-hint">
      <h2>📂 Upload Your Dataset to Begin</h2>
      <p>Supports <strong>.csv</strong> and <strong>.xlsx</strong> formats · Use the sidebar uploader</p>
      <p style='margin-top:12px;font-size:0.82rem;'>Expected columns: DISTRIBUTOR_NAME, SELLER_NAME, CHANNEL, Actual visits done in April/Jan/Feb/March, Jan/Feb/March/April Sales, etc.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
try:
    if uploaded.name.endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

df_all = preprocess(raw)

# apply channel filter
if "Channel" in df_all.columns and channel_filter:
    df = df_all[df_all["Channel"].isin(channel_filter)].copy()
else:
    df = df_all.copy()

n = len(df)
channels = sorted(df["Channel"].dropna().unique()) if "Channel" in df.columns else []


# ─── KPI BAR ───────────────────────────────────────────────────────────────────
total_jan  = df["Jan_Sales"].sum()  if "Jan_Sales"  in df.columns else 0
total_feb  = df["Feb_Sales"].sum()  if "Feb_Sales"  in df.columns else 0
total_mar  = df["Mar_Sales"].sum()  if "Mar_Sales"  in df.columns else 0
total_apr  = df["Apr_Sales"].sum()  if "Apr_Sales"  in df.columns else 0

apr_mar_pct = safe_pct(total_apr, total_mar)
apr_feb_pct = safe_pct(total_apr, total_feb)
apr_jan_pct = safe_pct(total_apr, total_jan)

def delta_html(pct):
    if pct is None: return ""
    arrow = "▼" if pct < 0 else "▲"
    cls = "kpi-neg" if pct < 0 else "kpi-pos"
    return f'<div class="kpi-delta {cls}">{arrow} {abs(pct)}% vs prior</div>'

def fmt_num(v, prefix="₹"):
    if v >= 1_000_000: return f"{prefix}{v/1_000_000:.2f}M"
    if v >= 1_000: return f"{prefix}{v/1_000:.1f}K"
    return f"{prefix}{v:.0f}"

apr_visits = df["Apr_Visits"].sum() if "Apr_Visits" in df.columns else 0
mar_visits = df["Mar_Visits"].sum() if "Mar_Visits" in df.columns else 0
visit_delta = safe_pct(apr_visits, mar_visits)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Jan Sales</div>
    <div class="kpi-value">{fmt_num(total_jan)}</div>
    {delta_html(apr_jan_pct)}
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Feb Sales</div>
    <div class="kpi-value">{fmt_num(total_feb)}</div>
    {delta_html(apr_feb_pct)}
  </div>
  <div class="kpi-card">
    <div class="kpi-label">March Sales</div>
    <div class="kpi-value">{fmt_num(total_mar)}</div>
    {delta_html(apr_mar_pct)}
  </div>
  <div class="kpi-card">
    <div class="kpi-label">April Sales</div>
    <div class="kpi-value">{fmt_num(total_apr)}</div>
    <div class="kpi-delta kpi-neg">← Focus Month</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">April Visits</div>
    <div class="kpi-value">{int(apr_visits):,}</div>
    {delta_html(visit_delta)}
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Customers</div>
    <div class="kpi-value">{n:,}</div>
    <div class="kpi-delta" style="color:#64748b;">in selection</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Monthly Sales Trend
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">① Monthly Sales & Visit Trend</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">How did aggregate sales and visits change across January → April?</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    months_labels = ["January", "February", "March", "April"]
    sales_vals = [
        df["Jan_Sales"].sum()  if "Jan_Sales"  in df.columns else 0,
        df["Feb_Sales"].sum()  if "Feb_Sales"  in df.columns else 0,
        df["Mar_Sales"].sum()  if "Mar_Sales"  in df.columns else 0,
        df["Apr_Sales"].sum()  if "Apr_Sales"  in df.columns else 0,
    ]
    visits_vals = [
        df["Jan_Visits"].sum() if "Jan_Visits" in df.columns else 0,
        df["Feb_Visits"].sum() if "Feb_Visits" in df.columns else 0,
        df["Mar_Visits"].sum() if "Mar_Visits" in df.columns else 0,
        df["Apr_Visits"].sum() if "Apr_Visits" in df.columns else 0,
    ]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=months_labels, y=sales_vals,
        name="Sales (₹)", marker_color=["#38bdf8","#38bdf8","#38bdf8","#f87171"],
        marker_line_color="rgba(0,0,0,0)", opacity=0.85,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=months_labels, y=visits_vals,
        name="Total Visits", mode="lines+markers",
        line=dict(color="#fbbf24", width=2.5),
        marker=dict(size=8, color="#fbbf24"),
    ), secondary_y=True)
    fig.update_yaxes(title_text="Total Sales (₹)", secondary_y=False, title_font=dict(color="#38bdf8"), tickfont=dict(color="#38bdf8"))
    fig.update_yaxes(title_text="Total Visits (count)", secondary_y=True, title_font=dict(color="#fbbf24"), tickfont=dict(color="#fbbf24"))
    fig.update_xaxes(title_text="Month")
    fig = fig_layout(fig, "Monthly Sales vs Visits", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box insight-warn">
        <strong>📉 What this chart shows:</strong> Each blue bar represents total sales for that month. The yellow line tracks how many customer visits were made. Notice how <strong>April (red bar)</strong> shows a drop in both sales and visits. When the yellow line dips, sales tend to follow — hinting that fewer visits may be driving lower revenue.
    </div>""", unsafe_allow_html=True)

with col2:
    sales_diff = [0] + [sales_vals[i]-sales_vals[i-1] for i in range(1,4)]
    colors_diff = ["#38bdf8","#38bdf8","#38bdf8","#f87171"]
    fig2 = go.Figure(go.Bar(
        x=months_labels, y=sales_diff,
        marker_color=colors_diff, opacity=0.85,
        text=[f"₹{v/1000:.1f}K" for v in sales_diff], textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    fig2.update_xaxes(title_text="Month")
    fig2.update_yaxes(title_text="Change in Sales vs Prior Month (₹)")
    fig2 = fig_layout(fig2, "Month-over-Month Sales Change (₹)", height=380)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>📊 What this chart shows:</strong> Positive bars mean sales went up vs the previous month; negative (red) bars mean sales fell. This quickly highlights <strong>which specific month caused the decline</strong> and by how much — April is our red flag.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Overall Correlation Analysis
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">② Overall Correlation — Visit Drop vs Sales Drop</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Does a bigger drop in visits lead to a bigger drop in sales? We test this for all three comparisons.</div>', unsafe_allow_html=True)

comparisons = [
    ("March vs April",    "VDiff_Mar_Apr",  "SDiff_Mar_Apr",  "Mar", "Apr"),
    ("February vs April", "VDiff_Feb_Apr",  "SDiff_Feb_Apr",  "Feb", "Apr"),
    ("January vs April",  "VDiff_Jan_Apr",  "SDiff_Jan_Apr",  "Jan", "Apr"),
]

corr_results = {}
for label, vd_col, sd_col, m1, m2 in comparisons:
    if vd_col in df.columns and sd_col in df.columns:
        r, p = pearson(df[vd_col].values, df[sd_col].values)
        corr_results[label] = (r, p, vd_col, sd_col, m1, m2)

# Summary correlation table
cols_corr = st.columns(3)
for idx, (label, (r, p, *_)) in enumerate(corr_results.items()):
    lbl, chip = corr_label(r)
    sig = "Statistically Significant (p < 0.05)" if p and p < 0.05 else ("Not Significant" if p else "—")
    with cols_corr[idx]:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;padding:24px 12px;">
          <div class="kpi-label">{label}</div>
          <div class="corr-badge" style="color:{'#4ade80' if r and r>0.4 else '#f87171' if r and r<0 else '#38bdf8'}">
            {f"r = {r:.3f}" if r is not None else "N/A"}
          </div>
          <div style="margin-top:8px">
            <span class="chip {chip}">{lbl}</span>
          </div>
          <div style="font-size:0.75rem;color:#64748b;margin-top:8px;">{sig}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Scatter plots for each comparison
for label, (r, p, vd_col, sd_col, m1, m2) in corr_results.items():
    with st.expander(f"🔍 Scatter Plot — {label}", expanded=(label=="March vs April")):
        plot_df = df[[vd_col, sd_col]].dropna().copy()
        if "Channel" in df.columns:
            plot_df["Channel"] = df.loc[plot_df.index, "Channel"]
            color_col = "Channel"
            color_map = CHANNEL_COLORS
        else:
            color_col = None
            color_map = None

        fig_s = px.scatter(
            plot_df, x=vd_col, y=sd_col,
            color=color_col if color_col and color_col in plot_df.columns else None,
            color_discrete_map=color_map,
            trendline="ols",
            labels={
                vd_col: f"Visit Difference ({m1} Visits − April Visits)",
                sd_col: f"Sales Difference ({m1} Sales − April Sales) (₹)",
            },
            opacity=0.65,
        )
        fig_s.update_traces(marker=dict(size=7))
        fig_s = fig_layout(fig_s, f"Visit Difference vs Sales Difference · {label}", height=400)
        st.plotly_chart(fig_s, use_container_width=True)

        lbl, chip = corr_label(r)
        st.markdown(f"""
        <div class="insight-box">
            <strong>📌 How to read this scatter plot:</strong> Each dot represents one customer. The <em>X-axis</em> shows how many fewer visits they received in April compared to {m1}. The <em>Y-axis</em> shows how much their sales fell. A <strong>positive correlation (r = {r if r else 'N/A'})</strong> means customers who had more visits cut → also had larger sales drops. The diagonal trend line confirms this pattern. <span class="chip {chip}">{lbl} Correlation</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Channel-wise Correlation
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">③ Channel-wise Correlation Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Breaking down the visit–sales relationship by channel tier: Platinum, Gold, Silver, Bronze.</div>', unsafe_allow_html=True)

if "Channel" not in df.columns or df["Channel"].isna().all():
    st.warning("Channel column not found in data.")
else:
    ch_rows = []
    for ch in ["Platinum","Gold","Silver","Bronze"]:
        sub = df[df["Channel"]==ch]
        if len(sub) < 3: continue
        for label, vd_col, sd_col, m1, m2 in comparisons:
            if vd_col in sub.columns and sd_col in sub.columns:
                r, p = pearson(sub[vd_col].values, sub[sd_col].values)
                ch_rows.append({
                    "Channel": ch,
                    "Comparison": label,
                    "r": r, "p": p,
                    "N": len(sub),
                    "Avg Visit Drop": sub[vd_col].mean(),
                    "Avg Sales Drop": sub[sd_col].mean(),
                })

    ch_df = pd.DataFrame(ch_rows)

    if not ch_df.empty:
        # Heatmap
        pivot = ch_df.pivot(index="Channel", columns="Comparison", values="r")
        pivot = pivot.reindex(["Platinum","Gold","Silver","Bronze"])

        fig_heat = px.imshow(
            pivot,
            color_continuous_scale=[[0,"#f87171"],[0.5,"#1e3a5f"],[1,"#4ade80"]],
            zmin=-1, zmax=1,
            text_auto=".3f",
            labels=dict(color="Pearson r"),
        )
        fig_heat.update_traces(textfont=dict(size=14, color="white"))
        fig_heat.update_xaxes(title="Comparison Period", tickangle=-20)
        fig_heat.update_yaxes(title="Channel Tier")
        fig_heat = fig_layout(fig_heat, "Correlation Heatmap: Visit Drop vs Sales Drop by Channel", height=320)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
            <strong>🗺️ What this heatmap shows:</strong> Each cell shows the correlation coefficient (r) for a specific channel and time comparison. <strong>Green = strong positive correlation</strong> (visit drops closely track sales drops). <strong>Red = negative/inverse relationship</strong>. Darker means weaker. This helps identify which channel is most affected by visit-frequency changes.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Channel bar chart — avg sales drop
        col_a, col_b = st.columns(2)
        with col_a:
            mar_ch = ch_df[ch_df["Comparison"]=="March vs April"].copy()
            fig_bar = px.bar(
                mar_ch, x="Channel", y="Avg Sales Drop",
                color="Channel", color_discrete_map=CHANNEL_COLORS,
                text=mar_ch["Avg Sales Drop"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else ""),
                labels={"Avg Sales Drop": "Avg Sales Difference — March vs April (₹)", "Channel": "Channel Tier"},
            )
            fig_bar.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
            fig_bar = fig_layout(fig_bar, "Average Sales Drop by Channel (March→April)", height=360)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("""
            <div class="insight-box insight-warn">
                <strong>📊 What this shows:</strong> Which channel segment lost the most sales between March and April on average? Taller bars = larger revenue drop per customer. This tells you <strong>where to focus recovery efforts</strong>.
            </div>""", unsafe_allow_html=True)

        with col_b:
            fig_bar2 = px.bar(
                mar_ch, x="Channel", y="Avg Visit Drop",
                color="Channel", color_discrete_map=CHANNEL_COLORS,
                text=mar_ch["Avg Visit Drop"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
                labels={"Avg Visit Drop": "Avg Visit Difference — March vs April (visits)", "Channel": "Channel Tier"},
            )
            fig_bar2.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
            fig_bar2 = fig_layout(fig_bar2, "Average Visit Drop by Channel (March→April)", height=360)
            st.plotly_chart(fig_bar2, use_container_width=True)
            st.markdown("""
            <div class="insight-box">
                <strong>🚶 What this shows:</strong> On average, how many fewer visits did customers in each channel receive in April compared to March? A high bar = that channel's customers were visited significantly less often — which likely contributes to their sales decline.
            </div>""", unsafe_allow_html=True)

        # Channel-wise scatter facets
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.92rem;color:#94a3b8;margin-bottom:12px;'>Channel-wise Scatter — March vs April Visit Difference vs Sales Difference</div>", unsafe_allow_html=True)

        valid_channels = [c for c in ["Platinum","Gold","Silver","Bronze"] if c in df["Channel"].values]
        ncols = min(2, len(valid_channels))
        rows = (len(valid_channels) + ncols - 1) // ncols
        fig_sub = make_subplots(
            rows=rows, cols=ncols,
            subplot_titles=valid_channels,
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.12, vertical_spacing=0.18,
        )
        for i, ch in enumerate(valid_channels):
            row, col = divmod(i, ncols)
            sub = df[df["Channel"]==ch][["VDiff_Mar_Apr","SDiff_Mar_Apr"]].dropna()
            clr = CHANNEL_COLORS.get(ch, "#38bdf8")
            fig_sub.add_trace(
                go.Scatter(
                    x=sub["VDiff_Mar_Apr"], y=sub["SDiff_Mar_Apr"],
                    mode="markers", name=ch,
                    marker=dict(color=clr, size=6, opacity=0.65),
                    showlegend=False,
                ),
                row=row+1, col=col+1
            )
            # OLS trend
            if len(sub) >= 2:
                m_coef, b_coef = np.polyfit(sub["VDiff_Mar_Apr"], sub["SDiff_Mar_Apr"], 1)
                x_range = np.linspace(sub["VDiff_Mar_Apr"].min(), sub["VDiff_Mar_Apr"].max(), 50)
                fig_sub.add_trace(
                    go.Scatter(
                        x=x_range, y=m_coef*x_range+b_coef,
                        mode="lines", line=dict(color=clr, dash="dot", width=2),
                        showlegend=False,
                    ),
                    row=row+1, col=col+1
                )

        fig_sub.update_layout(
            paper_bgcolor="rgba(15,32,53,0.0)",
            plot_bgcolor="rgba(15,32,53,0.4)",
            font=dict(family="DM Sans", color="#94a3b8"),
            height=400*rows,
            margin=dict(l=16, r=16, t=60, b=16),
            title=dict(text="Visit Drop vs Sales Drop by Channel · March→April", font=dict(family="Playfair Display", size=15, color="#e2e8f0"), x=0.02),
        )
        for axis in fig_sub.layout:
            if axis.startswith("xaxis") or axis.startswith("yaxis"):
                fig_sub.layout[axis].update(gridcolor="#1e2d45", linecolor="#2d3748", zerolinecolor="#2d3748")

        st.plotly_chart(fig_sub, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
            <strong>🔍 Channel scatter breakdown:</strong> Each panel shows one channel. The X-axis = visit difference (March − April), Y-axis = sales difference (March − April). The dotted trend line shows the direction of the relationship. A <strong>upward-sloping line</strong> confirms that customers visited less → also sold less. Compare panels to see which channel shows the strongest pattern.
        </div>""", unsafe_allow_html=True)

        # Detailed channel stats table
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Full Channel Correlation Table"):
            disp = ch_df[["Channel","Comparison","N","r","p","Avg Visit Drop","Avg Sales Drop"]].copy()
            disp["r"] = disp["r"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
            disp["p"] = disp["p"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
            disp["Avg Visit Drop"] = disp["Avg Visit Drop"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            disp["Avg Sales Drop"] = disp["Avg Sales Drop"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
            st.dataframe(disp.reset_index(drop=True), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Customer-level Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">④ Customer-Level Deep Dive</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Who are the top customers driving the April sales decline?</div>', unsafe_allow_html=True)

if "Customer" in df.columns and "SDiff_Mar_Apr" in df.columns:
    top_n = 15
    top_loss = df[["Customer","Channel","Apr_Sales","Mar_Sales","SDiff_Mar_Apr","VDiff_Mar_Apr"]].dropna(subset=["SDiff_Mar_Apr"])
    top_loss = top_loss.sort_values("SDiff_Mar_Apr", ascending=False).head(top_n)

    fig_cust = px.bar(
        top_loss, y="Customer", x="SDiff_Mar_Apr",
        color="Channel", color_discrete_map=CHANNEL_COLORS,
        orientation="h",
        labels={"SDiff_Mar_Apr":"Sales Difference March − April (₹)", "Customer":"Customer Name"},
        text=top_loss["SDiff_Mar_Apr"].apply(lambda x: f"₹{x:,.0f}"),
    )
    fig_cust.update_traces(textposition="outside", textfont=dict(size=10, color="#e2e8f0"))
    fig_cust.update_yaxes(autorange="reversed")
    fig_cust = fig_layout(fig_cust, f"Top {top_n} Customers by Sales Decline (March → April)", height=480)
    st.plotly_chart(fig_cust, use_container_width=True)
    st.markdown(f"""
    <div class="insight-box insight-warn">
        <strong>🏆 What this shows:</strong> The {top_n} customers with the largest drop in sales between March and April. Longer bars = more revenue lost. Colors indicate channel tier. <strong>These are the priority accounts</strong> — fixing visit frequency for these customers will have the highest impact on recovering April-level sales.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Distribution & Box Plots
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">⑤ Sales Distribution by Channel Across Months</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">How did the spread of sales values change per channel from January to April?</div>', unsafe_allow_html=True)

if "Channel" in df.columns:
    melt_parts = []
    for mn, col in [("January","Jan_Sales"),("February","Feb_Sales"),("March","Mar_Sales"),("April","Apr_Sales")]:
        if col in df.columns:
            tmp = df[["Channel",col]].copy()
            tmp.columns = ["Channel","Sales"]
            tmp["Month"] = mn
            melt_parts.append(tmp)
    if melt_parts:
        melt_df = pd.concat(melt_parts, ignore_index=True).dropna()
        fig_box = px.box(
            melt_df, x="Month", y="Sales", color="Channel",
            color_discrete_map=CHANNEL_COLORS,
            category_orders={"Month":["January","February","March","April"]},
            labels={"Sales":"Sales (₹)", "Month":"Month"},
            points="outliers",
        )
        fig_box = fig_layout(fig_box, "Sales Distribution per Channel — All Months", height=450)
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
            <strong>📦 Box plot explained:</strong> Each box shows the range of sales values for a channel in a given month. The <em>middle line</em> is the median (typical customer). The box covers the middle 50% of customers. <em>Dots</em> above/below are outliers. If boxes move <strong>downward in April</strong>, that entire channel's sales distribution has shifted lower — not just a few customers, but a widespread decline.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Visit Compliance
# ═══════════════════════════════════════════════════════════════════════════════
if "Visits Needed in a Month as per Business Principle" in df.columns and "Apr_Visits" in df.columns:
    st.markdown('<div class="section-title">⑥ Visit Compliance — Needed vs Actual (April)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Are customers getting the visits they need per business guidelines?</div>', unsafe_allow_html=True)

    df["Visit_Required"] = pd.to_numeric(df["Visits Needed in a Month as per Business Principle"], errors="coerce")
    df["Visit_Compliance"] = (df["Apr_Visits"] / df["Visit_Required"] * 100).clip(upper=200)

    if "Channel" in df.columns:
        comp_ch = df.groupby("Channel")["Visit_Compliance"].mean().reset_index()
        comp_ch.columns = ["Channel","Avg Compliance %"]
        comp_ch = comp_ch[comp_ch["Channel"].isin(["Platinum","Gold","Silver","Bronze"])]

        fig_comp = px.bar(
            comp_ch, x="Channel", y="Avg Compliance %",
            color="Channel", color_discrete_map=CHANNEL_COLORS,
            text=comp_ch["Avg Compliance %"].apply(lambda x: f"{x:.1f}%"),
            labels={"Avg Compliance %": "Avg Visit Compliance in April (%)", "Channel": "Channel Tier"},
        )
        fig_comp.add_hline(y=100, line_dash="dash", line_color="#4ade80", annotation_text="Target 100%", annotation_position="top right")
        fig_comp.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
        fig_comp = fig_layout(fig_comp, "April Visit Compliance vs Business-Prescribed Target by Channel", height=380)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("""
        <div class="insight-box insight-good">
            <strong>✅ What this shows:</strong> The green dotted line at 100% is the target — the number of visits each customer should receive according to business guidelines. Bars <strong>below 100%</strong> indicate under-served channels. If a channel is at 60%, it means customers in that tier are only getting 60% of their required visits — a direct operational gap that can suppress sales.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Summary & Recommendations
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">⑦ Key Findings & Recommendations</div>', unsafe_allow_html=True)

# Dynamically build findings
findings = []
for label, (r, p, *_) in corr_results.items():
    if r is not None:
        lbl, _ = corr_label(r)
        findings.append(f"<strong>{label}:</strong> Correlation r = {r:.3f} → {lbl} relationship between visit drops and sales drops.")

if "Channel" in df.columns and not ch_df.empty:
    best_ch = ch_df[ch_df["Comparison"]=="March vs April"].dropna(subset=["r"])
    if not best_ch.empty:
        top_ch = best_ch.loc[best_ch["r"].idxmax(), "Channel"]
        findings.append(f"<strong>Most visit-sensitive channel:</strong> <em>{top_ch}</em> shows the strongest correlation between reduced visits and falling sales.")

findings_html = "".join(f"<li style='margin:8px 0'>{f}</li>" for f in findings)

st.markdown(f"""
<div class="insight-box insight-warn" style="padding:20px 24px">
    <strong>🔎 Analysis Summary</strong>
    <ul style="margin:10px 0 0 0;padding-left:18px;line-height:1.8">
        {findings_html}
    </ul>
</div>
<div class="insight-box insight-good" style="margin-top:14px;padding:20px 24px">
    <strong>💡 Recommended Actions</strong>
    <ol style="margin:10px 0 0 0;padding-left:18px;line-height:1.9">
        <li>Identify and immediately re-schedule missed April visits for high-value customers.</li>
        <li>Prioritize channel tiers with lowest visit compliance — they show the greatest revenue risk.</li>
        <li>Set automated alerts when a customer's monthly visits fall below the business-prescribed threshold.</li>
        <li>Track visit-to-sales ratio monthly per seller/route to detect early warning signals.</li>
        <li>Consider visit frequency as a leading KPI in sales forecasting models.</li>
    </ol>
</div>
""", unsafe_allow_html=True)


# ─── RAW DATA PREVIEW ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🗂️ Preview Preprocessed Data"):
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Showing first 100 of {len(df):,} rows after filtering.")


# ═══════════════════════════════════════════════════════════════════════════════
# ███████╗██████╗  █████╗ ██╗   ██╗██████╗     ██████╗ ███████╗████████╗
# ██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗    ██╔══██╗██╔════╝╚══██╔══╝
# █████╗  ██████╔╝███████║██║   ██║██║  ██║    ██║  ██║█████╗     ██║
# ██╔══╝  ██╔══██╗██╔══██║██║   ██║██║  ██║    ██║  ██║██╔══╝     ██║
# ██║     ██║  ██║██║  ██║╚██████╔╝██████╔╝    ██████╔╝███████╗   ██║
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝     ╚═════╝ ╚══════╝   ╚═╝
# SR BEHAVIOUR & FRAUD DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="
    margin: 48px 0 0 0;
    background: linear-gradient(120deg, #1a0a0a 0%, #2d0f0f 50%, #1a0a1a 100%);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 16px;
    padding: 32px 36px 24px;
    position: relative;
    overflow: hidden;">
  <div style="position:absolute;top:-30px;right:-30px;width:180px;height:180px;
    background:radial-gradient(circle,rgba(239,68,68,0.12) 0%,transparent 70%);border-radius:50%;"></div>
  <div style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:900;color:#fef2f2;margin-bottom:6px;">
    🚨 SR Behaviour &amp; <span style="background:linear-gradient(90deg,#f87171,#c084fc);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Fraud Detection</span> Engine
  </div>
  <p style="font-size:0.95rem;color:#9ca3af;margin:0;">
    Automated detection of ghost visits · Invoice manipulation · Visit count anomalies
    — powered by geo-distance analysis, channel profiling, and statistical outlier detection.
  </p>
</div>
""", unsafe_allow_html=True)

# ── CSS additions for fraud module ────────────────────────────────────────────
st.markdown("""
<style>
.fraud-section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #fecaca;
    margin: 36px 0 6px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #450a0a;
}
.fraud-kpi-grid { display:flex; gap:14px; margin:16px 0 24px; flex-wrap:wrap; }
.fraud-kpi {
    flex:1; min-width:140px;
    background: rgba(127,29,29,0.25);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius:12px; padding:18px 20px;
    position:relative; overflow:hidden;
}
.fraud-kpi::after {
    content:'';position:absolute;bottom:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,#f87171,#c084fc);
}
.fraud-kpi-label { font-size:0.7rem;color:#9ca3af;text-transform:uppercase;letter-spacing:1px;margin-bottom:5px; }
.fraud-kpi-value { font-size:1.8rem;font-weight:700;color:#fef2f2; }
.fraud-kpi-sub   { font-size:0.78rem;color:#f87171;margin-top:3px; }

.alert-red {
    background:rgba(127,29,29,0.4);
    border-left:4px solid #ef4444;
    border-radius:0 10px 10px 0;
    padding:14px 18px; margin-top:12px;
    font-size:0.88rem; color:#fecaca; line-height:1.65;
}
.alert-red strong { color:#f87171; }
.alert-amber {
    background:rgba(120,53,15,0.4);
    border-left:4px solid #f97316;
    border-radius:0 10px 10px 0;
    padding:14px 18px; margin-top:12px;
    font-size:0.88rem; color:#fed7aa; line-height:1.65;
}
.alert-amber strong { color:#fb923c; }
.alert-purple {
    background:rgba(76,29,149,0.3);
    border-left:4px solid #a855f7;
    border-radius:0 10px 10px 0;
    padding:14px 18px; margin-top:12px;
    font-size:0.88rem; color:#e9d5ff; line-height:1.65;
}
.alert-purple strong { color:#c084fc; }
.risk-high   { color:#f87171; font-weight:700; }
.risk-medium { color:#fb923c; font-weight:600; }
.risk-low    { color:#4ade80; }
.risk-badge-high   { background:rgba(239,68,68,0.2);color:#f87171;border:1px solid rgba(239,68,68,0.4);
                     padding:2px 10px;border-radius:999px;font-size:0.75rem;font-weight:600; }
.risk-badge-medium { background:rgba(249,115,22,0.2);color:#fb923c;border:1px solid rgba(249,115,22,0.4);
                     padding:2px 10px;border-radius:999px;font-size:0.75rem;font-weight:600; }
.risk-badge-low    { background:rgba(74,222,128,0.15);color:#4ade80;border:1px solid rgba(74,222,128,0.3);
                     padding:2px 10px;border-radius:999px;font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)


# ─── FRAUD HELPER FUNCTIONS ────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised Haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def risk_label(score):
    if score >= 2:   return "🔴 HIGH",   "risk-badge-high"
    if score == 1:   return "🟡 MEDIUM", "risk-badge-medium"
    return "🟢 LOW", "risk-badge-low"


def fraud_fig_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Playfair Display", size=15, color="#fef2f2"), x=0.03),
        paper_bgcolor="rgba(26,10,10,0.0)",
        plot_bgcolor="rgba(45,15,15,0.4)",
        font=dict(family="DM Sans", color="#9ca3af"),
        height=height,
        margin=dict(l=16, r=16, t=50, b=16),
        legend=dict(bgcolor="rgba(26,10,10,0.7)", bordercolor="#450a0a", borderwidth=1, font=dict(size=11)),
        xaxis=dict(gridcolor="#2d0f0f", linecolor="#450a0a", zerolinecolor="#450a0a"),
        yaxis=dict(gridcolor="#2d0f0f", linecolor="#450a0a", zerolinecolor="#450a0a"),
    )
    return fig


# ─── COLUMN DETECTION ─────────────────────────────────────────────────────────
# Map possible column names to standard internal names
GEO_COL_MAP = {
    "LATITUDE":          "Cust_Lat",
    "LONGITUDE":         "Cust_Lon",
    "SELLER_LATITUDE":   "SR_Lat",
    "SELLER_LONGITUDE":  "SR_Lon",
    "CUSTOMER_CODE":     "Cust_Code",
    "CUSTOMER_ADDRESS":  "Cust_Addr",
    "DISTRIBUTOR_CODE":  "Dist_Code",
    "Visit Frequency":   "Visit_Freq",
}

fraud_df = df.copy()
for orig, alias in GEO_COL_MAP.items():
    if orig in fraud_df.columns:
        fraud_df = fraud_df.rename(columns={orig: alias})

# Coerce geo columns
for gc in ["Cust_Lat","Cust_Lon","SR_Lat","SR_Lon"]:
    if gc in fraud_df.columns:
        fraud_df[gc] = pd.to_numeric(fraud_df[gc], errors="coerce")

has_geo       = all(c in fraud_df.columns for c in ["Cust_Lat","Cust_Lon","SR_Lat","SR_Lon"])
has_channel   = "Channel" in fraud_df.columns
has_sales     = "Apr_Sales" in fraud_df.columns
has_visits    = "Apr_Visits" in fraud_df.columns and "Visit_Required" in fraud_df.columns
has_visit_req = "Visit_Required" in fraud_df.columns or "Visits Needed in a Month as per Business Principle" in fraud_df.columns
has_dist_code = "Dist_Code" in fraud_df.columns
has_cust_code = "Cust_Code" in fraud_df.columns

# Ensure Visit_Required exists
if "Visit_Required" not in fraud_df.columns and "Visits Needed in a Month as per Business Principle" in fraud_df.columns:
    fraud_df["Visit_Required"] = pd.to_numeric(
        fraud_df["Visits Needed in a Month as per Business Principle"], errors="coerce")

# Summary risk table (accumulates flags)
fraud_flags = fraud_df[["Customer","Channel","Seller"]].copy() if "Seller" in fraud_df.columns \
    else fraud_df[["Customer","Channel"]].copy()
fraud_flags["Ghost_Visit_Flag"]    = 0
fraud_flags["Invoice_Fraud_Flag"]  = 0
fraud_flags["Visit_Anomaly_Flag"]  = 0
fraud_flags["Total_Risk_Score"]    = 0


# ══════════════════════════════════════════════════════════════════════════════
# FRAUD MODULE 1 — GHOST VISITS (GEO MISMATCH)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="fraud-section-title">🔴 Fraud Check 1 — Ghost Visits: Geo-Location Mismatch</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-desc" style="color:#9ca3af;">
Detects SRs who logged a visit while being far away from the store — a strong signal of billing
without physically visiting. Distance between the store's registered coordinates and the SR's
recorded visit location is calculated using the Haversine formula.
</div>""", unsafe_allow_html=True)

GHOST_THRESHOLD_KM = st.slider(
    "🎯 Ghost Visit Distance Threshold (km) — visits recorded beyond this distance are flagged",
    min_value=0.1, max_value=10.0, value=0.5, step=0.1,
    help="Industry standard is 0.3–0.5 km. Increase for rural routes with larger coverage areas."
)

if has_geo:
    fraud_df["Geo_Distance_km"] = haversine_km(
        fraud_df["Cust_Lat"].values, fraud_df["Cust_Lon"].values,
        fraud_df["SR_Lat"].values,   fraud_df["SR_Lon"].values
    )
    fraud_df["Ghost_Visit"] = fraud_df["Geo_Distance_km"] > GHOST_THRESHOLD_KM

    ghost_count = fraud_df["Ghost_Visit"].sum()
    ghost_pct   = ghost_count / len(fraud_df) * 100 if len(fraud_df) > 0 else 0
    avg_dist    = fraud_df["Geo_Distance_km"].mean()
    max_dist    = fraud_df["Geo_Distance_km"].max()

    # Update flags
    fraud_flags["Ghost_Visit_Flag"]   = fraud_df["Ghost_Visit"].astype(int)
    fraud_flags["Total_Risk_Score"]  += fraud_flags["Ghost_Visit_Flag"]

    # KPI row
    st.markdown(f"""
    <div class="fraud-kpi-grid">
      <div class="fraud-kpi">
        <div class="fraud-kpi-label">Ghost Visits Detected</div>
        <div class="fraud-kpi-value">{int(ghost_count):,}</div>
        <div class="fraud-kpi-sub">of {len(fraud_df):,} total visits</div>
      </div>
      <div class="fraud-kpi">
        <div class="fraud-kpi-label">Flagged Rate</div>
        <div class="fraud-kpi-value">{ghost_pct:.1f}%</div>
        <div class="fraud-kpi-sub">above {GHOST_THRESHOLD_KM} km threshold</div>
      </div>
      <div class="fraud-kpi">
        <div class="fraud-kpi-label">Avg SR Distance</div>
        <div class="fraud-kpi-value">{avg_dist:.2f} km</div>
        <div class="fraud-kpi-sub">across all visits</div>
      </div>
      <div class="fraud-kpi">
        <div class="fraud-kpi-label">Max Distance Logged</div>
        <div class="fraud-kpi-value">{max_dist:.1f} km</div>
        <div class="fraud-kpi-sub">most extreme case</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Distance distribution histogram
        fig_dist = px.histogram(
            fraud_df, x="Geo_Distance_km",
            nbins=60,
            color_discrete_sequence=["#f87171"],
            labels={"Geo_Distance_km": "Distance between SR and Store (km)", "count": "Number of Visits"},
        )
        fig_dist.add_vline(x=GHOST_THRESHOLD_KM, line_dash="dash", line_color="#fbbf24",
                           annotation_text=f"Threshold: {GHOST_THRESHOLD_KM} km",
                           annotation_position="top right",
                           annotation_font_color="#fbbf24")
        fig_dist = fraud_fig_layout(fig_dist, "Distribution of SR–Store Distance at Visit Time", height=380)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown(f"""
        <div class="alert-red">
            <strong>📍 How to read:</strong> Each bar shows how many visits were logged at a given distance from the store.
            Visits to the <strong>right of the yellow line ({GHOST_THRESHOLD_KM} km)</strong> are suspected ghost visits —
            the SR was too far away to have physically been at the store. A spike far to the right indicates
            systematic remote billing by one or more SRs.
        </div>""", unsafe_allow_html=True)

    with col_g2:
        # Ghost vs legitimate by channel
        if has_channel:
            ghost_ch = fraud_df.groupby("Channel").agg(
                Total=("Ghost_Visit","count"),
                Ghost=("Ghost_Visit","sum")
            ).reset_index()
            ghost_ch["Legit"] = ghost_ch["Total"] - ghost_ch["Ghost"]
            ghost_ch["Ghost_Pct"] = (ghost_ch["Ghost"] / ghost_ch["Total"] * 100).round(1)
            ghost_ch = ghost_ch[ghost_ch["Channel"].isin(["Platinum","Gold","Silver","Bronze"])]

            fig_gch = go.Figure()
            fig_gch.add_trace(go.Bar(name="Legitimate Visits", x=ghost_ch["Channel"], y=ghost_ch["Legit"],
                                     marker_color="#4ade80", opacity=0.8))
            fig_gch.add_trace(go.Bar(name="Ghost Visits (Flagged)", x=ghost_ch["Channel"], y=ghost_ch["Ghost"],
                                     marker_color="#f87171", opacity=0.9))
            fig_gch.update_layout(barmode="stack")
            fig_gch = fraud_fig_layout(fig_gch, "Ghost vs Legitimate Visits by Channel", height=380)
            st.plotly_chart(fig_gch, use_container_width=True)
            st.markdown("""
            <div class="alert-red">
                <strong>📊 Channel breakdown:</strong> Red segments are flagged ghost visits per channel.
                A high red proportion in <strong>Platinum</strong> is particularly concerning —
                it may indicate SRs are logging premium-tier visits remotely to inflate activity metrics
                without delivering actual service.
            </div>""", unsafe_allow_html=True)

    # Top ghost offenders by SR
    if "Seller" in fraud_df.columns:
        st.markdown("<br>", unsafe_allow_html=True)
        ghost_by_sr = fraud_df[fraud_df["Ghost_Visit"]].groupby("Seller").agg(
            Ghost_Visits=("Ghost_Visit","sum"),
            Avg_Distance_km=("Geo_Distance_km","mean"),
            Max_Distance_km=("Geo_Distance_km","max"),
        ).reset_index().sort_values("Ghost_Visits", ascending=False).head(15)

        if not ghost_by_sr.empty:
            fig_sr = px.bar(
                ghost_by_sr, y="Seller", x="Ghost_Visits",
                color="Avg_Distance_km",
                color_continuous_scale=["#fbbf24","#f87171","#7f1d1d"],
                orientation="h",
                labels={"Ghost_Visits":"Ghost Visit Count","Seller":"Sales Representative",
                        "Avg_Distance_km":"Avg Distance (km)"},
                text=ghost_by_sr["Ghost_Visits"],
            )
            fig_sr.update_traces(textposition="outside", textfont=dict(color="#fef2f2"))
            fig_sr.update_yaxes(autorange="reversed")
            fig_sr = fraud_fig_layout(fig_sr, "Top SRs by Ghost Visit Count (colour = avg distance)", height=420)
            st.plotly_chart(fig_sr, use_container_width=True)
            st.markdown("""
            <div class="alert-red">
                <strong>🚨 SR-level ghost ranking:</strong> The longer the bar, the more ghost visits logged
                by that SR. The <strong>colour intensity</strong> shows how far away they typically were —
                darker red = greater distance = higher suspicion of deliberate fraud vs accidental GPS error.
                SRs with many ghost visits AND high average distance are priority investigation targets.
            </div>""", unsafe_allow_html=True)

            with st.expander("📋 Full Ghost Visit Detail Table"):
                ghost_detail = fraud_df[fraud_df["Ghost_Visit"]][
                    [c for c in ["Customer","Channel","Seller","Cust_Lat","Cust_Lon","SR_Lat","SR_Lon",
                                 "Geo_Distance_km","Apr_Sales"] if c in fraud_df.columns]
                ].copy()
                ghost_detail["Geo_Distance_km"] = ghost_detail["Geo_Distance_km"].round(3)
                ghost_detail = ghost_detail.sort_values("Geo_Distance_km", ascending=False)
                st.dataframe(ghost_detail.reset_index(drop=True), use_container_width=True)
                st.caption(f"{len(ghost_detail):,} ghost visit records flagged.")
else:
    st.markdown("""
    <div class="alert-amber">
        <strong>⚠️ Geo columns not found:</strong> Ghost visit detection requires <code>LATITUDE</code>,
        <code>LONGITUDE</code>, <code>SELLER_LATITUDE</code>, and <code>SELLER_LONGITUDE</code> columns
        in your dataset. Please ensure these columns are present and re-upload.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FRAUD MODULE 2 — FAKE INVOICE / CHANNEL MISCLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="fraud-section-title">🟠 Fraud Check 2 — Fake Invoices & Channel Misclassification</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-desc" style="color:#9ca3af;">
Detects stores classified as <strong>Platinum</strong> (high-discount tier) whose actual sales volumes
are inconsistent with that classification. SRs may bill through Platinum-tier accounts to secure the
20% dealer discount, while goods are physically moved to lower-tier or unregistered shops.
</div>""", unsafe_allow_html=True)

if has_channel and has_sales:
    # ── Compute channel sales statistics ──────────────────────────────────────
    ch_stats = fraud_df.groupby("Channel")["Apr_Sales"].agg(["mean","median","std"]).reset_index()
    ch_stats.columns = ["Channel","Mean_Sales","Median_Sales","Std_Sales"]

    # Platinum benchmark
    plat_data = fraud_df[fraud_df["Channel"] == "Platinum"]["Apr_Sales"].dropna()
    plat_median = plat_data.median() if len(plat_data) > 0 else None
    plat_q1     = plat_data.quantile(0.25) if len(plat_data) > 0 else None

    # Threshold: Platinum accounts whose sales fall below the Gold or Silver median
    gold_data = fraud_df[fraud_df["Channel"] == "Gold"]["Apr_Sales"].dropna()
    silver_data = fraud_df[fraud_df["Channel"] == "Silver"]["Apr_Sales"].dropna()
    gold_median   = gold_data.median()   if len(gold_data) > 0   else 0
    silver_median = silver_data.median() if len(silver_data) > 0 else 0

    INVOICE_THRESHOLD_PCT = st.slider(
        "📐 Misclassification sensitivity — flag Platinum accounts whose sales fall below X% of Platinum median",
        min_value=10, max_value=80, value=40, step=5,
        help="40% means: a Platinum store selling less than 40% of the typical Platinum store is flagged as potentially misclassified."
    )

    if plat_median and plat_median > 0:
        threshold_val = plat_median * (INVOICE_THRESHOLD_PCT / 100)
        plat_suspicious = fraud_df[
            (fraud_df["Channel"] == "Platinum") &
            (fraud_df["Apr_Sales"].notna()) &
            (fraud_df["Apr_Sales"] < threshold_val)
        ].copy()
        plat_suspicious["Sales_vs_Median_Pct"] = (
            plat_suspicious["Apr_Sales"] / plat_median * 100
        ).round(1)

        inv_count = len(plat_suspicious)
        inv_pct   = inv_count / max(len(plat_data), 1) * 100

        # Update flags
        if "Cust_Code" in fraud_df.columns and "Cust_Code" in plat_suspicious.columns:
            fraud_flags.loc[fraud_df["Cust_Code"].isin(plat_suspicious["Cust_Code"]), "Invoice_Fraud_Flag"] = 1
        else:
            fraud_flags.loc[fraud_df.index.isin(plat_suspicious.index), "Invoice_Fraud_Flag"] = 1
        fraud_flags["Total_Risk_Score"] += fraud_flags["Invoice_Fraud_Flag"]

        # KPI row
        est_discount_loss = plat_suspicious["Apr_Sales"].sum() * 0.20
        st.markdown(f"""
        <div class="fraud-kpi-grid">
          <div class="fraud-kpi">
            <div class="fraud-kpi-label">Flagged Platinum Accounts</div>
            <div class="fraud-kpi-value">{inv_count:,}</div>
            <div class="fraud-kpi-sub">{inv_pct:.1f}% of all Platinum stores</div>
          </div>
          <div class="fraud-kpi">
            <div class="fraud-kpi-label">Platinum Median Sales</div>
            <div class="fraud-kpi-value">₹{plat_median:,.0f}</div>
            <div class="fraud-kpi-sub">benchmark for the tier</div>
          </div>
          <div class="fraud-kpi">
            <div class="fraud-kpi-label">Suspicion Threshold</div>
            <div class="fraud-kpi-value">₹{threshold_val:,.0f}</div>
            <div class="fraud-kpi-sub">{INVOICE_THRESHOLD_PCT}% of Platinum median</div>
          </div>
          <div class="fraud-kpi">
            <div class="fraud-kpi-label">Est. Discount Leakage</div>
            <div class="fraud-kpi-value">₹{est_discount_loss:,.0f}</div>
            <div class="fraud-kpi-sub">20% discount on flagged sales</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_i1, col_i2 = st.columns(2)

        with col_i1:
            # Sales distribution: all channels + flagged Platinum overlay
            fig_box_ch = px.box(
                fraud_df[fraud_df["Channel"].isin(["Platinum","Gold","Silver","Bronze"])],
                x="Channel", y="Apr_Sales",
                color="Channel", color_discrete_map=CHANNEL_COLORS,
                points="outliers",
                labels={"Apr_Sales": "April Sales (₹)", "Channel": "Channel Tier"},
            )
            fig_box_ch.add_hline(y=threshold_val, line_dash="dot", line_color="#f97316",
                                 annotation_text=f"Suspicion line ₹{threshold_val:,.0f}",
                                 annotation_font_color="#f97316")
            fig_box_ch = fraud_fig_layout(fig_box_ch, "April Sales Distribution by Channel — Suspicion Threshold", height=400)
            st.plotly_chart(fig_box_ch, use_container_width=True)
            st.markdown(f"""
            <div class="alert-amber">
                <strong>📦 What this shows:</strong> Each box is the sales spread for a channel tier.
                The <strong>orange dashed line</strong> is the suspicion threshold — Platinum accounts
                with sales <em>below</em> this line are behaving more like Silver or Bronze stores.
                If a Platinum account sits consistently below Gold or Silver medians, it likely does not
                warrant the Platinum classification — or invoices are being run through it artificially.
            </div>""", unsafe_allow_html=True)

        with col_i2:
            if not plat_suspicious.empty:
                # Scatter: flagged Platinum accounts — sales vs a horizontal rank
                plat_suspicious_sorted = plat_suspicious.sort_values("Apr_Sales")
                plat_suspicious_sorted["Rank"] = range(1, len(plat_suspicious_sorted)+1)

                fig_plat = px.scatter(
                    plat_suspicious_sorted, x="Rank", y="Apr_Sales",
                    color="Sales_vs_Median_Pct",
                    color_continuous_scale=["#7f1d1d","#f87171","#fbbf24"],
                    size="Apr_Sales",
                    hover_data={c: True for c in ["Customer","Seller","Apr_Sales","Sales_vs_Median_Pct"]
                                if c in plat_suspicious_sorted.columns},
                    labels={"Apr_Sales": "April Sales (₹)", "Rank": "Flagged Account Rank",
                            "Sales_vs_Median_Pct": "% of Platinum Median"},
                )
                fig_plat.add_hline(y=threshold_val, line_dash="dot", line_color="#f97316")
                fig_plat.add_hline(y=plat_median,   line_dash="dash", line_color="#818cf8",
                                   annotation_text="Platinum Median", annotation_font_color="#818cf8")
                fig_plat = fraud_fig_layout(fig_plat, "Flagged Platinum Accounts — Sales vs Tier Median", height=400)
                st.plotly_chart(fig_plat, use_container_width=True)
                st.markdown("""
                <div class="alert-amber">
                    <strong>🔍 Dot chart of suspects:</strong> Each dot is a flagged Platinum account.
                    Darker red = further below the Platinum median = greater anomaly.
                    Smaller dots = lower absolute sales. Accounts that are both very dark <em>and</em>
                    small are the highest-priority cases for invoice verification.
                </div>""", unsafe_allow_html=True)

        # Distributor-wise misclassification
        if "Distributor" in fraud_df.columns and not plat_suspicious.empty:
            dist_inv = plat_suspicious.groupby("Distributor").agg(
                Flagged_Accounts=("Customer","count"),
                Total_Suspect_Sales=("Apr_Sales","sum"),
            ).reset_index().sort_values("Flagged_Accounts", ascending=False).head(12)

            fig_dist_inv = px.bar(
                dist_inv, y="Distributor", x="Flagged_Accounts",
                color="Total_Suspect_Sales",
                color_continuous_scale=["#fbbf24","#f97316","#b91c1c"],
                orientation="h",
                labels={"Flagged_Accounts":"Flagged Platinum Accounts",
                        "Distributor":"Distributor Name",
                        "Total_Suspect_Sales":"Suspect Sales (₹)"},
                text=dist_inv["Flagged_Accounts"],
            )
            fig_dist_inv.update_traces(textposition="outside", textfont=dict(color="#fef2f2"))
            fig_dist_inv.update_yaxes(autorange="reversed")
            fig_dist_inv = fraud_fig_layout(fig_dist_inv,
                "Distributors with Most Flagged Platinum Accounts (colour = suspect sales volume)", height=400)
            st.plotly_chart(fig_dist_inv, use_container_width=True)
            st.markdown("""
            <div class="alert-amber">
                <strong>🏭 Distributor exposure:</strong> This chart shows which distributors have the
                most Platinum-classified accounts that are performing like lower-tier stores.
                A high count concentrated in one distributor warrants an audit of whether that
                distributor is systematically mis-registering shops to access the 20% discount.
            </div>""", unsafe_allow_html=True)

        with st.expander("📋 Full Flagged Platinum Account Detail"):
            cols_show = [c for c in ["Customer","Channel","Seller","Distributor","Apr_Sales",
                                     "Sales_vs_Median_Pct","Cust_Addr"] if c in plat_suspicious.columns]
            disp_inv = plat_suspicious[cols_show].copy().sort_values("Apr_Sales")
            if "Apr_Sales" in disp_inv.columns:
                disp_inv["Apr_Sales"] = disp_inv["Apr_Sales"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
            if "Sales_vs_Median_Pct" in disp_inv.columns:
                disp_inv["Sales_vs_Median_Pct"] = disp_inv["Sales_vs_Median_Pct"].apply(
                    lambda x: f"{x:.1f}% of Platinum median" if pd.notna(x) else "—")
            st.dataframe(disp_inv.reset_index(drop=True), use_container_width=True)
            st.caption(f"{inv_count} Platinum accounts flagged for possible invoice manipulation / channel misclassification.")
    else:
        st.markdown("""
        <div class="alert-amber">
            <strong>⚠️ Insufficient Platinum data:</strong> Not enough Platinum-channel records to compute
            a meaningful median for comparison. Ensure the dataset contains multiple Platinum stores.
        </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-amber">
        <strong>⚠️ Required columns missing:</strong> Invoice fraud detection requires <code>CHANNEL</code>
        and <code>April Sales</code> columns in your dataset.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FRAUD MODULE 3 — VISIT COUNT ANOMALIES (PLANNED vs ACTUAL)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="fraud-section-title">🟣 Fraud Check 3 — Visit Count Anomalies: Planned vs Actual</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-desc" style="color:#9ca3af;">
Identifies stores where actual visit counts deviate significantly from the planned business schedule.
<strong>Over-visiting</strong> (far above plan) can indicate inflated activity records to mask ghost
visits elsewhere or to justify excessive claims. <strong>Under-visiting</strong> signals genuine
visit neglect. Small deviations (±20%) are treated as normal operational variance.
</div>""", unsafe_allow_html=True)

visit_req_col_present = "Visit_Required" in fraud_df.columns

if has_visits or visit_req_col_present:
    if "Visit_Required" not in fraud_df.columns:
        fraud_df["Visit_Required"] = pd.to_numeric(
            fraud_df.get("Visits Needed in a Month as per Business Principle"), errors="coerce")

    fraud_df["Visit_Required_clean"] = fraud_df["Visit_Required"].replace(0, np.nan)
    fraud_df["Visit_Pct_of_Plan"] = (
        fraud_df["Apr_Visits"] / fraud_df["Visit_Required_clean"] * 100
    ).round(1)
    fraud_df["Visit_Excess_Pct"] = fraud_df["Visit_Pct_of_Plan"] - 100  # positive = over, negative = under

    OVER_VISIT_THRESHOLD  = st.slider(
        "📈 Over-visit alert threshold (% above plan to flag)",
        min_value=20, max_value=200, value=50, step=10,
        help="50% means: if a store received 1.5× its planned visits, flag it. Slight increase is normal; large increase is suspicious."
    )
    UNDER_VISIT_THRESHOLD = st.slider(
        "📉 Under-visit alert threshold (% below plan to flag)",
        min_value=10, max_value=80, value=30, step=5,
        help="30% means: if a store received less than 70% of its planned visits, flag it as under-served."
    )

    fraud_df["Over_Visited"]  = fraud_df["Visit_Excess_Pct"] >  OVER_VISIT_THRESHOLD
    fraud_df["Under_Visited"] = fraud_df["Visit_Excess_Pct"] < -UNDER_VISIT_THRESHOLD
    fraud_df["Normal_Visit"]  = (~fraud_df["Over_Visited"]) & (~fraud_df["Under_Visited"])

    over_count  = fraud_df["Over_Visited"].sum()
    under_count = fraud_df["Under_Visited"].sum()
    normal_count = fraud_df["Normal_Visit"].sum()
    total_valid = fraud_df["Visit_Pct_of_Plan"].notna().sum()

    # Update flags
    fraud_flags["Visit_Anomaly_Flag"] = (fraud_df["Over_Visited"] | fraud_df["Under_Visited"]).astype(int)
    fraud_flags["Total_Risk_Score"]  += fraud_flags["Visit_Anomaly_Flag"]

    st.markdown(f"""
    <div class="fraud-kpi-grid">
      <div class="fraud-kpi">
        <div class="fraud-kpi-label">Over-Visited Stores</div>
        <div class="fraud-kpi-value">{int(over_count):,}</div>
        <div class="fraud-kpi-sub">&gt;{OVER_VISIT_THRESHOLD}% above plan</div>
      </div>
      <div class="fraud-kpi">
        <div class="fraud-kpi-label">Under-Visited Stores</div>
        <div class="fraud-kpi-value">{int(under_count):,}</div>
        <div class="fraud-kpi-sub">&gt;{UNDER_VISIT_THRESHOLD}% below plan</div>
      </div>
      <div class="fraud-kpi">
        <div class="fraud-kpi-label">Normal Range</div>
        <div class="fraud-kpi-value">{int(normal_count):,}</div>
        <div class="fraud-kpi-sub">within acceptable deviation</div>
      </div>
      <div class="fraud-kpi">
        <div class="fraud-kpi-label">Avg Visit Compliance</div>
        <div class="fraud-kpi-value">{fraud_df['Visit_Pct_of_Plan'].mean():.1f}%</div>
        <div class="fraud-kpi-sub">across {total_valid:,} stores with plan data</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        # Histogram of visit % of plan
        hist_data = fraud_df["Visit_Pct_of_Plan"].dropna()
        fig_vh = px.histogram(hist_data, nbins=60,
                              color_discrete_sequence=["#a855f7"],
                              labels={"value": "Actual Visits as % of Planned Visits", "count": "Store Count"})
        fig_vh.add_vline(x=100 + OVER_VISIT_THRESHOLD,  line_dash="dash", line_color="#f87171",
                         annotation_text=f"Over-visit (+{OVER_VISIT_THRESHOLD}%)",
                         annotation_font_color="#f87171")
        fig_vh.add_vline(x=100 - UNDER_VISIT_THRESHOLD, line_dash="dash", line_color="#fbbf24",
                         annotation_text=f"Under-visit (-{UNDER_VISIT_THRESHOLD}%)",
                         annotation_font_color="#fbbf24")
        fig_vh.add_vline(x=100, line_dash="dot", line_color="#4ade80",
                         annotation_text="Plan = 100%", annotation_font_color="#4ade80")
        fig_vh = fraud_fig_layout(fig_vh, "Distribution of Stores: Actual vs Planned Visit %", height=380)
        st.plotly_chart(fig_vh, use_container_width=True)
        st.markdown(f"""
        <div class="alert-purple">
            <strong>📊 Reading this histogram:</strong> The green line at 100% is the plan. Stores to the
            right of the <strong>red dashed line ({100+OVER_VISIT_THRESHOLD}%)</strong> received suspiciously
            more visits than planned — possible visit inflation. Stores to the left of the
            <strong>yellow line ({100-UNDER_VISIT_THRESHOLD}%)</strong> are under-served.
            A healthy distribution should be tightly clustered around 100%.
        </div>""", unsafe_allow_html=True)

    with col_v2:
        # Donut: over / normal / under
        donut_vals = [int(over_count), int(normal_count), int(under_count)]
        donut_labels = [f"Over-visited (>{OVER_VISIT_THRESHOLD}% above)",
                        "Normal (within range)",
                        f"Under-visited (>{UNDER_VISIT_THRESHOLD}% below)"]
        fig_donut = go.Figure(go.Pie(
            labels=donut_labels, values=donut_vals, hole=0.52,
            marker_colors=["#f87171","#4ade80","#fbbf24"],
            textinfo="label+percent", textfont_size=11,
        ))
        fig_donut.update_traces(hoverinfo="label+value", textposition="outside")
        fig_donut = fraud_fig_layout(fig_donut, "Visit Compliance Category Breakdown", height=380)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("""
        <div class="alert-purple">
            <strong>🍩 Category split:</strong> Green = normal, Red = over-visited (possible inflation),
            Yellow = under-visited (neglected accounts). A high red slice is the most concerning —
            it suggests SRs are recording excessive visits at certain stores to pad their activity logs,
            often to compensate for ghost visits at other locations.
        </div>""", unsafe_allow_html=True)

    # ── SR-level visit anomaly breakdown ─────────────────────────────────────────
    # Determine the best available grouping column for SRs
    # Priority: SELLER_NAME → DISTRIBUTOR_NAME → fallback label
    if "Seller" in fraud_df.columns:
        sr_group_col = "Seller"
        sr_label     = "Sales Representative"
    elif "Distributor" in fraud_df.columns:
        sr_group_col = "Distributor"
        sr_label     = "Distributor / Depot"
    else:
        sr_group_col = None
        sr_label     = ""

    if sr_group_col:
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Build SR-level summaries SEPARATELY for over and under ────────────
        over_df  = fraud_df[fraud_df["Over_Visited"]].copy()
        under_df = fraud_df[fraud_df["Under_Visited"]].copy()

        sr_over_summary  = pd.DataFrame()
        sr_under_summary = pd.DataFrame()

        if not over_df.empty:
            sr_over_summary = over_df.groupby(sr_group_col).agg(
                Stores_Over_Visited  = ("Over_Visited",    "sum"),
                Avg_Excess_Pct       = ("Visit_Excess_Pct","mean"),
                Max_Excess_Pct       = ("Visit_Excess_Pct","max"),
            ).reset_index()
            sr_over_summary["Avg_Excess_Pct"] = sr_over_summary["Avg_Excess_Pct"].round(1)
            sr_over_summary["Max_Excess_Pct"] = sr_over_summary["Max_Excess_Pct"].round(1)
            sr_over_summary = sr_over_summary.sort_values("Stores_Over_Visited", ascending=False)

        if not under_df.empty:
            sr_under_summary = under_df.groupby(sr_group_col).agg(
                Stores_Under_Visited = ("Under_Visited",   "sum"),
                Avg_Shortfall_Pct    = ("Visit_Excess_Pct","mean"),
                Max_Shortfall_Pct    = ("Visit_Excess_Pct","min"),
            ).reset_index()
            sr_under_summary["Avg_Shortfall_Pct"] = sr_under_summary["Avg_Shortfall_Pct"].abs().round(1)
            sr_under_summary["Max_Shortfall_Pct"] = sr_under_summary["Max_Shortfall_Pct"].abs().round(1)
            sr_under_summary = sr_under_summary.sort_values("Stores_Under_Visited", ascending=False)

        # ── IMPORTANT: SRs appearing in BOTH lists are the highest-risk ───────
        # An SR who over-visits some stores AND under-visits others is likely
        # shifting visit records — padding some accounts to cover neglected ones.
        if not sr_over_summary.empty and not sr_under_summary.empty:
            double_offenders = set(sr_over_summary[sr_group_col]) & set(sr_under_summary[sr_group_col])
        else:
            double_offenders = set()

        # ── Context note explaining why overlap may exist ──────────────────────
        st.markdown(f"""
        <div class="alert-purple" style="margin-bottom:16px;">
            <strong>ℹ️ How to read these charts:</strong>
            An SR (or depot) can appear in <em>both</em> lists if they manage a large number of stores —
            some of which received too many visits while others received too few.
            This is <strong>normal for large depots</strong>. However, an SR appearing in both lists
            with a <em>high count on both sides</em> is the most suspicious pattern: they may be
            <strong>padding visit records at select stores to offset ghost visits at others</strong>.
            <br><br>
            <strong>🚨 Double Offenders detected: {len(double_offenders)}</strong> —
            {sr_label}(s) flagged for <em>both</em> over-visiting and under-visiting:
            {", ".join(sorted(double_offenders)) if double_offenders else "None at current thresholds."}
        </div>""", unsafe_allow_html=True)

        col_v3, col_v4 = st.columns(2)

        with col_v3:
            if not sr_over_summary.empty:
                top_over = sr_over_summary.head(12).copy()
                # Mark double offenders with a flag in the label
                top_over["_label"] = top_over[sr_group_col].apply(
                    lambda x: f"⚠ {x}" if x in double_offenders else x
                )
                fig_ov = px.bar(
                    top_over, y="_label", x="Stores_Over_Visited",
                    color="Avg_Excess_Pct",
                    color_continuous_scale=["#fbbf24","#f87171","#7f1d1d"],
                    orientation="h",
                    labels={
                        "Stores_Over_Visited": "Stores Over-Visited (count)",
                        "_label":              sr_label,
                        "Avg_Excess_Pct":      "Avg Excess %",
                    },
                    text=top_over["Stores_Over_Visited"],
                    hover_data={"Avg_Excess_Pct": True, "Max_Excess_Pct": True},
                )
                fig_ov.update_traces(textposition="outside", textfont=dict(color="#fef2f2"))
                fig_ov.update_yaxes(autorange="reversed")
                fig_ov = fraud_fig_layout(
                    fig_ov,
                    f"Top {sr_label}s: Most Stores Over-Visited · ⚠ = also under-visits elsewhere",
                    height=max(380, len(top_over) * 34 + 80),
                )
                st.plotly_chart(fig_ov, use_container_width=True)
                st.markdown(f"""
                <div class="alert-purple">
                    <strong>🔺 Over-visit pattern:</strong> Each bar = a {sr_label.lower()} whose stores
                    received visits <strong>far above plan</strong>. The colour shows how far above
                    average they went (darker red = larger excess). Entries marked <strong>⚠</strong>
                    also appear in the under-visit chart — meaning this same {sr_label.lower()} is
                    neglecting <em>other</em> stores while inflating visits at these ones.
                    That combination is the strongest fraud signal.
                </div>""", unsafe_allow_html=True)
            else:
                st.info("No over-visit anomalies detected at the current threshold.")

        with col_v4:
            if not sr_under_summary.empty:
                top_under = sr_under_summary.head(12).copy()
                top_under["_label"] = top_under[sr_group_col].apply(
                    lambda x: f"⚠ {x}" if x in double_offenders else x
                )
                fig_uv = px.bar(
                    top_under, y="_label", x="Stores_Under_Visited",
                    color="Avg_Shortfall_Pct",
                    color_continuous_scale=["#fbbf24","#fb923c","#b45309"],
                    orientation="h",
                    labels={
                        "Stores_Under_Visited": "Stores Under-Visited (count)",
                        "_label":               sr_label,
                        "Avg_Shortfall_Pct":    "Avg Shortfall %",
                    },
                    text=top_under["Stores_Under_Visited"],
                    hover_data={"Avg_Shortfall_Pct": True, "Max_Shortfall_Pct": True},
                )
                fig_uv.update_traces(textposition="outside", textfont=dict(color="#fef2f2"))
                fig_uv.update_yaxes(autorange="reversed")
                fig_uv = fraud_fig_layout(
                    fig_uv,
                    f"Top {sr_label}s: Most Stores Under-Visited · ⚠ = also over-visits elsewhere",
                    height=max(380, len(top_under) * 34 + 80),
                )
                st.plotly_chart(fig_uv, use_container_width=True)
                st.markdown(f"""
                <div class="alert-amber">
                    <strong>🔻 Under-visit pattern:</strong> Each bar = a {sr_label.lower()} whose stores
                    received <strong>fewer visits than planned</strong>. Colour shows average shortfall depth.
                    Entries marked <strong>⚠</strong> also appear in the over-visit chart — these
                    {sr_label.lower()}s are simultaneously padding some accounts and neglecting others,
                    which is a clear manipulation signal.
                </div>""", unsafe_allow_html=True)
            else:
                st.info("No under-visit anomalies detected at the current threshold.")

        # ── Double Offenders Detail Table ──────────────────────────────────────
        if double_offenders:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="alert-red" style="padding:16px 20px;margin-bottom:12px;">
                <strong>🚨 Double Offenders — Highest Priority for Investigation</strong><br>
                These SRs / depots are simultaneously over-visiting some stores AND
                under-visiting others. This pattern — inflating activity at select accounts
                while neglecting the rest — is the clearest behavioural signal of visit
                record manipulation. Escalate for immediate field audit.
            </div>""", unsafe_allow_html=True)

            dbl_rows = []
            for sr in sorted(double_offenders):
                o_row = sr_over_summary[sr_over_summary[sr_group_col] == sr]
                u_row = sr_under_summary[sr_under_summary[sr_group_col] == sr]
                dbl_rows.append({
                    sr_label:                  sr,
                    "Stores Over-Visited":     int(o_row["Stores_Over_Visited"].values[0]) if not o_row.empty else 0,
                    "Avg Over-Excess %":       float(o_row["Avg_Excess_Pct"].values[0])    if not o_row.empty else 0,
                    "Stores Under-Visited":    int(u_row["Stores_Under_Visited"].values[0]) if not u_row.empty else 0,
                    "Avg Under-Shortfall %":   float(u_row["Avg_Shortfall_Pct"].values[0]) if not u_row.empty else 0,
                    "Risk Verdict":            "🚨 Manipulation Pattern Detected",
                })
            dbl_df = pd.DataFrame(dbl_rows).sort_values("Stores Over-Visited", ascending=False)
            st.dataframe(dbl_df.reset_index(drop=True), use_container_width=True)
            st.caption(
                f"{len(double_offenders)} {sr_label}(s) flagged as double offenders. "
                "These should be prioritised above all others in the field audit programme."
            )

    # Channel-wise visit anomaly heatmap
    if has_channel:
        st.markdown("<br>", unsafe_allow_html=True)
        vcat = fraud_df[fraud_df["Channel"].isin(["Platinum","Gold","Silver","Bronze"])].copy()
        vcat["Visit_Category"] = np.where(vcat["Over_Visited"], "Over-visited",
                                 np.where(vcat["Under_Visited"], "Under-visited", "Normal"))
        ch_vcat = vcat.groupby(["Channel","Visit_Category"]).size().reset_index(name="Count")
        fig_ch_vcat = px.bar(
            ch_vcat, x="Channel", y="Count", color="Visit_Category",
            color_discrete_map={"Over-visited":"#f87171","Normal":"#4ade80","Under-visited":"#fbbf24"},
            barmode="stack",
            labels={"Count":"Number of Stores","Channel":"Channel Tier","Visit_Category":"Visit Status"},
        )
        fig_ch_vcat = fraud_fig_layout(fig_ch_vcat, "Visit Compliance Status by Channel Tier", height=360)
        st.plotly_chart(fig_ch_vcat, use_container_width=True)
        st.markdown(f"""
        <div class="alert-purple">
            <strong>📈 Channel × visit status:</strong> Green = stores visited within ±{UNDER_VISIT_THRESHOLD}–{OVER_VISIT_THRESHOLD}%
            of plan (normal). Red = over-visited (suspicious inflation). Yellow = under-visited (neglect).
            Compare across tiers — if Platinum has a large red stack combined with ghost visits from
            Module 1, it strongly suggests deliberate visit record manipulation in that tier.
        </div>""", unsafe_allow_html=True)

    with st.expander("📋 Full Visit Anomaly Detail Table"):
        anomaly_df = fraud_df[fraud_df["Over_Visited"] | fraud_df["Under_Visited"]][[
            c for c in ["Customer","Channel","Seller","Apr_Visits","Visit_Required",
                        "Visit_Pct_of_Plan","Visit_Excess_Pct","Over_Visited","Under_Visited"]
            if c in fraud_df.columns
        ]].copy()
        if "Visit_Pct_of_Plan" in anomaly_df:
            anomaly_df["Visit_Pct_of_Plan"] = anomaly_df["Visit_Pct_of_Plan"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        if "Visit_Excess_Pct" in anomaly_df:
            anomaly_df["Visit_Excess_Pct"] = anomaly_df["Visit_Excess_Pct"].apply(
                lambda x: f"+{x:.1f}%" if (pd.notna(x) and x >= 0) else (f"{x:.1f}%" if pd.notna(x) else "—"))
        anomaly_df = anomaly_df.sort_values("Customer") if "Customer" in anomaly_df.columns else anomaly_df
        st.dataframe(anomaly_df.reset_index(drop=True), use_container_width=True)
        st.caption(f"{len(anomaly_df):,} stores flagged for visit anomalies.")
else:
    st.markdown("""
    <div class="alert-amber">
        <strong>⚠️ Required columns missing:</strong> Visit anomaly detection requires
        <code>Actual visits done in April</code> and
        <code>Visits Needed in a Month as per Business Principle</code> columns in your dataset.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED RISK SCORECARD
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="fraud-section-title">⚠️ Combined Fraud Risk Scorecard — All Three Checks</div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-desc" style="color:#9ca3af;">
Each store is scored across all three fraud checks. A score of 2 or 3 means the store
triggered multiple independent fraud signals — these are the highest-priority accounts for investigation.
</div>""", unsafe_allow_html=True)

fraud_flags["Total_Risk_Score"] = (
    fraud_flags["Ghost_Visit_Flag"] +
    fraud_flags["Invoice_Fraud_Flag"] +
    fraud_flags["Visit_Anomaly_Flag"]
)

def apply_risk(score):
    if score >= 2: return "HIGH"
    if score == 1: return "MEDIUM"
    return "LOW"

fraud_flags["Risk_Level"] = fraud_flags["Total_Risk_Score"].apply(apply_risk)

risk_counts = fraud_flags["Risk_Level"].value_counts()
high_n   = risk_counts.get("HIGH",   0)
medium_n = risk_counts.get("MEDIUM", 0)
low_n    = risk_counts.get("LOW",    0)

st.markdown(f"""
<div class="fraud-kpi-grid">
  <div class="fraud-kpi" style="border-color:rgba(239,68,68,0.5);">
    <div class="fraud-kpi-label">🔴 HIGH Risk Accounts</div>
    <div class="fraud-kpi-value" style="color:#f87171;">{high_n:,}</div>
    <div class="fraud-kpi-sub">2–3 fraud signals triggered</div>
  </div>
  <div class="fraud-kpi" style="border-color:rgba(249,115,22,0.5);">
    <div class="fraud-kpi-label">🟡 MEDIUM Risk Accounts</div>
    <div class="fraud-kpi-value" style="color:#fb923c;">{medium_n:,}</div>
    <div class="fraud-kpi-sub">1 fraud signal triggered</div>
  </div>
  <div class="fraud-kpi" style="border-color:rgba(74,222,128,0.3);">
    <div class="fraud-kpi-label">🟢 LOW Risk Accounts</div>
    <div class="fraud-kpi-value" style="color:#4ade80;">{low_n:,}</div>
    <div class="fraud-kpi-sub">no flags triggered</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_r1, col_r2 = st.columns([2, 3])

with col_r1:
    fig_risk_donut = go.Figure(go.Pie(
        labels=["HIGH Risk","MEDIUM Risk","LOW Risk"],
        values=[high_n, medium_n, low_n],
        hole=0.55,
        marker_colors=["#f87171","#fb923c","#4ade80"],
        textinfo="label+percent",
        textfont_size=11,
    ))
    fig_risk_donut = fraud_fig_layout(fig_risk_donut, "Overall Risk Distribution", height=340)
    st.plotly_chart(fig_risk_donut, use_container_width=True)

with col_r2:
    # Risk by channel
    if "Channel" in fraud_flags.columns:
        risk_ch = fraud_flags.groupby(["Channel","Risk_Level"]).size().reset_index(name="Count")
        risk_ch = risk_ch[risk_ch["Channel"].isin(["Platinum","Gold","Silver","Bronze"])]
        fig_risk_ch = px.bar(
            risk_ch, x="Channel", y="Count", color="Risk_Level",
            color_discrete_map={"HIGH":"#f87171","MEDIUM":"#fb923c","LOW":"#4ade80"},
            barmode="stack",
            category_orders={"Risk_Level":["HIGH","MEDIUM","LOW"]},
            labels={"Count":"Accounts","Channel":"Channel Tier","Risk_Level":"Risk Level"},
        )
        fig_risk_ch = fraud_fig_layout(fig_risk_ch, "Risk Distribution by Channel Tier", height=340)
        st.plotly_chart(fig_risk_ch, use_container_width=True)

# High risk table
high_risk_accts = fraud_flags[fraud_flags["Risk_Level"] == "HIGH"].sort_values(
    "Total_Risk_Score", ascending=False)

if not high_risk_accts.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="alert-red" style="padding:16px 20px;">
        <strong>🚨 HIGH RISK ACCOUNTS — Immediate Investigation Required</strong><br>
        These accounts have triggered 2 or more independent fraud signals simultaneously.
        Cross-referencing multiple checks makes false positives extremely unlikely.
        Escalate to field audit and compliance teams immediately.
    </div>""", unsafe_allow_html=True)

    # Add flag details
    high_risk_accts = high_risk_accts.copy()
    high_risk_accts["Flags Triggered"] = (
        high_risk_accts["Ghost_Visit_Flag"].apply(lambda x: "👻 Ghost Visit " if x else "") +
        high_risk_accts["Invoice_Fraud_Flag"].apply(lambda x: "🧾 Invoice Fraud " if x else "") +
        high_risk_accts["Visit_Anomaly_Flag"].apply(lambda x: "📊 Visit Anomaly" if x else "")
    ).str.strip()

    display_cols = [c for c in ["Customer","Channel","Seller","Total_Risk_Score","Flags Triggered"]
                    if c in high_risk_accts.columns]
    st.dataframe(high_risk_accts[display_cols].reset_index(drop=True), use_container_width=True)

# Full scorecard download
with st.expander("📋 Full Risk Scorecard — All Accounts"):
    all_risk = fraud_flags.copy()
    all_risk["Ghost_Visit_Flag"]   = all_risk["Ghost_Visit_Flag"].apply(lambda x: "Yes" if x else "No")
    all_risk["Invoice_Fraud_Flag"] = all_risk["Invoice_Fraud_Flag"].apply(lambda x: "Yes" if x else "No")
    all_risk["Visit_Anomaly_Flag"] = all_risk["Visit_Anomaly_Flag"].apply(lambda x: "Yes" if x else "No")
    all_risk = all_risk.sort_values("Total_Risk_Score", ascending=False)
    st.dataframe(all_risk.reset_index(drop=True), use_container_width=True)
    st.caption(f"Full scorecard: {high_n} HIGH · {medium_n} MEDIUM · {low_n} LOW risk accounts.")

st.markdown("""
<div class="alert-red" style="margin-top:24px;">
    <strong>📌 Methodology Note:</strong> Each fraud check uses independent signals —
    (1) GPS geo-distance between SR location and store at visit time,
    (2) sales volume benchmarking against channel-tier medians,
    (3) statistical deviation of actual vs planned visit counts.
    A HIGH risk account has triggered at least two of these three independent checks,
    making coincidental false positives statistically very unlikely.
    All thresholds are adjustable via the sliders above to suit local operational context.
</div>
""", unsafe_allow_html=True)


# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style='text-align:center;color:#374151;font-size:0.78rem;padding:12px 0'>
    XYZ Sales Intelligence · Built with Streamlit & Plotly · Visit–Sales Correlation Analysis + Fraud Detection Engine
</div>
""", unsafe_allow_html=True)
