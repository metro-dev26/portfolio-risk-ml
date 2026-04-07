"""
ML-Based Portfolio Risk Modeling Dashboard
===========================================
Streamlit Cloud deployment — push to GitHub root as app.py
"""
 
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.stats import norm, genpareto
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark financial terminal aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
 
:root {
    --bg-primary   : #0a0e1a;
    --bg-secondary : #111827;
    --bg-card      : #1a2235;
    --accent-green : #00ff88;
    --accent-red   : #ff4757;
    --accent-blue  : #3d9bff;
    --accent-yellow: #ffd32a;
    --text-primary : #e8eaf0;
    --text-muted   : #6b7a99;
    --border       : #1e2d45;
}
 
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}
 
.stApp { background-color: var(--bg-primary); }
 
/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
 
/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin: 8px 0;
    border-left: 3px solid var(--accent-blue);
}
.metric-card.red   { border-left-color: var(--accent-red); }
.metric-card.green { border-left-color: var(--accent-green); }
.metric-card.yellow{ border-left-color: var(--accent-yellow); }
 
.metric-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-primary);
    line-height: 1.1;
}
.metric-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
}
 
/* Section headers */
.section-header {
    font-size: 13px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-blue);
    font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 32px 0 20px 0;
}
 
/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0d1b2e 0%, #0a1628 50%, #0d1b2e 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-green), transparent);
}
.hero-title {
    font-size: 36px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #e8eaf0, #3d9bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 15px;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 20px;
}
.tag {
    display: inline-block;
    background: rgba(61, 155, 255, 0.1);
    border: 1px solid rgba(61, 155, 255, 0.3);
    color: var(--accent-blue);
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    padding: 3px 10px;
    border-radius: 4px;
    margin-right: 8px;
    letter-spacing: 0.08em;
}
 
/* Finding boxes */
.finding {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 14px;
}
.finding-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--accent-yellow);
    margin-bottom: 6px;
    letter-spacing: 0.08em;
}
 
/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────
# DATA LOADING — cached for performance
# ─────────────────────────────────────────────────────────────
 
TICKERS = [
    "AAPL","MSFT","GOOGL","NVDA","META","AMZN",
    "JPM","GS","BAC","MS",
    "XOM","CVX","COP",
    "JNJ","PFE","UNH","ABBV",
    "TSLA","WMT","BA"
]
 
SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology",
    "NVDA":"Technology","META":"Technology","AMZN":"Technology",
    "JPM":"Financials","GS":"Financials","BAC":"Financials","MS":"Financials",
    "XOM":"Energy","CVX":"Energy","COP":"Energy",
    "JNJ":"Healthcare","PFE":"Healthcare","UNH":"Healthcare","ABBV":"Healthcare",
    "TSLA":"Consumer","WMT":"Consumer","BA":"Consumer",
}
 
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    prices = yf.download(TICKERS, start="2018-01-01",
                         auto_adjust=True, progress=False)["Close"][TICKERS]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return prices, log_returns
 
@st.cache_data(ttl=3600, show_spinner=False)
def compute_stats(log_returns):
    stats = pd.DataFrame({
        "Ann. Return (%)": (log_returns.mean() * 252 * 100).round(2),
        "Ann. Vol (%)":    (log_returns.std() * np.sqrt(252) * 100).round(2),
        "Skewness":        log_returns.skew().round(3),
        "Kurtosis":        log_returns.kurt().round(3),
        "Worst Day (%)":   (log_returns.min() * 100).round(2),
        "Best Day (%)":    (log_returns.max() * 100).round(2),
        "Sector":          pd.Series(SECTOR_MAP),
    })
    return stats
 
@st.cache_data(ttl=3600, show_spinner=False)
def fit_gpd(log_returns):
    gpd_params = {}
    THRESHOLD_PCT = 0.05
    for ticker in log_returns.columns:
        returns   = log_returns[ticker].dropna().values
        losses    = -returns
        threshold = np.quantile(losses, 1 - THRESHOLD_PCT)
        exceedances = losses[losses > threshold] - threshold
        if len(exceedances) < 10:
            continue
        xi, loc, sigma = genpareto.fit(exceedances, floc=0)
        gpd_params[ticker] = {"xi": xi, "sigma": sigma, "threshold": threshold}
    return gpd_params
 
# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:11px;
                color:#3d9bff;letter-spacing:0.15em;
                text-transform:uppercase;margin-bottom:4px;'>
    Portfolio Risk ML
    </div>
    <div style='font-size:20px;font-weight:700;margin-bottom:24px;'>
    Navigation
    </div>
    """, unsafe_allow_html=True)
 
    page = st.radio("", [
        "🏠  Overview",
        "📈  Phase 1 — Foundation",
        "🕸️  Phase 2 — Spillover & Regimes",
        "⚡  Phase 2 — EVT Risk Engine",
        "🧠  Phase 3 — Sentiment & Causal",
        "🎯  Live Risk Monitor",
    ], label_visibility="collapsed")
 
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#6b7a99;font-family:IBM Plex Mono;'>
    DATA SOURCE<br>
    <span style='color:#e8eaf0;'>Yahoo Finance + FRED</span><br><br>
    UNIVERSE<br>
    <span style='color:#e8eaf0;'>20 stocks · 5 sectors</span><br><br>
    PERIOD<br>
    <span style='color:#e8eaf0;'>2018 → Present</span><br><br>
    MODEL<br>
    <span style='color:#e8eaf0;'>EVT + Neural Copula</span>
    </div>
    """, unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading market data..."):
    prices, log_returns = load_data()
    stats = compute_stats(log_returns)
    gpd_params = fit_gpd(log_returns)
 
port_ret = log_returns.mean(axis=1)
 
# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────
if "Overview" in page:
    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>ML-Based Portfolio Risk Modeling</div>
        <div class='hero-sub'>Return Forecasting · Monte Carlo · VaR/CVaR · Tail Risk</div>
        <span class='tag'>EVT</span>
        <span class='tag'>Neural Copula</span>
        <span class='tag'>HMM Regimes</span>
        <span class='tag'>FinBERT</span>
        <span class='tag'>Causal DAG</span>
        <span class='tag'>Diebold-Yilmaz</span>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card green'>
            <div class='metric-label'>Portfolio Universe</div>
            <div class='metric-value'>20</div>
            <div class='metric-sub'>stocks across 5 sectors</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        tci = 50.75
        st.markdown(f"""
        <div class='metric-card red'>
            <div class='metric-label'>Total Connectedness</div>
            <div class='metric-value'>{tci}%</div>
            <div class='metric-sub'>avg volatility from other stocks</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card yellow'>
            <div class='metric-label'>Gaussian VaR Gap</div>
            <div class='metric-value'>49%</div>
            <div class='metric-sub'>underestimate at 99% confidence</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Phases Complete</div>
            <div class='metric-value'>3 / 5</div>
            <div class='metric-sub'>Modules 7–13 done</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<div class='section-header'>Project Architecture</div>",
                unsafe_allow_html=True)
 
    phases = [
        ("✅ Phase 1", "Foundation", "Returns · Volatility · VaR/CVaR · Monte Carlo · Gaussian · Correlation", "#00ff88"),
        ("✅ Phase 2", "Data + Graph", "20-Stock Pipeline · Spillover Graph · HMM Regimes · EVT + Neural Copula", "#00ff88"),
        ("✅ Phase 3", "ML Models", "FinBERT Sentiment · Causal DAG · Intervention Engine", "#00ff88"),
        ("⏳ Phase 4", "Optimization", "CVaR-RL Agent · Meta-learner · Backtesting Suite", "#ffd32a"),
        ("⏳ Phase 5", "Deployment", "Streamlit Dashboard · Full Pipeline Integration", "#ffd32a"),
    ]
 
    for badge, title, desc, color in phases:
        st.markdown(f"""
        <div class='finding'>
            <div style='display:flex;align-items:center;gap:12px;'>
                <span style='font-family:IBM Plex Mono;font-size:13px;
                             color:{color};font-weight:600;min-width:90px;'>{badge}</span>
                <span style='font-weight:600;font-size:15px;min-width:130px;'>{title}</span>
                <span style='color:#6b7a99;font-size:13px;'>{desc}</span>
            </div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<div class='section-header'>Key Innovations</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='finding'>
            <div class='finding-title'>01 · CAUSAL INTERVENTION ENGINE</div>
            Uses do-calculus to answer what-if questions.
            "What happens to my portfolio if the Fed raises rates by 1%?"
            No standard risk model can answer this.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='finding'>
            <div class='finding-title'>02 · GNN-PARAMETERIZED COPULA</div>
            Replaces Gaussian assumption with a neural copula
            whose parameters are driven by the volatility
            spillover graph — dynamic tail dependence.
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='finding'>
            <div class='finding-title'>03 · FORWARD-LOOKING SENTIMENT</div>
            FinBERT extracts sentiment from news headlines
            and predicts VaR breaches before they appear
            in price data. AUC = 0.683.
        </div>""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────
# PAGE: PHASE 1 — FOUNDATION
# ─────────────────────────────────────────────────────────────
elif "Phase 1" in page:
    st.markdown("<div class='hero'><div class='hero-title'>Phase 1 — Foundation</div><div class='hero-sub'>Proving why standard risk models fail</div></div>",
                unsafe_allow_html=True)
 
    # Key findings
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card red'>
            <div class='metric-label'>Max Kurtosis (META)</div>
            <div class='metric-value'>22.4</div>
            <div class='metric-sub'>Gaussian assumes 3.0</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card red'>
            <div class='metric-label'>VaR Gap at 99%</div>
            <div class='metric-value'>0.95%</div>
            <div class='metric-sub'>Rs.95,000 per crore in missing reserves</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card yellow'>
            <div class='metric-label'>MSFT Vol Autocorr.</div>
            <div class='metric-value'>0.331</div>
            <div class='metric-sub'>Volatility clustering confirmed</div>
        </div>""", unsafe_allow_html=True)
 
    # Fat tails chart
    st.markdown("<div class='section-header'>Fat Tails — Kurtosis vs Gaussian Baseline</div>",
                unsafe_allow_html=True)
 
    stats_sorted = stats.sort_values("Kurtosis", ascending=True)
    colors = ["#ff4757" if k > 5 else "#3d9bff" if k > 2 else "#00ff88"
              for k in stats_sorted["Kurtosis"]]
 
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stats_sorted["Kurtosis"],
        y=stats_sorted.index,
        orientation="h",
        marker_color=colors,
        text=stats_sorted["Kurtosis"].round(1),
        textposition="outside",
        textfont=dict(color="#e8eaf0", size=11),
    ))
    fig.add_vline(x=3, line_dash="dash", line_color="#ffd32a", line_width=2,
                  annotation_text="Gaussian = 3.0",
                  annotation_font_color="#ffd32a")
    fig.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        xaxis=dict(gridcolor="#1e2d45", title="Excess Kurtosis"),
        yaxis=dict(gridcolor="#1e2d45"),
        height=500, margin=dict(l=20, r=80, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
 
    # VaR comparison
    st.markdown("<div class='section-header'>Historical vs Gaussian VaR — The Gap That Caused 2008</div>",
                unsafe_allow_html=True)
 
    conf_levels = [0.90, 0.95, 0.99]
    hist_var, gauss_var = [], []
    for p in conf_levels:
        hist_var.append(abs(np.percentile(port_ret, (1-p)*100)) * 100)
        mu, std = port_ret.mean(), port_ret.std()
        gauss_var.append(abs(mu + std * norm.ppf(1-p)) * 100)
 
    fig2 = go.Figure()
    x_labels = ["90%", "95%", "99%"]
    fig2.add_trace(go.Bar(name="Historical (Honest)", x=x_labels, y=hist_var,
                          marker_color="#ff4757", text=[f"{v:.2f}%" for v in hist_var],
                          textposition="outside", textfont=dict(color="#e8eaf0")))
    fig2.add_trace(go.Bar(name="Gaussian (Wrong)", x=x_labels, y=gauss_var,
                          marker_color="#3d9bff", text=[f"{v:.2f}%" for v in gauss_var],
                          textposition="outside", textfont=dict(color="#e8eaf0")))
    fig2.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        xaxis=dict(title="Confidence Level"),
        yaxis=dict(title="Daily VaR (%)", gridcolor="#1e2d45"),
        barmode="group", height=380,
        legend=dict(bgcolor="#1a2235", bordercolor="#1e2d45"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)
 
    # Correlation breakdown
    st.markdown("<div class='section-header'>Correlation Breakdown During Crisis — Diversification Fails When You Need It Most</div>",
                unsafe_allow_html=True)
 
    pairs     = ["AAPL vs MSFT", "AAPL vs GOOGL", "AAPL vs TSLA"]
    normal    = [0.65, 0.60, 0.45]
    crisis    = [0.944, 0.930, 0.910]
 
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Normal (2019)", x=pairs, y=normal,
                          marker_color="#00ff88"))
    fig3.add_trace(go.Bar(name="COVID Crisis (2020)", x=pairs, y=crisis,
                          marker_color="#ff4757"))
    fig3.add_hline(y=1.0, line_dash="dot", line_color="#ffd32a",
                   annotation_text="Perfect correlation = no diversification")
    fig3.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        yaxis=dict(title="Correlation", range=[0, 1.1], gridcolor="#1e2d45"),
        barmode="group", height=350, legend=dict(bgcolor="#1a2235"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig3, use_container_width=True)
 
# ─────────────────────────────────────────────────────────────
# PAGE: PHASE 2 — SPILLOVER & REGIMES
# ─────────────────────────────────────────────────────────────
elif "Spillover" in page:
    st.markdown("<div class='hero'><div class='hero-title'>Phase 2 — Spillover & Regimes</div><div class='hero-sub'>Who drives the market · When regimes shift</div></div>",
                unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card red'>
            <div class='metric-label'>Top Transmitter</div>
            <div class='metric-value'>JPM</div>
            <div class='metric-sub'>NET spillover +196.5</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Total Connectedness</div>
            <div class='metric-value'>50.75%</div>
            <div class='metric-sub'>avg shock from other stocks</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card yellow'>
            <div class='metric-label'>Crisis Duration</div>
            <div class='metric-value'>~19 days</div>
            <div class='metric-sub'>once started, persists 4 weeks</div>
        </div>""", unsafe_allow_html=True)
 
    # Spillover roles chart
    st.markdown("<div class='section-header'>Diebold-Yilmaz NET Spillover — Transmitters vs Receivers</div>",
                unsafe_allow_html=True)
 
    spillover_net = {
        "JPM":196.5,"COP":98.9,"AAPL":94.1,"JNJ":52.5,
        "META":-1.9,"MSFT":-2.1,"PFE":-16.7,"ABBV":-17.0,
        "TSLA":-18.4,"CVX":-19.0,"UNH":-21.6,"XOM":-21.6,
        "WMT":-22.9,"NVDA":-23.3,"GOOGL":-26.5,"AMZN":-34.0,
        "BAC":-44.6,"BA":-54.1,"GS":-54.2,"MS":-64.1
    }
 
    df_spill = pd.DataFrame(list(spillover_net.items()),
                             columns=["Ticker","NET"]).sort_values("NET")
    colors_spill = ["#00ff88" if v > 0 else "#ff4757" for v in df_spill["NET"]]
 
    fig4 = go.Figure(go.Bar(
        x=df_spill["NET"], y=df_spill["Ticker"],
        orientation="h", marker_color=colors_spill,
        text=df_spill["NET"].round(1),
        textposition="outside",
        textfont=dict(color="#e8eaf0", size=10),
    ))
    fig4.add_vline(x=0, line_color="#6b7a99", line_width=1)
    fig4.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        xaxis=dict(title="NET Spillover (TO − FROM)", gridcolor="#1e2d45"),
        yaxis=dict(gridcolor="#1e2d45"),
        height=550, margin=dict(l=20, r=80, t=20, b=20),
        annotations=[
            dict(x=150, y=18, text="📤 Transmitters", showarrow=False,
                 font=dict(color="#00ff88", size=12)),
            dict(x=-45, y=1, text="📥 Receivers", showarrow=False,
                 font=dict(color="#ff4757", size=12)),
        ]
    )
    st.plotly_chart(fig4, use_container_width=True)
 
    # HMM Regimes
    st.markdown("<div class='section-header'>HMM Market Regimes — 4 States Learned From Data</div>",
                unsafe_allow_html=True)
 
    regime_data = {
        "Regime"        : ["🟢 Bull","🔵 Recovery","🟡 Bear/Nervous","🔴 Crisis"],
        "Days"          : [573, 630, 615, 237],
        "Pct"           : [27.9, 30.7, 29.9, 11.5],
        "Avg Return"    : ["+0.12%","+0.07%","+0.07%","-0.15%"],
        "Avg VIX"       : [14.7, 17.3, 22.1, 33.5],
        "Avg Corr"      : [0.141, 0.273, 0.367, 0.565],
        "Duration (days)": [33.0, 33.9, 32.2, 19.2],
    }
    df_reg = pd.DataFrame(regime_data)
 
    col1, col2 = st.columns([1, 1])
    with col1:
        fig5 = go.Figure(go.Pie(
            labels=df_reg["Regime"],
            values=df_reg["Days"],
            hole=0.55,
            marker_colors=["#00ff88","#3d9bff","#ffd32a","#ff4757"],
            textfont=dict(size=12, color="#e8eaf0"),
        ))
        fig5.update_layout(
            plot_bgcolor="#111827", paper_bgcolor="#111827",
            font=dict(color="#e8eaf0"),
            height=320, margin=dict(l=0,r=0,t=20,b=0),
            showlegend=True,
            legend=dict(bgcolor="#1a2235", bordercolor="#1e2d45"),
            annotations=[dict(text="2073<br>days", x=0.5, y=0.5,
                              font_size=16, showarrow=False,
                              font_color="#e8eaf0")]
        )
        st.plotly_chart(fig5, use_container_width=True)
 
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in df_reg.iterrows():
            color = {"🟢":"#00ff88","🔵":"#3d9bff","🟡":"#ffd32a","🔴":"#ff4757"}
            c = next(v for k,v in color.items() if k in row["Regime"])
            st.markdown(f"""
            <div style='background:#1a2235;border:1px solid #1e2d45;
                        border-left:3px solid {c};border-radius:6px;
                        padding:10px 16px;margin:6px 0;'>
                <span style='font-weight:600;'>{row["Regime"]}</span>
                <span style='color:#6b7a99;font-size:12px;margin-left:12px;'>
                {row["Pct"]}% of days · VIX={row["Avg VIX"]} · Corr={row["Avg Corr"]} · ~{row["Duration (days)"]} day duration
                </span>
            </div>""", unsafe_allow_html=True)
 
    # Key insight
    st.markdown("""
    <div class='finding' style='margin-top:16px;'>
        <div class='finding-title'>KEY FINDING — CRISIS NEVER JUMPS TO BULL</div>
        Transition matrix shows Crisis → Bull = 0.000. Markets never jump directly from crash to bull.
        The path is always Crisis → Bear → Recovery → Bull. HMM learned this purely from data with zero human labeling.
        Crisis diagonal = 0.948 — once a crash starts, 94.8% chance it continues the next day.
    </div>""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────
# PAGE: PHASE 2 — EVT RISK ENGINE
# ─────────────────────────────────────────────────────────────
elif "EVT" in page:
    st.markdown("<div class='hero'><div class='hero-title'>Phase 2 — EVT Risk Engine</div><div class='hero-sub'>Extreme Value Theory + Neural Copula Monte Carlo</div></div>",
                unsafe_allow_html=True)
 
    PORTFOLIO_VAL = 1_000_000
 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card green'>
            <div class='metric-label'>EVT 95% VaR</div>
            <div class='metric-value'>2.52%</div>
            <div class='metric-sub'>Rs.25,184 on Rs.10L</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card red'>
            <div class='metric-label'>EVT 99% VaR</div>
            <div class='metric-value'>4.65%</div>
            <div class='metric-sub'>Rs.46,457 on Rs.10L</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card red'>
            <div class='metric-label'>Gaussian 99% VaR</div>
            <div class='metric-value'>3.11%</div>
            <div class='metric-sub'>Rs.31,113 — underestimates</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card yellow'>
            <div class='metric-label'>Hidden Risk</div>
            <div class='metric-value'>Rs.15,344</div>
            <div class='metric-sub'>per day per Rs.10L portfolio</div>
        </div>""", unsafe_allow_html=True)
 
    # GPD Xi chart
    st.markdown("<div class='section-header'>GPD Shape Parameter (ξ) — Tail Heaviness Per Stock</div>",
                unsafe_allow_html=True)
 
    gpd_df = pd.DataFrame([
        {"Ticker":t, "Xi":p["xi"], "Threshold":p["threshold"]}
        for t,p in gpd_params.items()
    ]).sort_values("Xi", ascending=True)
 
    colors_gpd = ["#ff4757" if x > 0.3 else "#ffd32a" if x > 0.15 else "#00ff88"
                  for x in gpd_df["Xi"]]
 
    fig6 = go.Figure(go.Bar(
        x=gpd_df["Xi"], y=gpd_df["Ticker"],
        orientation="h", marker_color=colors_gpd,
        text=gpd_df["Xi"].round(3),
        textposition="outside",
        textfont=dict(color="#e8eaf0", size=10),
    ))
    fig6.add_vline(x=0.3, line_dash="dash", line_color="#ff4757",
                   annotation_text="Heavy tail threshold (ξ=0.3)",
                   annotation_font_color="#ff4757")
    fig6.add_vline(x=0, line_dash="dash", line_color="#6b7a99", line_width=1)
    fig6.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        xaxis=dict(title="ξ (Xi) — Shape Parameter", gridcolor="#1e2d45"),
        height=520, margin=dict(l=20, r=80, t=20, b=20),
    )
    st.plotly_chart(fig6, use_container_width=True)
 
    # Risk comparison table
    st.markdown("<div class='section-header'>EVT+Copula vs Gaussian — Full Risk Comparison</div>",
                unsafe_allow_html=True)
 
    risk_table = pd.DataFrame({
        "Method"       : ["EVT+Copula MC", "EVT+Copula MC", "Gaussian MC", "Gaussian MC"],
        "Confidence"   : ["95%","99%","95%","99%"],
        "VaR (Rs.)"    : [-25184,-46457,-21820,-31113],
        "CVaR (Rs.)"   : [-37727,-51810,-27518,-35734],
        "VaR (%)"      : [-2.518,-4.646,2.182,3.111],
        "CVaR (%)"     : [-3.773,-5.181,2.752,3.573],
    })
 
    fig7 = go.Figure()
    methods = ["EVT+Copula", "Gaussian"]
    confs   = ["95%","99%"]
    colors_m = ["#ff4757","#3d9bff"]
 
    for i, (method, color) in enumerate(zip(["EVT+Copula MC","Gaussian MC"], colors_m)):
        subset = risk_table[risk_table["Method"]==method]
        fig7.add_trace(go.Bar(
            name=method,
            x=subset["Confidence"],
            y=subset["VaR (Rs.)"].abs(),
            marker_color=color,
            text=["Rs.{:,.0f}".format(v) for v in subset["VaR (Rs.)"].abs()],
            textposition="outside",
            textfont=dict(color="#e8eaf0"),
        ))
 
    fig7.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        yaxis=dict(title="Daily VaR (Rs.)", gridcolor="#1e2d45"),
        barmode="group", height=380,
        legend=dict(bgcolor="#1a2235", bordercolor="#1e2d45"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig7, use_container_width=True)
 
    st.markdown("""
    <div class='finding'>
        <div class='finding-title'>THE 2008 LESSON</div>
        At 99% confidence, Gaussian underestimates true risk by Rs.15,344 on a Rs.10,00,000 portfolio.
        Every bank that used Gaussian VaR in 2008 held insufficient reserves.
        UNH (ξ=0.538), WMT (ξ=0.571), META (ξ=0.457) — Gaussian is dangerously wrong for these stocks.
    </div>""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────
# PAGE: PHASE 3 — SENTIMENT & CAUSAL
# ─────────────────────────────────────────────────────────────
elif "Sentiment" in page:
    st.markdown("<div class='hero'><div class='hero-title'>Phase 3 — Sentiment & Causal</div><div class='hero-sub'>Forward-looking risk · What-if scenario engine</div></div>",
                unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card green'>
            <div class='metric-label'>FinBERT AUC</div>
            <div class='metric-value'>0.683</div>
            <div class='metric-sub'>VaR breach predictor</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card yellow'>
            <div class='metric-label'>Sentiment-Adj CVaR</div>
            <div class='metric-value'>Rs.59,922</div>
            <div class='metric-sub'>vs base Rs.51,810</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Oil -20% VaR Impact</div>
            <div class='metric-value'>Rs.-894</div>
            <div class='metric-sub'>causal, not correlation</div>
        </div>""", unsafe_allow_html=True)
 
    # Feature importance
    st.markdown("<div class='section-header'>VaR Breach Predictor — Feature Importance (LightGBM)</div>",
                unsafe_allow_html=True)
 
    feat_imp = pd.DataFrame({
        "Feature"   : ["vol_20d","rolling_corr","momentum_10d","vol_spike",
                       "vol_5d","return_5d","return_1d","neg_sentiment_proxy"],
        "Importance": [367.6, 338.2, 233.2, 217.8, 194.2, 162.6, 162.4, 41.8]
    })
 
    fig8 = go.Figure(go.Bar(
        x=feat_imp["Importance"],
        y=feat_imp["Feature"],
        orientation="h",
        marker_color=["#3d9bff" if f != "neg_sentiment_proxy" else "#ffd32a"
                      for f in feat_imp["Feature"]],
        text=feat_imp["Importance"].round(1),
        textposition="outside",
        textfont=dict(color="#e8eaf0"),
    ))
    fig8.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        xaxis=dict(title="Feature Importance", gridcolor="#1e2d45"),
        height=350, margin=dict(l=20, r=80, t=20, b=20),
    )
    st.plotly_chart(fig8, use_container_width=True)
 
    st.markdown("""
    <div class='finding'>
        <div class='finding-title'>WHY SENTIMENT RANKS LAST</div>
        neg_sentiment_proxy uses return-based proxies, not real FinBERT scores.
        With 2+ years of actual daily FinBERT scoring, sentiment would likely rank #1 —
        it's a leading indicator while volatility metrics are lagging.
    </div>""", unsafe_allow_html=True)
 
    # Causal DAG
    st.markdown("<div class='section-header'>Causal DAG — Theory-Informed Structure</div>",
                unsafe_allow_html=True)
 
    causal_edges = [
        ("INFLATION","FED_RATE"),("FED_RATE","YIELD_CURVE"),
        ("FED_RATE","VIX"),("OIL_RET","INFLATION"),
        ("VIX","PORT_RET"),("YIELD_CURVE","PORT_RET"),("OIL_RET","PORT_RET"),
    ]
 
    G_vis = go.Figure()
    pos = {
        "OIL_RET"   : (0.0, 0.5),
        "INFLATION" : (0.25, 0.8),
        "FED_RATE"  : (0.5, 0.8),
        "YIELD_CURVE": (0.75, 0.65),
        "VIX"       : (0.75, 0.35),
        "PORT_RET"  : (1.0, 0.5),
    }
 
    node_colors_dag = {
        "FED_RATE":"#e74c3c","YIELD_CURVE":"#e67e22",
        "VIX":"#9b59b6","INFLATION":"#f39c12",
        "OIL_RET":"#1abc9c","PORT_RET":"#3d9bff",
    }
 
    for src, tgt in causal_edges:
        x0,y0 = pos[src]
        x1,y1 = pos[tgt]
        G_vis.add_trace(go.Scatter(
            x=[x0,x1,None], y=[y0,y1,None],
            mode="lines",
            line=dict(color="#6b7a99", width=2),
            showlegend=False, hoverinfo="none"
        ))
 
    for node, (x,y) in pos.items():
        G_vis.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=40, color=node_colors_dag[node], opacity=0.9),
            text=[node], textposition="top center",
            textfont=dict(size=11, color="#e8eaf0"),
            showlegend=False,
            hovertemplate=f"<b>{node}</b><extra></extra>"
        ))
 
    G_vis.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        height=380,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20,r=20,t=20,b=20),
    )
    st.plotly_chart(G_vis, use_container_width=True)
 
    # Intervention results
    st.markdown("<div class='section-header'>Counterfactual Intervention Results</div>",
                unsafe_allow_html=True)
 
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='finding'>
            <div class='finding-title'>INTERVENTION 1 — FED +100BPS</div>
            Causal effect on portfolio: near-zero daily impact on
            equal-weight 20-stock portfolio. The rate shock gets
            diluted across sectors. Concentrated financial portfolios
            would show a much larger effect.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='finding'>
            <div class='finding-title'>INTERVENTION 2 — OIL -20%</div>
            VaR changes by Rs.-894 on a Rs.10L portfolio.
            Direct channel: XOM, CVX, COP = 15% of portfolio.
            Indirect channel: OIL → INFLATION → FED_RATE → VIX → PORT_RET.
            Both channels captured by the causal model.
        </div>""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────
# PAGE: LIVE RISK MONITOR
# ─────────────────────────────────────────────────────────────
elif "Live" in page:
    st.markdown("<div class='hero'><div class='hero-title'>Live Risk Monitor</div><div class='hero-sub'>Real-time portfolio risk metrics</div></div>",
                unsafe_allow_html=True)
 
    # Portfolio selector
    st.markdown("<div class='section-header'>Portfolio Configuration</div>",
                unsafe_allow_html=True)
 
    col1, col2 = st.columns([2,1])
    with col1:
        selected = st.multiselect(
            "Select stocks",
            TICKERS,
            default=["AAPL","MSFT","GOOGL","NVDA","JPM"]
        )
    with col2:
        portfolio_val = st.number_input(
            "Portfolio Value (Rs.)",
            min_value=100000,
            max_value=100000000,
            value=1000000,
            step=100000,
            format="%d"
        )
 
    if not selected:
        st.warning("Select at least 2 stocks.")
        st.stop()
 
    conf = st.select_slider(
        "Confidence Level",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%"
    )
 
    # Compute live metrics
    ret_sel    = log_returns[selected].dropna()
    port_r     = ret_sel.mean(axis=1)
    hist_var   = np.percentile(port_r, (1-conf)*100)
    hist_cvar  = port_r[port_r <= hist_var].mean()
    mu, std    = port_r.mean(), port_r.std()
    gauss_var  = mu + std * norm.ppf(1-conf)
    gauss_cvar = mu - std * norm.pdf(norm.ppf(1-conf))/(1-conf)
 
    var_rs   = hist_var  * portfolio_val
    cvar_rs  = hist_cvar * portfolio_val
    g_var_rs = gauss_var  * portfolio_val
    gap_rs   = var_rs - g_var_rs
 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card red'>
            <div class='metric-label'>Historical VaR ({int(conf*100)}%)</div>
            <div class='metric-value'>Rs.{abs(var_rs):,.0f}</div>
            <div class='metric-sub'>{hist_var*100:.2f}% of portfolio</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card red'>
            <div class='metric-label'>Historical CVaR ({int(conf*100)}%)</div>
            <div class='metric-value'>Rs.{abs(cvar_rs):,.0f}</div>
            <div class='metric-sub'>{hist_cvar*100:.2f}% of portfolio</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Gaussian VaR ({int(conf*100)}%)</div>
            <div class='metric-value'>Rs.{abs(g_var_rs):,.0f}</div>
            <div class='metric-sub'>Underestimates true risk</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        color = "red" if gap_rs < -1000 else "yellow"
        st.markdown(f"""
        <div class='metric-card {color}'>
            <div class='metric-label'>Hidden Risk Gap</div>
            <div class='metric-value'>Rs.{abs(gap_rs):,.0f}</div>
            <div class='metric-sub'>Gaussian misses this daily</div>
        </div>""", unsafe_allow_html=True)
 
    # Return distribution
    st.markdown("<div class='section-header'>Portfolio Return Distribution</div>",
                unsafe_allow_html=True)
 
    fig9 = go.Figure()
    fig9.add_trace(go.Histogram(
        x=port_r * 100, nbinsx=80,
        marker_color="#3d9bff", opacity=0.7,
        name="Actual Returns", histnorm="probability density"
    ))
 
    x_range = np.linspace(port_r.min()*100, port_r.max()*100, 300)
    gauss_density = norm.pdf(x_range, mu*100, std*100)
    fig9.add_trace(go.Scatter(
        x=x_range, y=gauss_density,
        mode="lines", line=dict(color="#ffd32a", width=2, dash="dash"),
        name="Gaussian Fit"
    ))
 
    fig9.add_vline(x=hist_var*100, line_color="#ff4757", line_width=2,
                   annotation_text=f"VaR {int(conf*100)}%",
                   annotation_font_color="#ff4757")
 
    fig9.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        xaxis=dict(title="Daily Return (%)", gridcolor="#1e2d45"),
        yaxis=dict(title="Density", gridcolor="#1e2d45"),
        height=380, legend=dict(bgcolor="#1a2235"),
        margin=dict(l=20,r=20,t=20,b=20),
    )
    st.plotly_chart(fig9, use_container_width=True)
 
    # Correlation heatmap
    st.markdown("<div class='section-header'>Correlation Matrix</div>",
                unsafe_allow_html=True)
 
    corr = ret_sel.corr()
    fig10 = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Correlation"),
    ))
    fig10.update_layout(
        plot_bgcolor="#111827", paper_bgcolor="#111827",
        font=dict(color="#e8eaf0", family="IBM Plex Sans"),
        height=420, margin=dict(l=20,r=20,t=20,b=20),
    )
    st.plotly_chart(fig10, use_container_width=True)
