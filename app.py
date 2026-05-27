import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from model import fetch_data, add_features, train_model
from datetime import datetime

TICKER_MAP = {
    "apple": "AAPL", "google": "GOOGL", "alphabet": "GOOGL",
    "microsoft": "MSFT", "amazon": "AMZN", "tesla": "TSLA",
    "nvidia": "NVDA", "meta": "META", "facebook": "META",
    "netflix": "NFLX", "spotify": "SPOT", "intel": "INTC",
    "amd": "AMD", "disney": "DIS", "coca cola": "KO",
    "nike": "NKE", "toyota": "TM", "walmart": "WMT",
    "jpmorgan": "JPM", "uber": "UBER", "adobe": "ADBE",
    "paypal": "PYPL", "airbnb": "ABNB", "goldman sachs": "GS",
}

def resolve_ticker(user_input):
    cleaned = user_input.strip().lower()
    return TICKER_MAP.get(cleaned, user_input.strip().upper())

st.set_page_config(page_title="StockSense", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0B0F1A !important;
    color: #E2E8F0 !important;
}
.stApp, .main { background: #0B0F1A !important; }

.stTextInput > div > div > input {
    background: #141824 !important;
    border: 1px solid #1E2535 !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 14px 18px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #4F8EF7 !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #3A4560 !important; }

.stButton > button {
    background: #4F8EF7 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 14px 32px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #3B7AE8 !important; }

.stExpander {
    background: #141824 !important;
    border: 1px solid #1E2535 !important;
    border-radius: 10px !important;
}

div[data-testid="stHorizontalBlock"] { gap: 12px !important; }

.mcard {
    background: #141824;
    border: 1px solid #1E2535;
    border-radius: 10px;
    padding: 16px 20px;
}
.mcard.blue  { border-top: 2px solid #4F8EF7; }
.mcard.green { border-top: 2px solid #22C55E; }
.mcard.red   { border-top: 2px solid #EF4444; }
.mcard.amber { border-top: 2px solid #F59E0B; }

.mc-label { font-size: 11px; color: #6B7A99; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
.mc-val   { font-size: 26px; font-weight: 700; color: #fff; }
.mc-val.blue  { color: #4F8EF7; }
.mc-val.green { color: #22C55E; }
.mc-val.red   { color: #EF4444; }
.mc-val.amber { color: #F59E0B; }
.mc-sub   { font-size: 11px; color: #6B7A99; margin-top: 6px; }
.mc-badge { display: inline-block; font-size: 11px; padding: 3px 10px; border-radius: 20px; margin-top: 8px; font-weight: 500; }
.mc-badge.up   { background: #052010; color: #22C55E; }
.mc-badge.down { background: #1A0505; color: #EF4444; }

.section-hdr {
    font-size: 11px;
    color: #6B7A99;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #1E2535;
    padding-bottom: 10px;
    margin: 28px 0 16px;
}

.signal-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #0F1520;
    font-size: 13px;
}
.sig-label { color: #6B7A99; }
.sig-val   { font-weight: 600; color: #fff; }
.sig-val.green { color: #22C55E; }
.sig-val.red   { color: #EF4444; }
.sig-val.amber { color: #F59E0B; }

.pill { display: inline-block; font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 600; }
.pill.bullish { background: #052010; color: #22C55E; }
.pill.bearish { background: #1A0505; color: #EF4444; }
.pill.neutral { background: #1A1200; color: #F59E0B; }

.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.chip { background: #141824; border: 1px solid #1E2535; color: #6B7A99; font-size: 12px; padding: 5px 14px; border-radius: 20px; }

.disclaimer {
    font-size: 11px;
    color: #3A4560;
    text-align: center;
    padding: 20px;
    border-top: 1px solid #1E2535;
    margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)

# ── Navbar ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
     border-bottom:1px solid #1E2535;padding-bottom:16px;margin-bottom:28px;">
  <div style="font-size:20px;font-weight:700;color:#fff;">
    Stock<span style="color:#4F8EF7;">Sense</span>
  </div>
  <div style="display:flex;gap:28px;font-size:13px;color:#6B7A99;">
    <span style="color:#fff;font-weight:500;">Dashboard</span>
    <span>Markets</span>
    <span>About</span>
  </div>
  <div style="background:#141824;border:1px solid #1E2535;color:#4F8EF7;
       font-size:11px;padding:5px 14px;border-radius:20px;">
    AI Powered
  </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px;">
  <div style="font-size:11px;color:#4F8EF7;letter-spacing:2px;
       text-transform:uppercase;margin-bottom:8px;">
    Real-time AI Stock Analysis
  </div>
  <div style="font-size:28px;font-weight:700;color:#fff;line-height:1.3;margin-bottom:6px;">
    Predict smarter. <span style="color:#4F8EF7;">Invest better.</span>
  </div>
  <div style="font-size:13px;color:#6B7A99;">
    Enter any company name or ticker symbol to get AI-powered price prediction and trend signals.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Search ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input(
        "",
        placeholder="Search company or ticker — Apple, TSLA, Nvidia, Google...",
        label_visibility="collapsed"
    )
with col2:
    predict_btn = st.button("Analyse →", use_container_width=True)

st.markdown("""
<div class="chip-row">
  <div class="chip">Apple</div><div class="chip">Google</div>
  <div class="chip">Tesla</div><div class="chip">Nvidia</div>
  <div class="chip">Microsoft</div><div class="chip">Amazon</div>
  <div class="chip">Meta</div><div class="chip">Netflix</div>
</div>
""", unsafe_allow_html=True)

# ── Results ─────────────────────────────────────────────────────────────────
if predict_btn:
    if not user_input.strip():
        st.warning("Please enter a company name or ticker symbol.")
    else:
        ticker = resolve_ticker(user_input)

        with st.spinner(f"Fetching market data for {ticker}..."):
            df = fetch_data(ticker)

        if df.empty:
            st.error(f"No data found for '{user_input}'. Try the ticker directly e.g. AAPL for Apple.")
        else:
            with st.spinner("Training model and generating signals..."):
                df = add_features(df)
                model, X_test, y_test, predictions, mae, r2 = train_model(df)

            latest_price   = float(df['Close'].iloc[-1])
            predicted_next = float(predictions[-1])
            price_change   = predicted_next - latest_price
            pct_change     = (price_change / latest_price) * 100
            ma7            = float(df['MA7'].iloc[-1])
            ma21           = float(df['MA21'].iloc[-1])
            ma50           = float(df['MA50'].iloc[-1])
            volatility     = float(df['Volatility'].iloc[-1]) * 100
            daily_return   = float(df['Return'].iloc[-1]) * 100
            trend          = "Bullish" if ma7 > ma21 else "Bearish"
            trend_pill     = "bullish" if ma7 > ma21 else "bearish"
            momentum       = "Strong" if abs(pct_change) > 1 else "Neutral"
            mom_pill       = "bullish" if abs(pct_change) > 1 else "neutral"
            dir_badge      = "up" if price_change >= 0 else "down"
            dir_arrow      = "▲" if price_change >= 0 else "▼"
            val_cls        = "green" if price_change >= 0 else "red"
            ret_cls        = "green" if daily_return >= 0 else "red"

            # ── Metric cards ──────────────────────────────────────────────
            st.markdown(
                f'<p class="section-hdr">{ticker} · Analysis Results</p>',
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="mcard blue">
                  <div class="mc-label">Current Price</div>
                  <div class="mc-val blue">${latest_price:.2f}</div>
                  <div class="mc-sub">{ticker} · Last Close</div>
                  <div class="mc-badge up">● Live</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="mcard {val_cls}">
                  <div class="mc-label">Predicted Next Day</div>
                  <div class="mc-val {val_cls}">${predicted_next:.2f}</div>
                  <div class="mc-sub">Expected move</div>
                  <div class="mc-badge {dir_badge}">{dir_arrow} {abs(pct_change):.2f}%</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="mcard amber">
                  <div class="mc-label">Model Accuracy R²</div>
                  <div class="mc-val amber">{r2:.3f}</div>
                  <div class="mc-sub">Ridge Regression · Test Set</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="mcard green">
                  <div class="mc-label">Mean Abs Error</div>
                  <div class="mc-val green">${mae:.2f}</div>
                  <div class="mc-sub">Avg prediction deviation</div>
                </div>""", unsafe_allow_html=True)

            # ── Two column layout ─────────────────────────────────────────
            left, right = st.columns([2, 1])

            with left:
                # Chart 1
                st.markdown('<p class="section-hdr">Actual vs Predicted Price</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#141824')
                ax.set_facecolor('#141824')
                ax.plot(y_test.values, color='#4F8EF7', linewidth=2,   label='Actual')
                ax.plot(predictions,   color='#EF4444', linewidth=1.5, label='Predicted', linestyle='--')
                ax.fill_between(range(len(y_test)), y_test.values, predictions,
                                alpha=0.05, color='#4F8EF7')
                ax.set_xlabel('Days',        color='#6B7A99', fontsize=10)
                ax.set_ylabel('Price (USD)', color='#6B7A99', fontsize=10)
                ax.tick_params(colors='#6B7A99', labelsize=9)
                for spine in ax.spines.values():
                    spine.set_color('#1E2535')
                ax.grid(True, color='#1E2535', linewidth=0.6, alpha=0.8)
                ax.legend(facecolor='#141824', edgecolor='#1E2535',
                          labelcolor='#E2E8F0', fontsize=10)
                st.pyplot(fig)

                # Chart 2
                st.markdown('<p class="section-hdr">Moving Average Bands</p>', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                fig2.patch.set_facecolor('#141824')
                ax2.set_facecolor('#141824')
                ax2.plot(df['Close'].values, color='#E2E8F044', linewidth=1,   label='Close')
                ax2.plot(df['MA7'].values,   color='#4F8EF7',   linewidth=2,   label='MA 7D')
                ax2.plot(df['MA21'].values,  color='#F59E0B',   linewidth=2,   label='MA 21D')
                ax2.plot(df['MA50'].values,  color='#EF4444',   linewidth=1.5, label='MA 50D')
                ax2.tick_params(colors='#6B7A99', labelsize=9)
                for spine in ax2.spines.values():
                    spine.set_color('#1E2535')
                ax2.grid(True, color='#1E2535', linewidth=0.6, alpha=0.8)
                ax2.legend(facecolor='#141824', edgecolor='#1E2535',
                           labelcolor='#E2E8F0', fontsize=10)
                st.pyplot(fig2)

                with st.expander("View raw data"):
                    st.dataframe(df.tail(20), use_container_width=True)

            with right:
                # Signals panel
                st.markdown('<p class="section-hdr">Model Signals</p>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="signal-row">
                  <span class="sig-label">Trend</span>
                  <span class="pill {trend_pill}">{trend}</span>
                </div>
                <div class="signal-row">
                  <span class="sig-label">Momentum</span>
                  <span class="pill {mom_pill}">{momentum}</span>
                </div>
                <div class="signal-row">
                  <span class="sig-label">Direction</span>
                  <span class="sig-val {val_cls}">{dir_arrow} {abs(pct_change):.2f}%</span>
                </div>
                <div class="signal-row">
                  <span class="sig-label">Daily Return</span>
                  <span class="sig-val {ret_cls}">{daily_return:+.2f}%</span>
                </div>
                <div class="signal-row">
                  <span class="sig-label">Volatility (7D)</span>
                  <span class="sig-val amber">{volatility:.2f}%</span>
                </div>
                <div class="signal-row">
                  <span class="sig-label">Data Points</span>
                  <span class="sig-val">{len(df)}</span>
                </div>
                <div class="signal-row">
                  <span class="sig-label">History</span>
                  <span class="sig-val">2 Years</span>
                </div>
                """, unsafe_allow_html=True)

                # Moving averages
                st.markdown('<p class="section-hdr">Moving Averages</p>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="signal-row">
                  <span class="sig-label">MA 7-day</span>
                  <span class="sig-val">${ma7:.2f}</span>
                </div>
                <div class="signal-row">
                  <span class="sig-label">MA 21-day</span>
                  <span class="sig-val">${ma21:.2f}</span>
                </div>
                <div class="signal-row">
                  <span class="sig-label">MA 50-day</span>
                  <span class="sig-val">${ma50:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

            # ── Disclaimer ────────────────────────────────────────────────
            st.markdown("""
            <div class="disclaimer">
              Not financial advice · Educational use only · Data sourced from Yahoo Finance via yfinance
            </div>
            """, unsafe_allow_html=True)