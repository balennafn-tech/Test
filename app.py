"""
Dip Screener — หาหุ้นที่ราคาน่าช้อน (US + Thai/SET)
ออกแบบให้ใช้งานง่ายทั้งบนมือถือและคอมพิวเตอร์

รันด้วย: streamlit run app.py

ข้อมูลราคาดึงจาก Yahoo Finance ผ่านไลบรารี yfinance (ฟรี, ไม่ต้องมี API key)
ไม่ใช่ real-time แบบ tick-by-tick เหมือนแอปโบรกเกอร์ แต่โดยทั่วไปหน่วง
ไม่กี่นาทีถึงราว 15 นาที เพียงพอสำหรับใช้สแกนหาโอกาส ไม่เหมาะสำหรับ day trading
ที่ต้องการราคาแบบ tick จริง
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Dip Screener", page_icon="📉", layout="centered")

# ---------------------------------------------------------------------------
# Mobile-friendly styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 760px; }
    div.stButton > button {
        height: 3em; font-size: 1.05em; font-weight: 600; border-radius: 10px;
    }
    div[data-testid="stMetric"] {
        background: rgba(120,120,120,0.08); border-radius: 12px; padding: 10px 14px;
    }
    .card {
        border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;
        background: rgba(120,120,120,0.06); border: 1px solid rgba(120,120,120,0.15);
    }
    .card-top { display: flex; justify-content: space-between; align-items: center; }
    .ticker-name { font-size: 1.15em; font-weight: 700; }
    .price-text { font-size: 0.95em; opacity: 0.8; }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.85em; font-weight: 700;
    }
    .badge-high { background: #1e7e34; color: white; }
    .badge-mid { background: #b58105; color: white; }
    .badge-low { background: #6c757d; color: white; }
    .metric-row { font-size: 0.85em; opacity: 0.75; margin-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Default watchlists (แก้ไข/เพิ่มเติมได้ในช่องตั้งค่าขั้นสูง)
# ---------------------------------------------------------------------------

DEFAULT_US = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM", "CVX", "KO",
    "PEP", "ABBV", "MRK", "COST", "AVGO", "ADBE", "CRM", "NFLX", "AMD", "INTC",
    "CSCO", "PFE", "TMO", "ABT", "ACN", "NKE", "TXN", "MCD", "QCOM", "IBM",
]

DEFAULT_TH = [
    "PTT.BK", "PTTEP.BK", "CPALL.BK", "AOT.BK", "ADVANC.BK", "SCB.BK", "KBANK.BK",
    "BBL.BK", "SCC.BK", "CPF.BK", "TRUE.BK", "TOP.BK", "IVL.BK", "BDMS.BK",
    "BH.BK", "CENTEL.BK", "MINT.BK", "HMPRO.BK", "CRC.BK", "GULF.BK", "GPSC.BK",
    "EGCO.BK", "RATCH.BK", "BGRIM.BK", "TISCO.BK", "TTB.BK", "KTB.BK", "KKP.BK",
    "OSP.BK", "CBG.BK", "TU.BK", "SAWAD.BK", "MTC.BK", "JMT.BK", "AWC.BK",
    "LH.BK", "SPALI.BK", "AMATA.BK", "WHA.BK", "BTS.BK", "BEM.BK", "DELTA.BK",
    "KCE.BK", "HANA.BK", "SIRI.BK", "SCGP.BK", "COM7.BK",
]
# หมายเหตุ: รายชื่อ SET เป็นชุดหุ้นขนาดใหญ่ที่คุ้นเคย ไม่ใช่ SET50 อย่างเป๊ะ ณ ปัจจุบัน
# ตรวจ/ปรับรายชื่อจริงได้ที่เว็บ SET ก่อนใช้งานจริง

# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else np.nan


@st.cache_data(ttl=300, show_spinner=False)
def analyze_ticker(ticker: str) -> dict | None:
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
    except Exception:
        return None

    if hist is None or hist.empty or len(hist) < 60:
        return None

    close = hist["Close"].dropna()
    last_price = float(close.iloc[-1])
    high_52w = float(close.max())
    low_52w = float(close.min())

    pct_off_high = (high_52w - last_price) / high_52w * 100 if high_52w else np.nan

    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan

    pct_vs_ma50 = (last_price - ma50) / ma50 * 100 if pd.notna(ma50) else np.nan
    pct_vs_ma200 = (last_price - ma200) / ma200 * 100 if pd.notna(ma200) else np.nan

    rsi = compute_rsi(close)

    return {
        "Ticker": ticker,
        "ราคาล่าสุด": round(last_price, 2),
        "52wHigh": round(high_52w, 2),
        "52wLow": round(low_52w, 2),
        "%ต่ำกว่าจุดสูงสุด": round(pct_off_high, 1) if pd.notna(pct_off_high) else np.nan,
        "RSI(14)": round(rsi, 1) if pd.notna(rsi) else np.nan,
        "%เทียบMA50": round(pct_vs_ma50, 1) if pd.notna(pct_vs_ma50) else np.nan,
        "%เทียบMA200": round(pct_vs_ma200, 1) if pd.notna(pct_vs_ma200) else np.nan,
    }


def normalize_score(row: pd.Series, weights: dict) -> float:
    score = 0.0
    total_weight = 0.0

    if pd.notna(row["%ต่ำกว่าจุดสูงสุด"]) and weights["off_high"] > 0:
        s = min(row["%ต่ำกว่าจุดสูงสุด"] / 50 * 100, 100)
        s = max(s, 0)
        score += s * weights["off_high"]
        total_weight += weights["off_high"]

    if pd.notna(row["RSI(14)"]) and weights["rsi"] > 0:
        s = np.interp(row["RSI(14)"], [20, 30, 50, 70], [100, 100, 40, 0])
        score += s * weights["rsi"]
        total_weight += weights["rsi"]

    if pd.notna(row["%เทียบMA50"]) and weights["ma50"] > 0:
        s = np.interp(row["%เทียบMA50"], [-20, 0, 10], [100, 30, 0])
        score += s * weights["ma50"]
        total_weight += weights["ma50"]

    if pd.notna(row["%เทียบMA200"]) and weights["ma200"] > 0:
        s = np.interp(row["%เทียบMA200"], [-20, 0, 15], [100, 30, 0])
        score += s * weights["ma200"]
        total_weight += weights["ma200"]

    return round(score / total_weight, 1) if total_weight > 0 else np.nan


def score_badge(score: float) -> str:
    if pd.isna(score):
        return '<span class="badge badge-low">ไม่มีข้อมูล</span>'
    if score >= 70:
        return f'<span class="badge badge-high">🟢 {score:.0f} น่าสนใจมาก</span>'
    if score >= 40:
        return f'<span class="badge badge-mid">🟡 {score:.0f} ปานกลาง</span>'
    return f'<span class="badge badge-low">⚪ {score:.0f} ยังไม่เข้าเกณฑ์</span>'


DEFAULT_WEIGHTS = {"off_high": 1.0, "rsi": 1.0, "ma50": 1.0, "ma200": 0.5}


def verdict_text(score: float) -> tuple[str, str]:
    if pd.isna(score):
        return "ไม่มีข้อมูลเพียงพอสำหรับประเมิน", "⚪"
    if score >= 70:
        return "น่าช้อน — เข้าเกณฑ์หลายอย่างพร้อมกัน ราคาปรับตัวลงมาน่าสนใจ", "🟢"
    if score >= 40:
        return "อยู่ในการจับตา — เริ่มมีสัญญาณน่าสนใจ แต่ยังไม่ชัดเจนพอ", "🟡"
    return "ยังไม่เข้าเกณฑ์ — ราคายังไม่ได้ปรับตัวลงมากพอตามเกณฑ์ที่ตั้งไว้", "⚪"


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------

st.title("📉 Dip Screener")
st.caption("หาหุ้นที่ราคาอยู่ในโซนน่าช้อน ใช้งานง่าย เปิดได้ทั้งมือถือและคอมพิวเตอร์")

st.subheader("🔍 ค้นหาหุ้นที่สนใจ")
col_a, col_b = st.columns([3, 1])
with col_a:
    search_ticker = st.text_input(
        "พิมพ์ชื่อย่อหุ้น",
        placeholder="เช่น AAPL, PTT.BK",
        label_visibility="collapsed",
    )
with col_b:
    search_btn = st.button("ค้นหา", use_container_width=True)

if search_btn and search_ticker.strip():
    t = search_ticker.strip().upper()
    with st.spinner(f"กำลังตรวจสอบ {t}..."):
        result = analyze_ticker(t)

    if result is None:
        st.error(
            f"ไม่พบข้อมูลหุ้น {t} ตรวจสอบชื่อย่ออีกครั้ง "
            "(หุ้นไทยต้องมี .BK ต่อท้าย เช่น PTT.BK)"
        )
    else:
        score = normalize_score(pd.Series(result), DEFAULT_WEIGHTS)
        msg, emoji = verdict_text(score)
        st.markdown(
            f"""
            <div class="card" style="border-width:2px;">
                <div class="card-top">
                    <span class="ticker-name" style="font-size:1.4em">{result['Ticker']}</span>
                    {score_badge(score)}
                </div>
                <div class="price-text">ราคาล่าสุด {result['ราคาล่าสุด']:.2f}
                    (52w สูง {result['52wHigh']:.2f} / ต่ำ {result['52wLow']:.2f})</div>
                <div style="margin-top:10px; font-size:1.05em;">{emoji} {msg}</div>
                <div class="metric-row" style="margin-top:8px;">
                    ตกจากจุดสูงสุด {result['%ต่ำกว่าจุดสูงสุด']:.1f}% ·
                    RSI {result['RSI(14)']:.1f} ·
                    เทียบ MA50 {result['%เทียบMA50']:.1f}% ·
                    เทียบ MA200 {result['%เทียบMA200']:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        hist = yf.Ticker(t).history(period="1y", interval="1d", auto_adjust=True)
        if not hist.empty:
            chart_df = hist[["Close"]].copy()
            chart_df["MA50"] = chart_df["Close"].rolling(50).mean()
            chart_df["MA200"] = chart_df["Close"].rolling(200).mean()
            st.line_chart(chart_df)

st.divider()
st.subheader("📋 หรือดูภาพรวมจากรายการเฝ้าดู")

market = st.radio(
    "เลือกตลาด", ["สหรัฐฯ (US)", "ไทย (SET)", "ทั้งสอง"], index=2, horizontal=True
)

with st.expander("⚙️ ตั้งค่าขั้นสูง (ไม่บังคับ)"):
    st.caption("ปรับน้ำหนักแต่ละเงื่อนไข หรือแก้รายชื่อหุ้นเองได้ที่นี่")
    c1, c2 = st.columns(2)
    with c1:
        w_off_high = st.slider("ตกจากจุดสูงสุด 52 สัปดาห์", 0.0, 3.0, 1.0, 0.1)
        w_rsi = st.slider("RSI ต่ำ (oversold)", 0.0, 3.0, 1.0, 0.1)
    with c2:
        w_ma50 = st.slider("ต่ำกว่าเส้นค่าเฉลี่ย 50 วัน", 0.0, 3.0, 1.0, 0.1)
        w_ma200 = st.slider("ต่ำกว่าเส้นค่าเฉลี่ย 200 วัน", 0.0, 3.0, 0.5, 0.1)

    us_text = st.text_area("รายชื่อหุ้น US", ", ".join(DEFAULT_US), height=80)
    th_text = st.text_area("รายชื่อหุ้น SET (.BK)", ", ".join(DEFAULT_TH), height=80)
    top_n = st.slider("แสดงกี่อันดับแรก", 5, 30, 10)

weights = {"off_high": w_off_high, "rsi": w_rsi, "ma50": w_ma50, "ma200": w_ma200}

tickers: list[str] = []
if market in ("สหรัฐฯ (US)", "ทั้งสอง"):
    tickers += [t.strip().upper() for t in us_text.split(",") if t.strip()]
if market in ("ไทย (SET)", "ทั้งสอง"):
    tickers += [t.strip().upper() for t in th_text.split(",") if t.strip()]
tickers = list(dict.fromkeys(tickers))

refresh = st.button("🔄 ดึงข้อมูลล่าสุด", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state["results"] = None
    st.session_state["fetched_at"] = None

should_fetch = refresh or st.session_state["results"] is None

if should_fetch:
    progress = st.progress(0.0, text="กำลังดึงข้อมูล...")
    rows = []
    for i, t in enumerate(tickers):
        r = analyze_ticker(t)
        if r:
            rows.append(r)
        progress.progress((i + 1) / max(len(tickers), 1), text=f"ดึงข้อมูล {t}")
        time.sleep(0.03)
    progress.empty()

    df = pd.DataFrame(rows)
    if not df.empty:
        df["คะแนนน่าช้อน"] = df.apply(lambda r: normalize_score(r, weights), axis=1)
        df = df.sort_values("คะแนนน่าช้อน", ascending=False).reset_index(drop=True)

    st.session_state["results"] = df
    st.session_state["fetched_at"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

df = st.session_state["results"]

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if df is None or df.empty:
    st.warning("ไม่พบข้อมูล ลองกด 🔄 ดึงข้อมูลล่าสุด อีกครั้ง")
else:
    st.caption(f"อัปเดตล่าสุด: {st.session_state['fetched_at']}")

    n_high = int((df["คะแนนน่าช้อน"] >= 70).sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("สแกนทั้งหมด", len(df))
    col2.metric("น่าสนใจมาก 🟢", n_high)
    col3.metric("คะแนนเฉลี่ย", f"{df['คะแนนน่าช้อน'].mean():.0f}")

    st.subheader(f"อันดับหุ้นน่าช้อน (Top {top_n})")

    top = df.head(top_n)
    for _, row in top.iterrows():
        st.markdown(
            f"""
            <div class="card">
                <div class="card-top">
                    <span class="ticker-name">{row['Ticker']}</span>
                    {score_badge(row['คะแนนน่าช้อน'])}
                </div>
                <div class="price-text">ราคาล่าสุด {row['ราคาล่าสุด']:.2f}
                    (52w สูง {row['52wHigh']:.2f} / ต่ำ {row['52wLow']:.2f})</div>
                <div class="metric-row">
                    ตกจากจุดสูงสุด {row['%ต่ำกว่าจุดสูงสุด']:.1f}% ·
                    RSI {row['RSI(14)']:.1f} ·
                    เทียบ MA50 {row['%เทียบMA50']:.1f}% ·
                    เทียบ MA200 {row['%เทียบMA200']:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📊 ดูตารางแบบเต็ม / ดาวน์โหลด CSV"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ ดาวน์โหลด CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="dip_screener.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("📈 ดูกราฟราคารายตัว"):
        pick = st.selectbox("เลือกหุ้น", df["Ticker"].tolist())
        if pick:
            hist = yf.Ticker(pick).history(period="1y", interval="1d", auto_adjust=True)
            if not hist.empty:
                chart_df = hist[["Close"]].copy()
                chart_df["MA50"] = chart_df["Close"].rolling(50).mean()
                chart_df["MA200"] = chart_df["Close"].rolling(200).mean()
                st.line_chart(chart_df)

st.divider()
with st.expander("ℹ️ หมายเหตุสำคัญ"):
    st.markdown(
        """
- **ไม่ใช่ real-time tick แบบแอปโบรกเกอร์** — ข้อมูลจาก Yahoo Finance ผ่าน yfinance
  ปกติหน่วงราวไม่กี่นาทีถึง ~15 นาที เพียงพอสำหรับสแกนหาโอกาส แต่ไม่เหมาะกับการเทรดที่ต้องการราคาสด
- **Webull ไม่มี public API อย่างเป็นทางการ** สำหรับนักพัฒนาภายนอก แอปนี้จึงใช้ Yahoo Finance แทน
  คุณยังส่งคำสั่งซื้อขายจริงผ่าน Webull ได้ตามปกติ
- คะแนน "น่าช้อน" เป็นแค่ตัวช่วยกรองตามเทคนิคที่คุณเลือกเอง **ไม่ใช่คำแนะนำการลงทุน**
- รายชื่อหุ้น SET เริ่มต้นเป็นชุดหุ้นใหญ่คุ้นเคย ไม่ใช่ SET50 ที่อัปเดตล่าสุดเป๊ะ แก้ไขได้ในตั้งค่าขั้นสูง
        """
    )
