"""
PEAK GUESSING — หาหุ้นที่ราคาอยู่ในโซนน่าช้อน (US + Thai/SET)
ธีมสปอร์ต มืด กระชับ ใช้งานง่ายทั้งมือถือและคอมพิวเตอร์

รันด้วย: streamlit run app.py

ข้อมูลราคาดึงจาก Yahoo Finance ผ่านไลบรารี yfinance (ฟรี, ไม่ต้องมี API key)
ไม่ใช่ real-time แบบ tick-by-tick เหมือนแอปโบรกเกอร์ แต่โดยทั่วไปหน่วง
ไม่กี่นาทีถึงราว 15 นาที เพียงพอสำหรับใช้สแกนหาโอกาส ไม่เหมาะสำหรับ day trading
ที่ต้องการราคาแบบ tick จริง
"""

import re
import time
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Peak Guessing", page_icon="🏁", layout="centered")


def _html(s: str) -> str:
    """Collapse a multi-line/indented HTML string to one line.

    Streamlit's markdown renderer treats lines indented 4+ spaces as a
    code block, which leaks raw tags as visible text on nested <div>
    markup. Stripping newlines/indentation avoids that entirely.
    """
    return re.sub(r"\n\s*", " ", s.strip())

# ---------------------------------------------------------------------------
# Sport theme styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    .block-container { padding-top: 1.3rem; padding-bottom: 3rem; max-width: 640px; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .kicker {
        font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: #C6FF3D;
        font-weight: 600;
    }
    .title-big {
        font-family: 'Oswald', sans-serif; font-size: 40px; line-height: 0.95; text-transform: uppercase;
        font-weight: 700; margin: 8px 0 0; letter-spacing: -0.01em;
    }
    .accent-bar { width: 80px; height: 6px; background: #C6FF3D; margin: 12px 0 14px; transform: skewX(-18deg); }
    .sub-text { font-size: 13px; color: #8A93A3; }
    .section-label {
        font-family: 'Oswald', sans-serif; text-transform: uppercase; letter-spacing: 0.05em;
        font-size: 14px; color: #F5F6F8; margin-top: 22px; margin-bottom: 8px; font-weight: 600;
    }

    div.stButton > button {
        background: #C6FF3D !important; color: #0A0C10 !important; border: none !important;
        font-family: 'Oswald', sans-serif !important; font-weight: 600 !important; letter-spacing: 0.03em !important;
        text-transform: uppercase !important; border-radius: 10px !important; height: 3em !important;
    }
    div[role="radiogroup"] label {
        background: #12151C; border: 1.5px solid #1E2430; border-radius: 8px; padding: 6px 14px !important;
        font-family: 'Oswald', sans-serif; text-transform: uppercase; font-size: 12.5px; font-weight: 700;
    }
    div[data-testid="stExpander"] { background: #12151C; border: 1.5px solid #1E2430; border-radius: 10px; }

    .scoreboard {
        display: flex; background: #12151C; border: 1.5px solid #1E2430; border-radius: 12px;
        overflow: hidden; margin: 16px 0;
    }
    .sb-cell { flex: 1; text-align: center; padding: 14px 8px; }
    .sb-cell + .sb-cell { border-left: 1.5px solid #1E2430; }
    .sb-v { font-family: 'Oswald', sans-serif; font-size: 26px; font-weight: 600; }
    .sb-l { font-size: 9.5px; letter-spacing: 0.08em; text-transform: uppercase; color: #8A93A3; margin-top: 2px; }

    .card {
        background: #12151C; border: 1.5px solid #1E2430; border-left-width: 4px; border-radius: 10px;
        padding: 14px; margin-bottom: 10px;
    }
    .card-top { display: flex; justify-content: space-between; align-items: center; }
    .left { display: flex; align-items: center; gap: 8px; }
    .flag {
        font-size: 9px; font-weight: 700; letter-spacing: 0.04em; color: #5B6472;
        border: 1px solid #262C38; border-radius: 4px; padding: 1px 5px;
    }
    .ticker { font-family: 'Oswald', sans-serif; font-size: 17px; font-weight: 600; }
    .tag {
        font-size: 10px; font-weight: 700; letter-spacing: 0.08em; padding: 3px 8px; border-radius: 5px;
        border: 1.5px solid; font-family: 'Oswald', sans-serif;
    }
    .card-mid { display: flex; justify-content: space-between; align-items: center; margin-top: 12px; }
    .price { font-family: 'Oswald', sans-serif; font-size: 24px; font-weight: 500; }
    .range { font-size: 10.5px; color: #5B6472; margin-top: 2px; }
    .verdict-label { font-size: 11px; color: #8A93A3; margin-top: 6px; }
    .stat-row { display: flex; gap: 16px; margin-top: 12px; padding-top: 12px; border-top: 1px solid #1E2430; }
    .stat .v { font-size: 13px; font-weight: 600; }
    .stat .l { font-size: 9px; color: #5B6472; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 1px; }

    .footnote-box {
        margin-top: 22px; padding: 14px; border-radius: 10px; background: #12151C;
        border: 1px dashed #1E2430; font-size: 12px; color: #8A93A3; line-height: 1.7;
    }
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


DEFAULT_WEIGHTS = {"off_high": 1.0, "rsi": 1.0, "ma50": 1.0, "ma200": 0.5}


def tier_info(score: float) -> dict:
    if pd.isna(score):
        return {"label": "ไม่มีข้อมูล", "tag": "N/A", "color": "#5B6472"}
    if score >= 70:
        return {"label": "น่าช้อน", "tag": "HOT", "color": "#C6FF3D"}
    if score >= 40:
        return {"label": "จับตา", "tag": "WATCH", "color": "#FF6B35"}
    return {"label": "ยังไม่เข้าเกณฑ์", "tag": "COLD", "color": "#5B6472"}


def market_flag(ticker: str) -> str:
    return "SET" if ticker.upper().endswith(".BK") else "US"


def score_ring_svg(score: float, color: str, size: int = 60) -> str:
    score_val = 0 if pd.isna(score) else max(0, min(score, 100))
    stroke = 5
    r = (size - stroke) / 2
    c = 2 * math.pi * r
    offset = c - (score_val / 100) * c
    cx = cy = size / 2
    label = "–" if pd.isna(score) else f"{score:.0f}"
    return _html(f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1E2430" stroke-width="{stroke}" />
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
            stroke-dasharray="{c:.2f}" stroke-dashoffset="{offset:.2f}" stroke-linecap="round"
            transform="rotate(-90 {cx} {cy})" />
        <text x="50%" y="52%" text-anchor="middle" dominant-baseline="middle"
            style="font-family:'Oswald',sans-serif;font-size:17px;font-weight:600;fill:#F5F6F8;">{label}</text>
    </svg>
    """)


def render_card(ticker: str, price: float, low: float, high: float,
                 off_high: float, rsi: float, ma50: float, score: float) -> None:
    t = tier_info(score)
    ring = score_ring_svg(score, t["color"])
    st.markdown(
        _html(f"""
        <div class="card" style="border-left-color:{t['color']}">
            <div class="card-top">
                <div class="left">
                    <span class="flag">{market_flag(ticker)}</span>
                    <span class="ticker">{ticker}</span>
                </div>
                <span class="tag" style="color:{t['color']};border-color:{t['color']}">{t['tag']}</span>
            </div>
            <div class="card-mid">
                <div>
                    <div class="price">{price:.2f}</div>
                    <div class="range">52w {low:.2f}–{high:.2f}</div>
                    <div class="verdict-label">{t['label']}</div>
                </div>
                {ring}
            </div>
            <div class="stat-row">
                <div class="stat"><div class="v">{off_high:.1f}%</div><div class="l">Off-High</div></div>
                <div class="stat"><div class="v">{rsi:.1f}</div><div class="l">RSI</div></div>
                <div class="stat"><div class="v">{ma50:.1f}%</div><div class="l">MA50</div></div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------

st.markdown(
    _html("""
    <div class="kicker">● PEAK GUESSING</div>
    <div class="title-big">PEAK<br/>GUESSING</div>
    <div class="accent-bar"></div>
    <div class="sub-text">จัดฟอร์มหุ้นแบบสนามแข่ง — หาราคาที่น่าช้อนได้ในไม่กี่วินาที</div>
    """),
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">🔍 ค้นหาฟอร์มหุ้น</div>', unsafe_allow_html=True)
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
    with st.spinner(f"กำลังตรวจสอบฟอร์ม {t}..."):
        result = analyze_ticker(t)

    if result is None:
        st.error(
            f"ไม่พบข้อมูลหุ้น {t} ตรวจสอบชื่อย่ออีกครั้ง "
            "(หุ้นไทยต้องมี .BK ต่อท้าย เช่น PTT.BK)"
        )
    else:
        score = normalize_score(pd.Series(result), DEFAULT_WEIGHTS)
        render_card(
            result["Ticker"], result["ราคาล่าสุด"], result["52wLow"], result["52wHigh"],
            result["%ต่ำกว่าจุดสูงสุด"], result["RSI(14)"], result["%เทียบMA50"], score,
        )
        hist = yf.Ticker(t).history(period="1y", interval="1d", auto_adjust=True)
        if not hist.empty:
            chart_df = hist[["Close"]].copy()
            chart_df["MA50"] = chart_df["Close"].rolling(50).mean()
            chart_df["MA200"] = chart_df["Close"].rolling(200).mean()
            st.line_chart(chart_df)

st.markdown('<div class="section-label">📋 ลีกหุ้นที่เฝ้าดู</div>', unsafe_allow_html=True)

market = st.radio(
    "เลือกตลาด", ["ทั้งหมด", "US", "SET"], index=0, horizontal=True, label_visibility="collapsed"
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
if market in ("US", "ทั้งหมด"):
    tickers += [x.strip().upper() for x in us_text.split(",") if x.strip()]
if market in ("SET", "ทั้งหมด"):
    tickers += [x.strip().upper() for x in th_text.split(",") if x.strip()]
tickers = list(dict.fromkeys(tickers))

refresh = st.button("🔄 ดึงข้อมูลล่าสุด", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state["results"] = None
    st.session_state["fetched_at"] = None

should_fetch = refresh or st.session_state["results"] is None

if should_fetch:
    progress = st.progress(0.0, text="กำลังดึงข้อมูล...")
    rows = []
    for i, tk in enumerate(tickers):
        r = analyze_ticker(tk)
        if r:
            rows.append(r)
        progress.progress((i + 1) / max(len(tickers), 1), text=f"ดึงข้อมูล {tk}")
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

    n_hot = int((df["คะแนนน่าช้อน"] >= 70).sum())
    avg_score = df["คะแนนน่าช้อน"].mean()
    st.markdown(
        _html(f"""
        <div class="scoreboard">
            <div class="sb-cell"><div class="sb-v">{len(df)}</div><div class="sb-l">สแกน</div></div>
            <div class="sb-cell"><div class="sb-v" style="color:#C6FF3D">{n_hot}</div><div class="sb-l">Hot</div></div>
            <div class="sb-cell"><div class="sb-v">{avg_score:.0f}</div><div class="sb-l">เฉลี่ย</div></div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    top = df.head(top_n)
    for _, row in top.iterrows():
        render_card(
            row["Ticker"], row["ราคาล่าสุด"], row["52wLow"], row["52wHigh"],
            row["%ต่ำกว่าจุดสูงสุด"], row["RSI(14)"], row["%เทียบMA50"], row["คะแนนน่าช้อน"],
        )

    with st.expander("📊 ดูตารางแบบเต็ม / ดาวน์โหลด CSV"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ ดาวน์โหลด CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="peak_guessing.csv",
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

st.markdown(
    _html("""
    <div class="footnote-box">
    ⓘ <b>ไม่ใช่ real-time tick แบบแอปโบรกเกอร์</b> — ข้อมูลจาก Yahoo Finance ผ่าน yfinance
    ปกติหน่วงราวไม่กี่นาทีถึง ~15 นาที เพียงพอสำหรับสแกนหาโอกาส แต่ไม่เหมาะกับการเทรดที่ต้องการราคาสด<br/><br/>
    <b>Webull ไม่มี public API อย่างเป็นทางการ</b> สำหรับนักพัฒนาภายนอก แอปนี้จึงใช้ Yahoo Finance แทน
    คุณยังส่งคำสั่งซื้อขายจริงผ่าน Webull ได้ตามปกติ<br/><br/>
    คะแนน "ฟอร์ม" เป็นแค่ตัวช่วยกรองตามเทคนิคที่คุณเลือกเอง <b>ไม่ใช่คำแนะนำการลงทุน</b> —
    รายชื่อหุ้น SET เริ่มต้นเป็นชุดหุ้นใหญ่คุ้นเคย ไม่ใช่ SET50 ที่อัปเดตล่าสุดเป๊ะ แก้ไขได้ในตั้งค่าขั้นสูง
    </div>
    """),
    unsafe_allow_html=True,
)
