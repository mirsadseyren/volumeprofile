import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import yfinance as yf
from scanner import read_tickers, calculate_poc

st.set_page_config(
    page_title="POC Sinyal Tarayıcı",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

h1 {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.2rem !important;
}

.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(10px);
    text-align: center;
}

.metric-card .label {
    color: rgba(255,255,255,0.55);
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.metric-card .value {
    color: #fff;
    font-size: 2rem;
    font-weight: 700;
}

.metric-card .value.green { color: #34d399; }
.metric-card .value.red   { color: #f87171; }
.metric-card .value.blue  { color: #60a5fa; }

.signal-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
    transition: all 0.25s ease;
}

.signal-card:hover {
    background: rgba(255,255,255,0.10);
    border-color: rgba(167, 139, 250, 0.7);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(167, 139, 250, 0.15);
}

.ticker-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: #fff;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 4px 14px;
    border-radius: 8px;
    margin-bottom: 8px;
    letter-spacing: 0.04em;
}

.info-row {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin-top: 8px;
}

.info-item { color: rgba(255,255,255,0.7); font-size: 0.85rem; }
.info-item span { color: #fff; font-weight: 600; }

.poc-up   { color: #34d399; font-weight: 700; }
.density  { color: #fbbf24; font-weight: 700; }

.section-title {
    color: rgba(255,255,255,0.85);
    font-size: 1.1rem;
    font-weight: 600;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    border: none;
    color: white;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    border-radius: 10px;
    transition: all 0.2s ease;
    width: 100%;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5);
    background: linear-gradient(135deg, #6d28d9, #1d4ed8);
}

div[data-testid="stMetric"] label { color: rgba(255,255,255,0.55) !important; }
div[data-testid="stMetric"] div   { color: #fff !important; }

.stSpinner > div { border-top-color: #7c3aed !important; }

.warning-box {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    color: #fbbf24;
    font-size: 0.9rem;
    margin-bottom: 16px;
}

.info-box {
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    color: #93c5fd;
    font-size: 0.9rem;
    margin-bottom: 16px;
}

.empty-box {
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    color: rgba(255,255,255,0.35);
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Yardımcı: Volume Density hesabı ───────────────────────────────────────
def compute_poc_density(close_series, vol_series, bins=50):
    valid = pd.concat([close_series, vol_series], axis=1).dropna()
    if valid.empty or valid.iloc[:,0].min() == valid.iloc[:,0].max():
        return None, 0.0
    c = valid.iloc[:, 0]
    v = valid.iloc[:, 1]
    price_bins = np.linspace(c.min(), c.max(), bins)
    idx = np.digitize(c, price_bins)
    vp = np.zeros(bins)
    for i in range(len(c)):
        j = idx[i] - 1
        j = max(0, min(j, bins - 1))
        vp[j] += float(v.iloc[i])
    total = vp.sum()
    poc_i = np.argmax(vp)
    density = vp[poc_i] / total if total > 0 else 0.0
    poc_px = price_bins[poc_i]
    return poc_px, density


# ─── Ana tarama fonksiyonu (BUGÜNÜN verisiyle YARIN alınacaklar) ────────────
def find_signals(
    intraday_cache="bist_all_backtest_cache.pkl",
    daily_cache="bist_daily_1mo_cache.pkl",
    monthly_perf_threshold=30.0,
    density_threshold=0.10,
    progress_cb=None,
):
    if not os.path.exists(intraday_cache):
        return None, "⚠️ Önbellek bulunamadı. Lütfen main.py'i çalıştırarak verileri indirin."

    data = pd.read_pickle(intraday_cache)
    dates = pd.Series(data.index).dt.normalize().unique()
    if len(dates) < 1:
        return None, "Yeterli veri yok!"

    today_date = dates[-1]
    today_mask = data.index.normalize() == today_date
    df_today = data.loc[today_mask]

    # Aylık filtre ─────────────────────────────────────────────────────────
    valid_monthly = set()
    if os.path.exists(daily_cache):
        try:
            daily = pd.read_pickle(daily_cache)
            for t in set(daily.columns.get_level_values(1)):
                if ('Close', t) in daily.columns:
                    s = daily['Close'][t].dropna()
                    if len(s) > 1 and s.iloc[0] > 0:
                        perf = (s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100
                        if perf > monthly_perf_threshold:
                            valid_monthly.add(t)
        except:
            pass

    try:
        all_tickers = sorted(set(data.columns.get_level_values(1)))
    except:
        return None, "Veri yapısı okunamadı."

    tickers = [t for t in all_tickers if t in valid_monthly] if valid_monthly else all_tickers

    signals = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(i / total, f"{ticker} inceleniyor… ({i+1}/{total})")
        try:
            if ('Close', ticker) not in df_today.columns or ('Volume', ticker) not in df_today.columns:
                continue
            close_t = df_today['Close'][ticker].dropna()
            vol_t   = df_today['Volume'][ticker].dropna()
            if close_t.empty or vol_t.empty:
                continue

            last_price = close_t.iloc[-1]
            poc_price, density = compute_poc_density(close_t, vol_t)

            if poc_price is None or last_price <= 0:
                continue
            if density < density_threshold:
                continue
            if last_price >= poc_price:          # Fiyat POC'nin altında olmalı
                continue

            diff_pct = (poc_price - last_price) / last_price * 100

            signals.append({
                'Hisse':           ticker,
                'Bugün Kapanış':   round(last_price, 2),
                'Hedef POC':       round(poc_price, 2),
                'POC Uzaklık (%)': round(diff_pct, 2),
                'POC Yoğunluk (%)':round(density * 100, 1),
            })
        except:
            continue

    if progress_cb:
        progress_cb(1.0, "Tarama tamamlandı!")

    if not signals:
        return pd.DataFrame(), None

    df = pd.DataFrame(signals).sort_values("POC Uzaklık (%)", ascending=False).reset_index(drop=True)
    return df, None


# ─── Streamlit Arayüzü ──────────────────────────────────────────────────────
st.markdown("<h1>📈 POC Sinyal Tarayıcı</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:rgba(255,255,255,0.55); margin-top:-12px; margin-bottom:24px;'>"
    "Bugünün verisinden yarın alınacak hisseleri tespit eder — "
    "Aylık %30+ getiri · POC yoğunluk · Fiyat &lt; POC</p>",
    unsafe_allow_html=True
)

# Sidebar──────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Filtre Parametreleri")
    monthly_thresh = st.slider("Aylık Getiri Eşiği (%)", 5, 100, 30, 5)
    density_thresh = st.slider("POC Yoğunluk Eşiği (%)", 1, 50, 10, 1) / 100.0
    st.markdown("---")
    st.markdown(
        "<div style='color:rgba(255,255,255,0.4); font-size:0.78rem;'>"
        "Veri kaynağı: <code>bist_all_backtest_cache.pkl</code><br>"
        "Yenilemek için main.py'i çalıştırın.</div>",
        unsafe_allow_html=True
    )

# Tarama butonu ───────────────────────────────────
col_btn, col_info = st.columns([2, 5])
with col_btn:
    run = st.button("🔍 Taramayı Başlat")

with col_info:
    if not os.path.exists("bist_all_backtest_cache.pkl"):
        st.markdown(
            "<div class='warning-box'>⚠️ Önbellek dosyası bulunamadı. "
            "Lütfen önce <code>main.py</code> çalıştırın.</div>",
            unsafe_allow_html=True
        )
    else:
        mtime = os.path.getmtime("bist_all_backtest_cache.pkl")
        mod_str = time.strftime("%d.%m.%Y %H:%M", time.localtime(mtime))
        st.markdown(
            f"<div class='info-box'>📦 Önbellek: <b>{mod_str}</b> tarihli veriler kullanılıyor.</div>",
            unsafe_allow_html=True
        )

# Sonuçlar ────────────────────────────────────────
if run:
    progress_bar = st.progress(0, text="Tarama başlıyor…")

    def update_progress(val, msg):
        progress_bar.progress(val, text=msg)

    with st.spinner("Hisseler taranıyor…"):
        df_signals, err = find_signals(
            monthly_perf_threshold=monthly_thresh,
            density_threshold=density_thresh,
            progress_cb=update_progress,
        )

    progress_bar.empty()

    if err:
        st.error(err)
    elif df_signals is None or df_signals.empty:
        st.markdown(
            "<div class='empty-box'>🔎 Kriterlere uyan hisse bulunamadı.<br>"
            "<small>Filtreleri gevşetmeyi deneyin.</small></div>",
            unsafe_allow_html=True
        )
    else:
        n = len(df_signals)
        avg_dist = df_signals["POC Uzaklık (%)"].mean()
        avg_dens = df_signals["POC Yoğunluk (%)"].mean()
        best     = df_signals.iloc[0]["Hisse"]

        # KPI kartlar
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(
            f"<div class='metric-card'><div class='label'>Sinyal Sayısı</div>"
            f"<div class='value blue'>{n}</div></div>", unsafe_allow_html=True)
        k2.markdown(
            f"<div class='metric-card'><div class='label'>Ort. POC Uzaklığı</div>"
            f"<div class='value green'>%{avg_dist:.1f}</div></div>", unsafe_allow_html=True)
        k3.markdown(
            f"<div class='metric-card'><div class='label'>Ort. POC Yoğunluğu</div>"
            f"<div class='value'>%{avg_dens:.1f}</div></div>", unsafe_allow_html=True)
        k4.markdown(
            f"<div class='metric-card'><div class='label'>En İyi Aday</div>"
            f"<div class='value' style='font-size:1.4rem'>{best}</div></div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='section-title'>📋 Yarın Alım Sinyali Veren Hisseler ({n} adet)</div>",
            unsafe_allow_html=True
        )

        # Kart grid
        for _, row in df_signals.iterrows():
            st.markdown(f"""
            <div class="signal-card">
                <div class="ticker-badge">{row['Hisse']}</div>
                <div class="info-row">
                    <div class="info-item">💰 Bugün Kapanış: <span>{row['Bugün Kapanış']:.2f} ₺</span></div>
                    <div class="info-item">🎯 Hedef POC: <span class="poc-up">{row['Hedef POC']:.2f} ₺</span></div>
                    <div class="info-item">📐 POC Uzaklığı: <span class="poc-up">%{row['POC Uzaklık (%)']:.2f}</span></div>
                    <div class="info-item">🔥 POC Yoğunluğu: <span class="density">%{row['POC Yoğunluk (%)']:.1f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Tablo + Excel indirme
        st.markdown("<div class='section-title'>📊 Tablo Görünümü</div>", unsafe_allow_html=True)
        st.dataframe(df_signals, use_container_width=True, hide_index=True)

        excel_buf = pd.ExcelWriter("/tmp/sinyaller.xlsx", engine="openpyxl")
        df_signals.to_excel(excel_buf, index=False, sheet_name="Sinyaller")
        excel_buf.close()
        with open("/tmp/sinyaller.xlsx", "rb") as f:
            st.download_button(
                "⬇️ Excel Olarak İndir",
                data=f.read(),
                file_name=f"poc_sinyaller_{time.strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.markdown(
        "<div class='empty-box' style='margin-top:40px;'>"
        "🔍 Taramayı başlatmak için butona basın.</div>",
        unsafe_allow_html=True
    )
