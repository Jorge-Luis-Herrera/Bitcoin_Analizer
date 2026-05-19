"""
V4 Interactive Dashboard — Streamlit + Plotly
==============================================
BTC price history, backtest (N-day walk-forward), and forward forecast
with real-time monitoring. Bilingual EN/ES.
"""

import os, numpy as np, pandas as pd, yfinance as yf, torch
import requests, threading
from datetime import datetime, timedelta
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

HISTORY_DAYS = 1000
MAX_FORECAST = 60
MAX_BACKTEST = 60

MODELS = {
    'Chronos-T5-Tiny': 'amazon/chronos-t5-tiny',
    'Chronos-T5-Small': 'amazon/chronos-t5-small',
    'Chronos-T5-Base': 'amazon/chronos-t5-base',
}

DEFAULT_MODEL = 'Chronos-T5-Tiny'

LANG = {
    'en': {
        'page_title': 'BTC Forecast V4',
        'controls': 'Controls',
        'model': 'Chronos Model',
        'forward': 'Forward',
        'forecast_days': 'Forecast days',
        'backtest': 'Backtest',
        'backtest_window': 'Backtest window (days)',
        'refresh_data': 'Refresh data',
        'legend': 'Legend',
        'legend_blue': 'BTC actual price',
        'legend_red': 'Chronos forecast (median)',
        'legend_shaded': 'P10-P90 band',
        'legend_points': 'Backtest (<2% / 2-5% / >5% error)',
        'insufficient_data': 'Insufficient data.',
        'market_context': 'Market Context',
        'tab_forward': 'Forward',
        'tab_backtest': 'Backtest',
        'tab_combined': 'Combined',
        'tab_live': 'Live',
        'forecast_n_days': 'Forecast {} days ahead',
        'backtest_n_days': 'Backtest: last {} days',
        'mape': 'MAPE',
        'hits': 'Hits (<2%)',
        'medium': 'Medium (2-5%)',
        'misses': 'Misses (>5%)',
        'daily_breakdown': 'Daily Breakdown',
        'combined_title': 'Combined: {}d backtest + {}d forecast',
        'current_price': 'Current Price',
        'backtest_mape': 'Backtest MAPE ({:d}d)',
        'forecast_n': 'Forecast +{:d}d',
        'forecast_1d': 'Forecast +1d',
        'waiting_data': 'Waiting for live data...',
        'language': 'Language',
        'updated': 'Updated',
        'change_pct': 'Change %',
        'day': 'Day',
        'date': 'Date',
        'forecast': 'Forecast',
        'actual': 'Actual',
        'prediction': 'Prediction',
        'error': 'Error %',
    },
    'es': {
        'page_title': 'BTC Pronostico V4',
        'controls': 'Controles',
        'model': 'Modelo Chronos',
        'forward': 'Futuro',
        'forecast_days': 'Dias de pronostico',
        'backtest': 'Backtest',
        'backtest_window': 'Ventana de backtest (dias)',
        'refresh_data': 'Actualizar datos',
        'legend': 'Leyenda',
        'legend_blue': 'Precio real BTC',
        'legend_red': 'Pronostico Chronos (mediana)',
        'legend_shaded': 'Banda P10-P90',
        'legend_points': 'Backtest (error <2% / 2-5% / >5%)',
        'insufficient_data': 'Datos insuficientes.',
        'market_context': 'Contexto de Mercado',
        'tab_forward': 'Futuro',
        'tab_backtest': 'Backtest',
        'tab_combined': 'Combinado',
        'tab_live': 'En vivo',
        'forecast_n_days': 'Pronostico {} dias adelante',
        'backtest_n_days': 'Backtest: ultimos {} dias',
        'mape': 'MAPE',
        'hits': 'Aciertos (<2%)',
        'medium': 'Medio (2-5%)',
        'misses': 'Fallos (>5%)',
        'daily_breakdown': 'Desglose diario',
        'combined_title': 'Combinado: {}d backtest + {}d pronostico',
        'current_price': 'Precio actual',
        'backtest_mape': 'MAPE backtest ({:d}d)',
        'forecast_n': 'Pronostico +{:d}d',
        'forecast_1d': 'Pronostico +1d',
        'waiting_data': 'Esperando datos en vivo...',
        'language': 'Idioma',
        'updated': 'Actualizado',
        'change_pct': 'Cambio %',
        'day': 'Dia',
        'date': 'Fecha',
        'forecast': 'Pronostico',
        'actual': 'Real',
        'prediction': 'Prediccion',
        'error': 'Error %',
    },
}

st.set_page_config(
    page_title="BTC Forecast V4",
    page_icon="\u20bf",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #f0f0f0 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
    .metric-card {
        background: #1a1d23; border-radius: 10px; padding: 12px;
        border: 1px solid #2d3139; text-align: center;
    }
    .metric-value { font-size: 22px; font-weight: bold; color: #f0f0f0; }
    .metric-label { font-size: 11px; color: #8b8fa3; }
    .live-dot {
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; background: #52b788;
        animation: pulse 1.5s infinite; margin-right: 6px;
    }
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.3); }
        100% { opacity: 1; transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)


# ===================== PIPELINE LOADING =====================

@st.cache_resource(show_spinner=False)
def load_chronos_pipeline(model_name):
    from chronos import ChronosPipeline
    hf_name = MODELS[model_name]
    return ChronosPipeline.from_pretrained(
        hf_name, device="cpu",
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_btc_data():
    end = datetime.now()
    start = end - timedelta(days=HISTORY_DAYS)
    result = []
    def _dl():
        try:
            btc = yf.download('BTC-USD', start=start.strftime('%Y-%m-%d'),
                               end=end.strftime('%Y-%m-%d'), progress=False)
            if not btc.empty:
                if isinstance(btc.columns, pd.MultiIndex):
                    btc.columns = btc.columns.droplevel(1)
                result.append(btc['Close'])
        except:
            pass
    t = threading.Thread(target=_dl, daemon=True)
    t.start()
    t.join(timeout=30)
    return result[0] if result else pd.Series(dtype=float)


def fetch_intraday_data():
    result = {'15m': None, '1h': None}
    def _dl_15m():
        try:
            d = yf.download('BTC-USD', period='3d', interval='15m', progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.droplevel(1)
                result['15m'] = d
        except:
            pass
    def _dl_1h():
        try:
            d = yf.download('BTC-USD', period='3d', interval='1h', progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.droplevel(1)
                result['1h'] = d
        except:
            pass
    t1 = threading.Thread(target=_dl_15m, daemon=True)
    t2 = threading.Thread(target=_dl_1h, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)
    return result


def compute_hourly_volatility(hourly_df):
    ret = hourly_df['Close'].pct_change().dropna()
    return ret.std() * 100 if len(ret) > 1 else 0.0


def fetch_live_context():
    ctx = {}
    btc_data = []
    def _dl_btc():
        try:
            d = yf.download('BTC-USD', period='5d', progress=False)
            if not d.empty:
                btc_data.append(d)
        except:
            pass
    t = threading.Thread(target=_dl_btc, daemon=True)
    t.start()
    t.join(timeout=15)
    if btc_data:
        btc = btc_data[0]
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = btc.columns.droplevel(1)
        ctx['BTC Price'] = f"${btc['Close'].iloc[-1]:,.2f}"
        ctx['24h Change'] = f"{btc['Close'].pct_change().iloc[-1] * 100:+.2f}%"
        ctx['Volume 24h'] = f"${btc['Volume'].iloc[-1]:,.0f}"
    else:
        ctx['BTC Price'] = ctx.get('BTC Price', 'N/A')
    try:
        fng = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5).json()
        if 'data' in fng and len(fng['data']) > 0:
            ctx['Fear & Greed'] = f"{fng['data'][0]['value']}/100 ({fng['data'][0]['value_classification']})"
    except:
        ctx['Fear & Greed'] = 'N/A'
    for tk, name in [('^GSPC', 'S&P 500'), ('GC=F', 'Gold'), ('DX-Y.NYB', 'DXY')]:
        ticker_data = []
        def _dl_tk():
            try:
                d = yf.download(tk, period='5d', progress=False)
                if not d.empty:
                    ticker_data.append(d)
            except:
                pass
        t = threading.Thread(target=_dl_tk, daemon=True)
        t.start()
        t.join(timeout=10)
        if ticker_data:
            df = ticker_data[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            v = df['Close'].iloc[-1]
            c = df['Close'].pct_change().iloc[-1] * 100
            ctx[name] = f"{v:,.2f} ({c:+.2f}%)"
        else:
            ctx[name] = 'N/A'
    ctx['Time'] = datetime.now().strftime('%H:%M:%S')
    return ctx


# ===================== PREDICTIONS =====================

@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest(_price, model_name, n_days):
    pipeline = load_chronos_pipeline(model_name)
    days, preds, actuals = [], [], []
    for i in range(n_days, 0, -1):
        train_end = len(_price) - i
        train_data = _price.iloc[:train_end]
        context = torch.tensor(train_data.values, dtype=torch.float32).squeeze().unsqueeze(0)
        forecast = pipeline.predict(context, prediction_length=1, num_samples=20)
        pred = float(np.quantile(forecast[0].numpy(), 0.5, axis=0)[0])
        days.append(_price.index[train_end])
        preds.append(pred)
        actuals.append(float(_price.iloc[train_end]))
    return pd.DataFrame({'Day': days, 'Prediction': preds, 'Actual': actuals})


@st.cache_data(ttl=3600, show_spinner=False)
def run_forecast(_price, model_name, horizon):
    pipeline = load_chronos_pipeline(model_name)
    context = torch.tensor(_price.values, dtype=torch.float32).squeeze().unsqueeze(0)
    forecast = pipeline.predict(context, prediction_length=horizon, num_samples=100)
    samples = forecast[0].numpy()
    median = np.median(samples, axis=0)
    p10 = np.percentile(samples, 10, axis=0)
    p90 = np.percentile(samples, 90, axis=0)
    return median, p10, p90


# ===================== UI =====================

def main():
    now = datetime.now()

    if 'lang' not in st.session_state:
        st.session_state['lang'] = 'es'

    L = LANG[st.session_state['lang']]

    with st.sidebar:
        st.markdown(f"### {L['controls']}")

        lang_opts = {'en': 'English', 'es': 'Espa\u00f1ol'}
        lang_sel = st.selectbox(
            L['language'],
            list(lang_opts.keys()),
            format_func=lambda k: lang_opts[k],
            index=list(lang_opts.keys()).index(st.session_state['lang']),
        )
        if lang_sel != st.session_state['lang']:
            st.session_state['lang'] = lang_sel
            st.rerun()

        st.markdown("---")

        modelo = st.selectbox(
            L['model'],
            list(MODELS.keys()),
            index=list(MODELS.keys()).index(DEFAULT_MODEL),
        )

        st.markdown(f"**{L['backtest']}**")
        bt_dias = st.slider(L['backtest_window'], 1, MAX_BACKTEST, 10, 1)

        st.divider()

        if st.button(L['refresh_data'], width='stretch', type="primary"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown(f"**{L['legend']}**")
        st.markdown(f"""
        - **{L['legend_blue']}**
        - **{L['legend_red']}**
        - **{L['legend_shaded']}**
        - **{L['legend_points']}**
        """)

    price = load_btc_data()
    if len(price) < 100:
        st.error(L['insufficient_data'])
        return

    last_date = price.index[-1]
    last_price = float(price.iloc[-1])

    st.title("BTC Forecast V4")
    st.markdown(f"<p style='color:#8b8fa3; margin-top:-10px;'>"
                f"{L['model']}: {modelo} | "
                f"{L['updated']}: {now.strftime('%Y-%m-%d %H:%M')}</p>",
                unsafe_allow_html=True)

    bt_df = run_backtest(price, modelo, bt_dias)
    forecast_data = run_forecast(price, modelo, MAX_FORECAST)
    median, p10, p90 = forecast_data
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=MAX_FORECAST, freq='D'
    )

    context = fetch_live_context()
    st.subheader(L['market_context'])
    cols = st.columns(len(context))
    for i, (k, v) in enumerate(context.items()):
        with cols[i]:
            cl = 'color: #52b788;' if k == 'Time' else ''
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{k}</div>'
                f'<div class="metric-value" style="{cl}">{v}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    tab_f, tab_b, tab_c, tab_l = st.tabs(
        [L['tab_forward'], L['tab_backtest'], L['tab_combined'], L['tab_live']]
    )

    # ----- TAB 1: FORWARD -----
    with tab_f:
        fwd_dias = st.slider(L['forecast_days'], 1, MAX_FORECAST, 10, 1, key='fwd')
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("1d", key='f1', width='stretch'): fwd_dias = 1
        with col_p2:
            if st.button("7d", key='f7', width='stretch'): fwd_dias = 7
        with col_p3:
            if st.button("30d", key='f30', width='stretch'): fwd_dias = 30
        st.subheader(L['forecast_n_days'].format(fwd_dias))

        fig_f = go.Figure()
        ctx = price.iloc[-min(90, len(price)):]
        fig_f.add_trace(go.Scatter(
            x=ctx.index, y=ctx.values, mode='lines', name='BTC History',
            line=dict(color='#457b9d', width=2),
        ))
        fig_f.add_trace(go.Scatter(
            x=future_dates[:fwd_dias], y=p90[:fwd_dias],
            mode='lines', line=dict(color='rgba(230,57,70,0)'), showlegend=False,
        ))
        fig_f.add_trace(go.Scatter(
            x=future_dates[:fwd_dias], y=median[:fwd_dias],
            mode='lines+markers', name=f'{modelo} (median)',
            line=dict(color='#e63946', width=3), marker=dict(size=5),
        ))
        fig_f.add_trace(go.Scatter(
            x=future_dates[:fwd_dias], y=p10[:fwd_dias],
            mode='lines', line=dict(color='rgba(230,57,70,0)'),
            fill='tonexty', fillcolor='rgba(230,57,70,0.12)',
            name='P10-P90',
        ))
        fig_f.add_vline(x=last_date, line_dash='dot', line_color='gray', opacity=0.5)
        fig_f.update_layout(
            template='plotly_dark', hovermode='x unified', height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title='Date'), yaxis=dict(title='BTC Price (USD)', tickformat='$,.0f'),
            legend=dict(orientation='h', y=1.02),
        )
        st.plotly_chart(fig_f, width='stretch')

        tbl = pd.DataFrame({
            L['day']: range(1, fwd_dias + 1),
            L['date']: future_dates[:fwd_dias].strftime('%Y-%m-%d'),
            L['forecast']: [f"${v:,.2f}" for v in median[:fwd_dias]],
            'P10': [f"${v:,.2f}" for v in p10[:fwd_dias]],
            'P90': [f"${v:,.2f}" for v in p90[:fwd_dias]],
            L['change_pct']: [f"{(v - last_price) / last_price * 100:+.2f}%" for v in median[:fwd_dias]],
        })
        st.dataframe(tbl, width='stretch', hide_index=True)

    # ----- TAB 2: BACKTEST -----
    with tab_b:
        st.subheader(L['backtest_n_days'].format(bt_dias))

        errors = abs(bt_df['Prediction'] - bt_df['Actual']) / bt_df['Actual'] * 100
        overall_mape = errors.mean()
        bt_df[L['error']] = errors.round(2)

        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=bt_df['Day'], y=bt_df['Actual'],
            mode='lines+markers', name=L['actual'],
            line=dict(color='#e63946', width=3), marker=dict(size=9),
        ))
        fig_b.add_trace(go.Scatter(
            x=bt_df['Day'], y=bt_df['Prediction'],
            mode='lines+markers', name=modelo,
            line=dict(color='#457b9d', width=3, dash='dash'), marker=dict(size=9),
        ))
        for _, row in bt_df.iterrows():
            e = row[L['error']]
            c = '#2a9d8f' if e < 2 else '#e9c46a' if e < 5 else '#e63946'
            fig_b.add_shape(type='line', x0=row['Day'], x1=row['Day'],
                             y0=row['Actual'], y1=row['Prediction'],
                             line=dict(color=c, width=2, dash='dot'))
        fig_b.update_layout(
            template='plotly_dark', hovermode='x unified', height=420,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title='Date'), yaxis=dict(title='BTC Price (USD)', tickformat='$,.0f'),
            legend=dict(orientation='h', y=1.02),
        )
        st.plotly_chart(fig_b, width='stretch')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(L['mape'], f"{overall_mape:.2f}%")
        c2.metric(L['hits'], f"{(errors < 2).sum()}/{bt_dias}")
        c3.metric(L['medium'], f"{((errors >= 2) & (errors < 5)).sum()}/{bt_dias}")
        c4.metric(L['misses'], f"{(errors >= 5).sum()}/{bt_dias}")

        st.subheader(L['daily_breakdown'])
        tbl_b = bt_df.copy()
        tbl_b['Day'] = tbl_b['Day'].dt.strftime('%Y-%m-%d')
        tbl_b[L['prediction']] = tbl_b['Prediction'].apply(lambda v: f"${v:,.2f}")
        tbl_b[L['actual']] = tbl_b['Actual'].apply(lambda v: f"${v:,.2f}")
        st.dataframe(tbl_b, width='stretch', hide_index=True)

    # ----- TAB 3: COMBINED -----
    with tab_c:
        com_dias = st.slider(L['forecast_days'], 1, MAX_FORECAST, 10, 1, key='com')
        st.subheader(L['combined_title'].format(bt_dias, com_dias))

        fig_c = go.Figure()
        ctx = price.iloc[-min(150, len(price)):]
        fig_c.add_trace(go.Scatter(
            x=ctx.index, y=ctx.values, mode='lines', name='BTC History',
            line=dict(color='#457b9d', width=1.5),
        ))
        for _, row in bt_df.iterrows():
            e = abs(row['Prediction'] - row['Actual']) / row['Actual'] * 100
            c = '#2a9d8f' if e < 2 else '#e9c46a' if e < 5 else '#e63946'
            fig_c.add_trace(go.Scatter(
                x=[row['Day']], y=[row['Prediction']],
                mode='markers', marker=dict(size=10, color=c, symbol='x'),
                showlegend=False,
                hovertemplate=(f"<b>{row['Day'].strftime('%Y-%m-%d')}</b><br>"
                               f"Pred: ${row['Prediction']:,.2f}<br>"
                               f"Actual: ${row['Actual']:,.2f}<br>"
                               f"Error: {e:.2f}%<extra></extra>"),
            ))
        fig_c.add_trace(go.Scatter(
            x=future_dates[:com_dias], y=p90[:com_dias],
            mode='lines', line=dict(color='rgba(230,57,70,0)'), showlegend=False,
        ))
        fig_c.add_trace(go.Scatter(
            x=future_dates[:com_dias], y=median[:com_dias],
            mode='lines+markers', name=f'Forecast {com_dias}d (median)',
            line=dict(color='#e63946', width=3), marker=dict(size=5),
        ))
        fig_c.add_trace(go.Scatter(
            x=future_dates[:com_dias], y=p10[:com_dias],
            mode='lines', line=dict(color='rgba(230,57,70,0)'),
            fill='tonexty', fillcolor='rgba(230,57,70,0.12)',
            name=f'P10-P90 ({com_dias}d)',
        ))
        fig_c.add_vline(x=last_date, line_dash='dot', line_color='gray', opacity=0.5)
        fig_c.update_layout(
            template='plotly_dark', hovermode='x unified', height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title='Date'), yaxis=dict(title='BTC Price (USD)', tickformat='$,.0f'),
            legend=dict(orientation='h', y=1.02),
        )
        st.plotly_chart(fig_c, width='stretch')

        c1, c2, c3 = st.columns(3)
        c1.metric(L['current_price'], f"${last_price:,.2f}", context.get('24h Change', ''))
        c2.metric(L['backtest_mape'].format(bt_dias), f"{overall_mape:.2f}%")
        cambio_f = (median[com_dias - 1] - last_price) / last_price * 100
        c3.metric(L['forecast_n'].format(com_dias), f"${median[com_dias - 1]:,.2f}", f"{cambio_f:+.2f}%")

    # ----- TAB 4: LIVE -----
    with tab_l:
        count = st_autorefresh(interval=60000, key="live")

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
            f'<span class="live-dot"></span>'
            f'<h3 style="margin:0;">LIVE &mdash; {modelo}</h3>'
            f'<span style="color:#555;font-size:12px;margin-left:auto;">'
            f'Refresh #{count} | 60s</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        intra = fetch_intraday_data()
        df_15m, df_1h = intra['15m'], intra['1h']

        live_dias_slider = st.slider(L['forecast_days'], 1, 3, 1, 1, key='live_fcast')

        if df_15m is not None and df_1h is not None and not df_15m.empty:
            today = datetime.now().date()
            today_15m = df_15m[df_15m.index.date == today]
            today_1h = df_1h[df_1h.index.date == today]

            if not today_15m.empty:
                o = today_1h['Open'].iloc[0] if not today_1h.empty else today_15m['Open'].iloc[0]
                h = today_15m['High'].max()
                l = today_15m['Low'].min()
                c = today_15m['Close'].iloc[-1]
                v = today_15m['Volume'].sum()
                rng = h - l
                rng_pct = rng / l * 100 if l > 0 else 0
                chg = (c - o) / o * 100 if o > 0 else 0
                vwap_val = ((today_15m['Close'] * today_15m['Volume']).sum()
                            / today_15m['Volume'].sum()) if today_15m['Volume'].sum() > 0 else c
                vol_1h = compute_hourly_volatility(today_1h) if not today_1h.empty else 0

                arrow = '\u25b2' if chg >= 0 else '\u25bc'
                color = '#26a69a' if chg >= 0 else '#ef5350'

                st.markdown(
                    f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px;">'
                    f'<div class="metric-card" style="flex:1;min-width:140px;">'
                    f'<div class="metric-label">Price</div>'
                    f'<div class="metric-value" style="color:{color}">{arrow} ${c:,.2f}</div>'
                    f'<div style="font-size:13px;color:{color};">{chg:+.2f}% today</div>'
                    f'</div>'
                    f'<div class="metric-card" style="flex:1;min-width:120px;">'
                    f'<div class="metric-label">Daily Range</div>'
                    f'<div class="metric-value">${rng:,.0f}</div>'
                    f'<div style="font-size:13px;color:#8b8fa3;">{rng_pct:.2f}%</div>'
                    f'</div>'
                    f'<div class="metric-card" style="flex:1;min-width:120px;">'
                    f'<div class="metric-label">Volume</div>'
                    f'<div class="metric-value">${v:,.0f}</div>'
                    f'</div>'
                    f'<div class="metric-card" style="flex:1;min-width:120px;">'
                    f'<div class="metric-label">VWAP</div>'
                    f'<div class="metric-value">${vwap_val:,.2f}</div>'
                    f'<div style="font-size:13px;color:#8b8fa3;">{(c - vwap_val) / vwap_val * 100:+.2f}%</div>'
                    f'</div>'
                    f'<div class="metric-card" style="flex:1;min-width:120px;">'
                    f'<div class="metric-label">Volatility (1h)</div>'
                    f'<div class="metric-value">{vol_1h:.2f}%</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                col_o, col_h, col_l = st.columns(3)
                col_o.metric('Open', f'${o:,.2f}')
                col_h.metric('High', f'${h:,.2f}')
                col_l.metric('Low', f'${l:,.2f}')

                two_days = today - timedelta(days=2)
                chart_data = df_15m[df_15m.index.date >= two_days].copy()

                fig_l = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       vertical_spacing=0.02, row_heights=[0.7, 0.3])
                fig_l.add_trace(go.Candlestick(
                    x=chart_data.index, open=chart_data['Open'], high=chart_data['High'],
                    low=chart_data['Low'], close=chart_data['Close'], name='BTC',
                ), row=1, col=1)
                bar_colors = ['#26a69a' if chart_data['Close'].iloc[i] >= chart_data['Open'].iloc[i]
                              else '#ef5350' for i in range(len(chart_data))]
                fig_l.add_trace(go.Bar(
                    x=chart_data.index, y=chart_data['Volume'], name='Volume',
                    marker_color=bar_colors, opacity=0.4,
                ), row=2, col=1)
                fig_l.add_hline(y=vwap_val, line_dash='dash', line_color='#ffd700', opacity=0.7,
                                annotation_text=f'VWAP ${vwap_val:,.0f}',
                                annotation_position='top left', row=1, col=1)
                fdates = future_dates[:live_dias_slider]
                fig_l.add_trace(go.Scatter(
                    x=fdates[:live_dias_slider], y=p90[:live_dias_slider],
                    mode='lines', line=dict(color='rgba(230,57,70,0)'), showlegend=False,
                ), row=1, col=1)
                fig_l.add_trace(go.Scatter(
                    x=fdates[:live_dias_slider], y=median[:live_dias_slider],
                    mode='lines+markers', name=modelo,
                    line=dict(color='#e63946', width=2), marker=dict(size=6),
                ), row=1, col=1)
                fig_l.add_trace(go.Scatter(
                    x=fdates[:live_dias_slider], y=p10[:live_dias_slider],
                    mode='lines', line=dict(color='rgba(230,57,70,0)'),
                    fill='tonexty', fillcolor='rgba(230,57,70,0.12)',
                    name='P10-P90',
                ), row=1, col=1)
                fig_l.update_layout(
                    template='plotly_dark', hovermode='x unified', height=480,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation='h', y=1.02),
                )
                fig_l.update_xaxes(title='', row=2, col=1)
                fig_l.update_yaxes(title='Price (USD)', row=1, col=1, tickformat='$,.0f')
                fig_l.update_yaxes(title='Volume', row=2, col=1)
                st.plotly_chart(fig_l, width='stretch')

                if not today_1h.empty:
                    st.markdown(f"<p style='color:#f0f0f0;font-weight:bold;margin:8px 0 4px;'>"
                                f"Hourly Detail | Last {min(8, len(today_1h))} hours</p>",
                                unsafe_allow_html=True)
                    last_h = today_1h.iloc[-8:] if len(today_1h) > 8 else today_1h
                    chg_col = []
                    for i in range(len(last_h)):
                        chg = ((last_h['Close'].iloc[i] - last_h['Open'].iloc[i])
                               / last_h['Open'].iloc[i] * 100)
                        chg_col.append(f'{chg:+.2f}')
                    tbl_h = pd.DataFrame({
                        'Time': last_h.index.strftime('%H:%M'),
                        'Open': [f'${v:,.0f}' for v in last_h['Open']],
                        'High': [f'${v:,.0f}' for v in last_h['High']],
                        'Low': [f'${v:,.0f}' for v in last_h['Low']],
                        'Close': [f'${v:,.0f}' for v in last_h['Close']],
                        'Vol': [f'{v:,.0f}' for v in last_h['Volume']],
                        'Chg%': chg_col,
                    })
                    st.dataframe(tbl_h, width='stretch', hide_index=True)

                st.markdown(f"<p style='color:#f0f0f0;font-weight:bold;margin:12px 0 4px;'>"
                            f"3+{live_dias_slider} Day View | Actual + Forecast</p>",
                            unsafe_allow_html=True)
                last3 = price.iloc[-3:]
                last3_dates = last3.index
                pred_dates = future_dates[:live_dias_slider]
                fig_bars = go.Figure()
                fig_bars.add_trace(go.Bar(
                    x=last3_dates, y=last3.values,
                    name='Actual', marker_color='#457b9d', width=0.5,
                ))
                fig_bars.add_trace(go.Bar(
                    x=pred_dates, y=median[:live_dias_slider],
                    name=f'{modelo}', marker_color='#e63946', width=0.5,
                ))
                fig_bars.add_trace(go.Scatter(
                    x=pred_dates, y=p90[:live_dias_slider],
                    mode='lines', line=dict(color='rgba(230,57,70,0)'), showlegend=False,
                ))
                fig_bars.add_trace(go.Scatter(
                    x=pred_dates, y=p10[:live_dias_slider],
                    mode='lines', line=dict(color='rgba(230,57,70,0)'),
                    fill='tonexty', fillcolor='rgba(230,57,70,0.12)',
                    name='P10-P90',
                ))
                fig_bars.update_layout(
                    template='plotly_dark', hovermode='x unified', height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    barmode='group', legend=dict(orientation='h', y=1.02),
                    yaxis=dict(title='BTC (USD)', tickformat='$,.0f'),
                )
                st.plotly_chart(fig_bars, width='stretch')

                st.markdown(
                    f"<p style='color:#555;font-size:12px;margin-top:10px;'>"
                    f"Forecast {live_dias_slider}d: ${median[live_dias_slider - 1]:,.0f} "
                    f"({(median[live_dias_slider - 1] - c) / c * 100:+.1f}%) &nbsp;|&nbsp; "
                    f"P10: ${p10[live_dias_slider - 1]:,.0f} &nbsp; "
                    f"P90: ${p90[live_dias_slider - 1]:,.0f}</p>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning(L['waiting_data'])

    st.divider()
    st.markdown(
        f"<p style='color:#555; font-size:11px; text-align:center;'>"
        f"BTC Forecast V4 | {modelo} | Data: Yahoo Finance + CoinGecko | "
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
