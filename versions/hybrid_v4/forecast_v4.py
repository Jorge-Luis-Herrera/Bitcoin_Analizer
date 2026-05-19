"""
Forecast V4 — Chronos forward prediction.
Projects N days ahead using historical price data and displays confidence bands.
"""

import os, sys, numpy as np, pandas as pd, yfinance as yf, torch
import requests
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

HISTORY_DAYS = 1000


def run_chronos_probabilistic(train_data, horizon, n_samples=100):
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32,
    )
    context = torch.tensor(train_data.values, dtype=torch.float32).squeeze().unsqueeze(0)
    forecast = pipeline.predict(context, prediction_length=horizon, num_samples=n_samples)
    samples = forecast[0].numpy()
    median = np.median(samples, axis=0)
    p10 = np.percentile(samples, 10, axis=0)
    p90 = np.percentile(samples, 90, axis=0)
    return median, p10, p90, samples


def get_market_snapshot():
    snapshot = {}
    btc = yf.download('BTC-USD', period='5d', progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.droplevel(1)
    if len(btc) > 0:
        snapshot['Current Price'] = f"${btc['Close'].iloc[-1]:,.2f}"
        snapshot['24h Change'] = f"{btc['Close'].pct_change().iloc[-1] * 100:+.2f}%"
        snapshot['Volume'] = f"${btc['Volume'].iloc[-1]:,.0f}"

    try:
        fng = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5).json()
        if 'data' in fng and len(fng['data']) > 0:
            snapshot['Fear & Greed'] = f"{fng['data'][0]['value']}/100 ({fng['data'][0]['value_classification']})"
    except:
        snapshot['Fear & Greed'] = 'N/A'

    tickers = {'^GSPC': 'S&P 500', 'GC=F': 'Gold', 'DX-Y.NYB': 'DXY'}
    for tk, name in tickers.items():
        try:
            df = yf.download(tk, period='5d', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if len(df) > 0:
                val = df['Close'].iloc[-1]
                chg = df['Close'].pct_change().iloc[-1] * 100
                snapshot[name] = f"{val:,.2f} ({chg:+.2f}%)"
        except:
            pass

    return snapshot


def forecast():
    import argparse
    parser = argparse.ArgumentParser(description='Chronos V4 Forecast')
    parser.add_argument('--dias', type=int, default=10,
                        help='Days to forecast (default: 10)')
    parser.add_argument('--muestras', type=int, default=100,
                        help='Number of probabilistic samples (default: 100)')
    args = parser.parse_args()

    HORIZON = args.dias
    N_SAMPLES = args.muestras

    print('-' * 55)
    print(f'  Forecast V4  |  Chronos  |  {HORIZON} days ahead')
    print('-' * 55)

    end = datetime.now()
    start = end - timedelta(days=HISTORY_DAYS)

    print('\n  Downloading data...')
    btc = yf.download('BTC-USD', start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'), progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.droplevel(1)
    price = btc['Close']
    print(f'  History: {len(price)} days')

    print(f'\n  Running Chronos ({N_SAMPLES} samples, {HORIZON} days)...')
    median, p10, p90, samples = run_chronos_probabilistic(price, HORIZON, N_SAMPLES)

    last_date = price.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                  periods=HORIZON, freq='D')
    last_price = price.iloc[-1]

    print(f'\n  {"Day":>4s} {"Date":>12s} {"Forecast":>14s} {"P10":>12s} {"P90":>12s} {"Change%":>8s}')
    print(f'  {"-" * 62}')
    print(f'  {0:>4d} {last_date.strftime("%Y-%m-%d"):>12s} '
          f'${last_price:>8,.2f} {"":>12s} {"":>12s} {"--":>8s}')
    for i in range(HORIZON):
        chg = (median[i] - last_price) / last_price * 100
        print(f'  {i+1:>4d} {future_dates[i].strftime("%Y-%m-%d"):>12s} '
              f'${median[i]:>8,.2f}  ${p10[i]:>8,.2f}  ${p90[i]:>8,.2f}  {chg:>+7.2f}%')

    print(f'\n  {"=" * 50}')
    print('  CURRENT MARKET CONTEXT')
    print(f'  {"=" * 50}')
    snapshot = get_market_snapshot()
    for k, v in snapshot.items():
        print(f'    {k:20s} -> {v}')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10),
                                     gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Forecast V4 — Chronos: {HORIZON} days ahead',
                 fontsize=15, fontweight='bold', y=0.98)

    ctx_days = min(90, len(price))
    ctx = price.iloc[-ctx_days:]

    ax1.plot(ctx.index, ctx.values, color='#1a1a2e', linewidth=1.5, label='BTC History')
    ax1.plot(future_dates, median, color='#e63946', linewidth=2.5, label='Chronos (median)')
    ax1.fill_between(future_dates, p10, p90, color='#e63946', alpha=0.15,
                      label='P10-P90')
    ax1.axvline(x=last_date, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('BTC Price (USD)', fontsize=11)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    last_30 = price.iloc[-30:]
    ax2.bar(last_30.index, last_30.values, color='#457b9d', alpha=0.7, width=0.8)
    ax2.set_ylabel('BTC (USD)', fontsize=11)
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    cell_text = [[k, v] for k, v in snapshot.items()]
    table = ax2.table(cellText=cell_text, colLabels=['Indicator', 'Value'],
                       loc='upper right', fontsize=8,
                       cellLoc='left', bbox=[0.65, 0.55, 0.33, 0.40])
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, 'forecast_v4.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\n  Chart: {path}')

    results = pd.DataFrame({
        'Day': range(1, HORIZON + 1),
        'Date': future_dates.strftime('%Y-%m-%d'),
        'Forecast': median,
        'P10': p10,
        'P90': p90,
    })
    csv_path = os.path.join(RESULTS_DIR, 'forecast_v4.csv')
    results.to_csv(csv_path, index=False)
    print(f'  CSV: {csv_path}')
    print(f'\n  Forecast complete.\n')


if __name__ == '__main__':
    forecast()
