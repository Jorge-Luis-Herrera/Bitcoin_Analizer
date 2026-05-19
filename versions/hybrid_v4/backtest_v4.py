"""
Backtest V4 — Chronos walk-forward validation.
Predicts the last N days using only data available before each prediction point.
"""

import os, numpy as np, pandas as pd, yfinance as yf, torch
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

HISTORY_DAYS = 1000
BACKTEST_DAYS = 10


def run_chronos(train_data, horizon=1):
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32,
    )
    context = torch.tensor(train_data.values, dtype=torch.float32).squeeze().unsqueeze(0)
    forecast = pipeline.predict(context, prediction_length=horizon, num_samples=20)
    return np.quantile(forecast[0].numpy(), 0.5, axis=0)


def backtest():
    print('-' * 55)
    print('  Backtest V4  |  Chronos  |  Walk-Forward Validation')
    print('-' * 55)

    end = datetime.now()
    start = end - timedelta(days=HISTORY_DAYS)

    btc = yf.download('BTC-USD', start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'), progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.droplevel(1)
    price = btc['Close']
    print(f'\n  Data: {len(price)} days downloaded')

    days = []
    preds = []
    actuals = []
    errors = []

    for i in range(BACKTEST_DAYS, 0, -1):
        train_end = len(price) - i
        train_data = price.iloc[:train_end]
        actual_val = price.iloc[train_end]

        pred_val = run_chronos(train_data, horizon=1)[0]

        day_label = price.index[train_end].strftime('%Y-%m-%d')
        error_pct = abs(pred_val - actual_val) / actual_val * 100

        days.append(day_label)
        preds.append(pred_val)
        actuals.append(actual_val)
        errors.append(error_pct)

        print(f'    {day_label}  |  pred: ${pred_val:>8,.2f}  actual: ${actual_val:>8,.2f}  '
              f'error: {error_pct:>5.2f}%')

    overall_mape = np.mean(errors)
    print(f'\n  {"=" * 45}')
    print(f'  Average MAPE ({BACKTEST_DAYS} days): {overall_mape:.2f}%')
    print(f'  {"=" * 45}')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                     gridspec_kw={'height_ratios': [2.5, 1]})
    fig.suptitle('Backtest V4 — Chronos 1-day Walk-Forward',
                 fontsize=14, fontweight='bold', y=0.98)

    x = range(len(days))
    ax1.plot(x, actuals, 'o-', color='#e63946', linewidth=2.5, label='Actual', markersize=8)
    ax1.plot(x, preds, 's--', color='#457b9d', linewidth=2.5, label='Chronos', markersize=8)
    for i in range(len(days)):
        err = errors[i]
        color = '#2a9d8f' if err < 3 else ('#e9c46a' if err < 6 else '#e63946')
        ax1.plot([i, i], [actuals[i], preds[i]], color=color, linewidth=2, alpha=0.7)
    ax1.set_ylabel('BTC Price (USD)', fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(days)))
    ax1.set_xticklabels(days, rotation=45, ha='right')

    colors = ['#2a9d8f' if e < 3 else ('#e9c46a' if e < 6 else '#e63946') for e in errors]
    bars = ax2.bar(range(len(errors)), errors, color=colors)
    for i, (bar, err) in enumerate(zip(bars, errors)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{err:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax2.axhline(y=overall_mape, color='red', linestyle='--', linewidth=1.5,
                label=f'Average MAPE: {overall_mape:.2f}%')
    ax2.set_ylabel('Error (%)', fontsize=11)
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(len(days)))
    ax2.set_xticklabels(days, rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, 'backtest_v4.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\n  Chart: {path}')

    results = pd.DataFrame({
        'Day': days, 'Prediction': preds, 'Actual': actuals, 'Error %': errors
    })
    csv_path = os.path.join(RESULTS_DIR, 'backtest_v4.csv')
    results.to_csv(csv_path, index=False)
    print('  CSV: ' + csv_path)
    print('\n  Backtest complete.\n')

    return overall_mape


if __name__ == '__main__':
    backtest()
