"""
Historical Backtest Study — V4
Evaluates Chronos variants and baseline models across multiple
independent test windows dating back to 2024.
"""

import os, sys, numpy as np, pandas as pd, yfinance as yf, torch
import warnings, time, json
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error
from itertools import product

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_TRAIN = 500
STEP = 30
HISTORY = 1000


def load_chronos(name="amazon/chronos-t5-small"):
    from chronos import ChronosPipeline
    return ChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)


def predict_chronos(pipeline, train_data, horizon=1, n_samples=20):
    context = torch.tensor(train_data.values, dtype=torch.float32).squeeze().unsqueeze(0)
    forecast = pipeline.predict(context, prediction_length=horizon, num_samples=n_samples)
    return np.quantile(forecast[0].numpy(), 0.5, axis=0)


def predict_arima(train_data, horizon=1):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train_data.values, order=(5,1,0))
    fitted = model.fit()
    return fitted.forecast(horizon)


def predict_naive(train_data, horizon=1):
    return np.full(horizon, train_data.iloc[-1])


def predict_timesfm(train_data, horizon=1):
    import timesfm
    try:
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu", per_core_batch_size=32, horizon_len=horizon,
                context_len=512, input_patch_len=32, output_patch_len=128,
                num_layers=20, model_dims=1280,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
            )
        )
        context = train_data.values[-512:]
        forecast = tfm.forecast([context], freq=[0])
        return forecast[0][0][:horizon]
    except:
        return None


def predict_moirai(train_data, horizon=1):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.evaluation import make_evaluation_predictions
    try:
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small"),
            prediction_length=horizon, context_length=min(512, len(train_data)),
            patch_size="auto", num_samples=100, target_dim=1,
            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=32)
        df_m = pd.DataFrame({'Close': train_data.values}, index=train_data.index)
        df_m.index = pd.DatetimeIndex(df_m.index)
        df_m = df_m.asfreq('D').ffill().dropna()
        ds = PandasDataset(df_m, target="Close")
        it, _ = make_evaluation_predictions(dataset=ds, predictor=predictor, num_samples=100)
        forecast = list(it)[0]
        return forecast.quantile(0.5)[:horizon]
    except:
        return None


def run_study():
    print('\n' + '=' * 65)
    print('  HISTORICAL BACKTEST STUDY  |  Multi-Model Comparison')
    print('=' * 65)

    end = datetime.now()
    start = end - timedelta(days=HISTORY + 365)
    btc = yf.download('BTC-USD', start='2020-01-01',
                       end=end.strftime('%Y-%m-%d'), progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.droplevel(1)
    price = btc['Close']
    price = price[price.index >= pd.Timestamp(start)]
    price = price[price.index <= pd.Timestamp(end)]

    if len(price) == 0:
        print('Error: no data downloaded. Aborting.')
        return
    print(f'\n  Total data: {len(price)} days ({price.index[0].date()} -> {price.index[-1].date()})')

    test_points = []
    for i in range(len(price) - MIN_TRAIN - 1, 0, -STEP):
        if i < MIN_TRAIN or len(test_points) >= 50:
            break
        test_points.append(i)

    test_points = test_points[::-1]
    print(f'  Test windows: {len(test_points)} (every {STEP} days)')
    print(f'     {price.index[test_points[0]].date()} -> {price.index[test_points[-1]].date()}\n')

    models = {
        'Chronos-Small': 'amazon/chronos-t5-small',
        'Chronos-Tiny': 'amazon/chronos-t5-tiny',
        'Chronos-Base': 'amazon/chronos-t5-base',
    }

    pipelines = {}
    print('  Loading Chronos models...')
    for name, hf_name in models.items():
        print(f'    -> {name}...', end=' ', flush=True)
        pipelines[name] = load_chronos(hf_name)
        print('done')

    print('')
    results = {name: [] for name in list(models.keys()) + ['ARIMA', 'Naive', 'TimesFM', 'MOIRAI']}
    times = {name: [] for name in results}

    for idx, tp in enumerate(test_points):
        train_data = price.iloc[tp - MIN_TRAIN:tp]
        actual = float(price.iloc[tp])
        pct = (idx + 1) / len(test_points) * 100

        print(f'\r  Progress: {pct:.0f}% | {price.index[tp].date()} -> ${actual:,.0f}', end='', flush=True)

        for name, pipeline in pipelines.items():
            t0 = time.time()
            try:
                pred = predict_chronos(pipeline, train_data)
                err = abs(pred[0] - actual) / actual * 100
                results[name].append(err)
                times[name].append(time.time() - t0)
            except:
                results[name].append(np.nan)
                times[name].append(0)

        t0 = time.time()
        try:
            pred = predict_arima(train_data)
            err = abs(pred[0] - actual) / actual * 100
            results['ARIMA'].append(err)
        except:
            results['ARIMA'].append(np.nan)
        times['ARIMA'].append(time.time() - t0)

        t0 = time.time()
        pred = predict_naive(train_data)
        err = abs(pred[0] - actual) / actual * 100
        results['Naive'].append(err)
        times['Naive'].append(time.time() - t0)

        if 'TimesFM' in results and idx % 3 == 0:
            t0 = time.time()
            try:
                pred = predict_timesfm(train_data)
                if pred is not None:
                    err = abs(pred[0] - actual) / actual * 100
                    results['TimesFM'].append(err)
                else:
                    results['TimesFM'].append(np.nan)
            except:
                results['TimesFM'].append(np.nan)
            times['TimesFM'].append(time.time() - t0)
        elif 'TimesFM' in results:
            results['TimesFM'].append(np.nan)
            times['TimesFM'].append(0)

        if 'MOIRAI' in results and idx % 3 == 0:
            t0 = time.time()
            try:
                pred = predict_moirai(train_data)
                if pred is not None:
                    err = abs(pred[0] - actual) / actual * 100
                    results['MOIRAI'].append(err)
                else:
                    results['MOIRAI'].append(np.nan)
            except:
                results['MOIRAI'].append(np.nan)
            times['MOIRAI'].append(time.time() - t0)
        elif 'MOIRAI' in results:
            results['MOIRAI'].append(np.nan)
            times['MOIRAI'].append(0)

    print('\n')

    print('\n' + '=' * 65)
    print('  RESULTS')
    print('=' * 65)
    print(f'  {"Model":25s} {"MAPE":>8s} {"Std":>8s} {"Min":>8s} {"Max":>8s} {"Time":>8s}')
    print(f'  {"-" * 60}')

    summary = []
    for name in results:
        vals = [v for v in results[name] if not np.isnan(v)]
        if len(vals) > 0:
            mape = np.mean(vals)
            std = np.std(vals)
            min_v = np.min(vals)
            max_v = np.max(vals)
            t_avg = np.mean(times[name]) if times[name] else 0
            print(f'  {name:25s} {mape:>7.2f}% {std:>7.2f}% {min_v:>7.2f}% {max_v:>7.2f}% {t_avg:>7.3f}s')
            summary.append({'Model': name, 'MAPE': mape, 'Std': std,
                           'Min': min_v, 'Max': max_v, 'Samples': len(vals)})
        else:
            print(f'  {name:25s} {"N/A":>8s}')

    df_results = pd.DataFrame(summary).sort_values('MAPE')
    csv_path = os.path.join(RESULTS_DIR, 'estudio_backtest_modelos.csv')
    df_results.to_csv(csv_path, index=False)
    print(f'\n  CSV: {csv_path}')

    naive_mape = df_results[df_results['Model'] == 'Naive']['MAPE'].values[0]
    print(f'\n  IMPROVEMENT VS NAIVE ({naive_mape:.2f}%):')
    for _, row in df_results.iterrows():
        if row['Model'] != 'Naive' and row['Samples'] > 5:
            mejora = (naive_mape - row['MAPE']) / naive_mape * 100
            print(f'    {row["Model"]:25s} {mejora:+.1f}%')

    print(f'\n  MAPE BY YEAR (Chronos-Small):')
    years = {}
    for idx, tp in enumerate(test_points):
        year = price.index[tp].year
        if year not in years:
            years[year] = []
        years[year].append(results['Chronos-Small'][idx])
    for year in sorted(years.keys()):
        vals = [v for v in years[year] if not np.isnan(v)]
        if vals:
            print(f'    {year}: {np.mean(vals):.2f}% ({len(vals)} samples)')

    print(f'\n  Study complete.\n')


if __name__ == '__main__':
    run_study()
