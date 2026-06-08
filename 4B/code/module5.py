import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# To display Chinese correctly in matplotlib
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# Load test data and model
df_test = pd.read_csv('../results/test_data.csv')
factor_cols = ['Reversal', 'Liquidity', 'MoneyFlow', 'Value']

df_test = df_test.dropna(subset=factor_cols + ['next_excess_ret']).copy()

# Standardize test data cross-sectionally per day
def z_score(series):
    # handle 0 variance case
    if series.std() == 0:
        return series - series.mean()
    return (series - series.mean()) / series.std()

df_test[factor_cols] = df_test.groupby('trade_date')[factor_cols].transform(z_score)
df_test = df_test.dropna(subset=factor_cols).copy()

# Load model and predict
model = joblib.load('../results/lgbm_model.pkl')
df_test['pred_excess_ret'] = model.predict(df_test[factor_cols])

# Sort and select Top 5 per day
# Sort by pred_excess_ret descending
df_test = df_test.sort_values(['trade_date', 'pred_excess_ret'], ascending=[True, False])

# Get top 5 per day
top5_df = df_test.groupby('trade_date').head(5).copy()

# Backtest parameters
initial_capital = 1_000_000
transaction_cost = 0.001  # 0.1% single side

# Realized return
top5_df['next_ret'] = top5_df['next_excess_ret'] + 0.0001

dates = sorted(top5_df['trade_date'].unique())

port_rets = []
port_dates = []

prev_holdings = set()

for date in dates:
    daily_stocks = top5_df[top5_df['trade_date'] == date]
    current_holdings = set(daily_stocks['ts_code'])

    ret_mean = daily_stocks['next_ret'].mean()

    sell_count = len(prev_holdings - current_holdings)
    buy_count = len(current_holdings - prev_holdings)

    turnover_cost = (sell_count * 0.2 + buy_count * 0.2) * transaction_cost

    if len(prev_holdings) == 0:
        turnover_cost = 1.0 * transaction_cost

    net_ret = ret_mean - turnover_cost

    port_rets.append(net_ret)
    port_dates.append(date)

    prev_holdings = current_holdings

index_df = pd.read_parquet('../data/index_daily.parquet')
index_df['trade_date'] = pd.to_datetime(index_df['trade_date']).dt.strftime('%Y%m%d')

res_df = pd.DataFrame({
    'trade_date': port_dates,
    'port_ret': port_rets
})

date_series = pd.Series(dates)
res_df['realized_date'] = date_series.shift(-1)
# Handle the last date by just dropping it, as we can't observe the next_ret realized
res_df = res_df.dropna()
res_df['realized_date'] = res_df['realized_date'].astype(int).astype(str)

backtest_df = pd.merge(res_df, index_df[['trade_date', 'pct_chg']], left_on='realized_date', right_on='trade_date', how='inner')
backtest_df['index_ret'] = backtest_df['pct_chg'].astype(float) / 100.0

backtest_df['port_cum'] = (1 + backtest_df['port_ret']).cumprod()
backtest_df['index_cum'] = (1 + backtest_df['index_ret']).cumprod()

plt.figure(figsize=(10, 5))
plt.plot(pd.to_datetime(backtest_df['realized_date']), backtest_df['port_cum'], label='策略净值 (Strategy)')
plt.plot(pd.to_datetime(backtest_df['realized_date']), backtest_df['index_cum'], label='沪深300 (HS300)')
plt.title('中频短线策略样本外回测 (Out-of-sample Backtest)')
plt.xlabel('日期 (Date)')
plt.ylabel('累计收益 (Cumulative Return)')
plt.legend()
plt.grid()
plt.savefig('../results/backtest_curve.png')

cum_ret = backtest_df['port_cum'].iloc[-1] - 1
ann_ret = (1 + cum_ret) ** (252 / len(backtest_df)) - 1
daily_rets = backtest_df['port_ret']

max_drawdown = ((backtest_df['port_cum'].cummax() - backtest_df['port_cum']) / backtest_df['port_cum'].cummax()).max()
win_rate = (daily_rets > 0).mean()
sharpe_ratio = daily_rets.mean() / daily_rets.std() * np.sqrt(252)

index_ann_ret = (1 + backtest_df['index_cum'].iloc[-1] - 1) ** (252 / len(backtest_df)) - 1
excess_ann_ret = ann_ret - index_ann_ret

metrics = pd.DataFrame({
    '指标 (Metric)': ['累计收益率 (Cumulative Return)', '年化收益率 (Annualized Return)',
                    '最大回撤 (Max Drawdown)', '日胜率 (Daily Win Rate)',
                    '夏普比率 (Sharpe Ratio)', '超额年化收益 (Excess Ann. Return)'],
    '数值 (Value)': [f"{cum_ret*100:.2f}%", f"{ann_ret*100:.2f}%",
                   f"{max_drawdown*100:.2f}%", f"{win_rate*100:.2f}%",
                   f"{sharpe_ratio:.2f}", f"{excess_ann_ret*100:.2f}%"]
})

metrics.to_csv('../results/backtest_metrics.csv', index=False)
top5_df[['trade_date', 'ts_code', 'pred_excess_ret']].to_csv('../results/daily_top5_picks.csv', index=False)

print("\n回测指标 (Backtest Metrics):")
print(metrics)

print("Module 5 complete.")
