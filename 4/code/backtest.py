import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# User requested Chinese fonts:
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

test_df = pd.read_csv("4/code/test_df.csv")
with open('4/code/valid_factors.txt', 'r') as f:
    valid_factors = f.read().split(',')
weights_df = pd.read_csv("4/code/factor_weights.csv")
weights = dict(zip(weights_df['Factor'], weights_df['Weight']))

def standardize(s):
    if s.std() == 0: return s - s.mean()
    return (s - s.mean()) / s.std()

INIT_CAPITAL = 1000000
FEE = 0.003
TOP_N = 3

# Standardize test data cross-sectionally
std_test_list = []
for month in test_df['year_month_str'].unique():
    sub = test_df[test_df['year_month_str'] == month].copy()
    for f in valid_factors:
        sub[f'{f}_std'] = standardize(sub[f])
    std_test_list.append(sub)
test_df = pd.concat(std_test_list)

std_cols = [f'{f}_std' for f in valid_factors]

def score_func(row):
    return sum(weights[f] * row[f'{f}_std'] for f in valid_factors)

nav = [INIT_CAPITAL]
history = []
months = sorted(test_df['year_month_str'].unique())

for month in months:
    sub = test_df[test_df['year_month_str'] == month].copy()
    sub['score'] = sub.apply(score_func, axis=1)
    sub = sub.sort_values('score', ascending=False)

    # Top 3 stocks
    top_stocks = sub.head(TOP_N)['code'].tolist()

    # Check current actual return
    month_return = test_df[(test_df['code'].isin(top_stocks)) & (test_df['year_month_str'] == month)]['return'].mean()
    if np.isnan(month_return):
        month_return = 0

    # Transaction cost approximation (assuming complete turnover every month for simplicity)
    month_return -= 2 * FEE

    new_nav = nav[-1] * (1 + month_return)
    nav.append(new_nav)
    history.append({'month': month, 'hold': top_stocks, 'return': month_return, 'nav': new_nav})

history_df = pd.DataFrame(history)
history_df.to_csv("4/results/trade_history.csv", index=False)

# Metrics
nav_series = pd.Series(nav[1:]) # ignore initial 1M nav logic
if len(nav_series) == 0:
    print("No backtest results.")
else:
    cum_return = (nav_series.iloc[-1] / INIT_CAPITAL) - 1
    max_drawdown = ((nav_series.cummax() - nav_series) / nav_series.cummax()).max()
    win_rate = len([x for x in history if x['return'] > 0]) / len(history) if len(history)>0 else 0

    # Get Market NAV
    mkt_test = pd.read_csv("4/code/mkt_data.csv")
    mkt_test = mkt_test[(mkt_test['date'] >= '2024-01-01') & (mkt_test['date'] <= '2025-12-31')].copy()
    mkt_test['year_month_str'] = mkt_test['year_month'].astype(str)

    mkt_nav = [INIT_CAPITAL]
    for month in months:
        mkt_ret = mkt_test[mkt_test['year_month_str'] == month]['mkt_return'].mean()
        if np.isnan(mkt_ret):
            mkt_ret = 0
        mkt_nav.append(mkt_nav[-1] * (1 + mkt_ret))

    plt.figure(figsize=(12, 5))
    plt.plot(months, nav[1:], label='策略淨值', marker='o')
    plt.plot(months, mkt_nav[1:len(months)+1], label='上證指數淨值', marker='x', linestyle='--')
    plt.xticks(rotation=45)
    plt.legend()
    plt.title('策略淨值VS上證指數 (Strategy NAV vs SSE Index)')
    plt.tight_layout()
    plt.savefig("4/results/nav_plot.png")

    metrics = {
        '累計收益率 (Cumulative Return)': f"{cum_return:.2%}",
        '最大回撤 (Max Drawdown)': f"{max_drawdown:.2%}",
        '月度勝率 (Monthly Win Rate)': f"{win_rate:.2%}"
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=['指標', '數值'])
    metrics_df.to_csv("4/results/backtest_metrics.csv", index=False)
    print(metrics_df)
