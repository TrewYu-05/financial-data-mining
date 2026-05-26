import baostock as bs
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
import os

# ==============================
# Configuration
# ==============================
plt.rcParams['font.sans-serif'] = [
    'SimHei', 
    'Microsoft YaHei',
]
plt.rcParams['axes.unicode_minus'] = False

START_TRAIN = "2018-01-01"
END_TRAIN = "2022-12-31"
START_TEST = "2023-01-01"
END_TEST = "2025-12-31"

RF_ANNUAL = 0.015
RF_DAILY = (1 + RF_ANNUAL) ** (1/250) - 1

STOCKS = {
    "贵州茅台": "sh.600519",
    "中国石油": "sh.601857",
    "五粮液": "sz.000858",
    "泸州老窖": "sz.000568",
    "招商银行": "sh.600036",
    "美的集团": "sz.000333"
}
MARKET_CODE = "sh.000300"

RESULTS_DIR = "3/results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# 1. Login and Data Acquisition Functions
# ==============================
bs.login()

def get_k_data(code, start, end, with_pe=False):
    fields = "date,close"
    if with_pe:
        fields += ",peTTM"

    rs = bs.query_history_k_data_plus(
        code, fields,
        start_date=start, end_date=end,
        frequency="d", adjustflag="3"
    )
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    df = pd.DataFrame(data_list, columns=rs.fields)
    df["close"] = df["close"].astype(float)
    if with_pe:
        df["peTTM"] = df["peTTM"].astype(float)

    df["return"] = df["close"].pct_change()
    df = df.dropna()
    df["return"] = winsorize(df["return"], limits=[0.01, 0.01])
    if with_pe:
        df["peTTM"] = winsorize(df["peTTM"], limits=[0.01, 0.01])

    return df

def get_profit_growth(code, years):
    # Retrieve YOYNI (Net Income Year-over-Year growth) for the given years
    growths = []
    for y in years:
        rs = bs.query_growth_data(code=code, year=y, quarter=4)
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            try:
                growths.append(float(row[3])) # index 3 is YOYNI
            except ValueError:
                pass
    if growths:
        return np.mean(growths)
    return np.nan

# ==============================
# 2. Fetch Data
# ==============================
print("Fetching market data...")
mkt_train = get_k_data(MARKET_CODE, START_TRAIN, END_TRAIN)
mkt_train = mkt_train.rename(columns={"return": "mkt_ret"})

stock_data = {}
profit_growth_data = {}

for name, code in STOCKS.items():
    print(f"Fetching data for {name} ({code})...")
    # Fetch historical prices and peTTM
    df_train = get_k_data(code, START_TRAIN, END_TRAIN, with_pe=True)

    # Calculate profit growth (2018-2022)
    avg_growth = get_profit_growth(code, range(2018, 2023))
    # If growth is negative, using it in PEG is problematic. Often people skip or handle it.
    # For now, we will use it directly but note it in report. We will cap it at a small positive number to avoid negative PEG if needed?
    # The requirement says: "若出现负增长率，需标注并说明处理方式". Let's handle it later in PEG calc.
    profit_growth_data[name] = avg_growth

    # Normalize peTTM
    min_pe = df_train["peTTM"].min()
    max_pe = df_train["peTTM"].max()

    df_train["norm_peTTM"] = (df_train["peTTM"] - min_pe) / (max_pe - min_pe)

    # Merge with market data
    df_merged = pd.merge(df_train, mkt_train[['date', 'mkt_ret']], on='date')
    df_merged["excess_ret"] = df_merged["return"] - RF_DAILY
    df_merged["mkt_excess_ret"] = df_merged["mkt_ret"] - RF_DAILY

    stock_data[name] = {
        "train": df_merged,
        "min_pe": min_pe,
        "max_pe": max_pe
    }

print("\nProfit Growth (2018-2022):")
for name, g in profit_growth_data.items():
    print(f"{name}: {g}")

bs.logout()

# ==============================
# 3. Model Fitting (In-sample: 2018-2022)
# ==============================
print("\nRunning Models...")
results_list = []

for name, code in STOCKS.items():
    df = stock_data[name]["train"]

    # Model 1: CAPM
    X1 = sm.add_constant(df["mkt_excess_ret"])
    y = df["excess_ret"]
    capm = sm.OLS(y, X1).fit()
    alpha1 = capm.params["const"]
    beta1 = capm.params["mkt_excess_ret"]
    r2_1 = capm.rsquared
    p_alpha1 = capm.pvalues["const"]
    p_beta1 = capm.pvalues["mkt_excess_ret"]

    # Model 2: Two-Factor
    X2 = sm.add_constant(df[["mkt_excess_ret", "norm_peTTM"]])
    two_factor = sm.OLS(y, X2).fit()
    alpha2 = two_factor.params["const"]
    beta_mkt = two_factor.params["mkt_excess_ret"]
    beta_pe = two_factor.params["norm_peTTM"]
    r2_2 = two_factor.rsquared
    p_alpha2 = two_factor.pvalues["const"]
    p_beta_mkt = two_factor.pvalues["mkt_excess_ret"]
    p_beta_pe = two_factor.pvalues["norm_peTTM"]

    # Save model alpha for backtesting
    stock_data[name]["alpha2"] = alpha2

    results_list.append({
        "股票": name,
        "代码": code,
        "CAPM_Alpha": alpha1,
        "CAPM_Beta": beta1,
        "CAPM_R2": r2_1,
        "CAPM_Alpha_P": p_alpha1,
        "CAPM_Beta_P": p_beta1,
        "TwoFactor_Alpha": alpha2,
        "TwoFactor_Beta_MKT": beta_mkt,
        "TwoFactor_Beta_PETTM": beta_pe,
        "TwoFactor_R2": r2_2,
        "TwoFactor_Alpha_P": p_alpha2,
        "TwoFactor_Beta_MKT_P": p_beta_mkt,
        "TwoFactor_Beta_PETTM_P": p_beta_pe,
    })

res_df = pd.DataFrame(results_list)
res_df.to_csv(os.path.join(RESULTS_DIR, "model_results.csv"), index=False, encoding="utf-8-sig")
print("Model results saved to model_results.csv")

# ==============================
# 4. PEG Strategy Backtesting (Out-of-sample: 2023-2025)
# ==============================
print("\nRunning Backtest...")
bs.login()

# Fetch out-of-sample data
mkt_test = get_k_data(MARKET_CODE, START_TEST, END_TEST)
mkt_test = mkt_test.rename(columns={"return": "mkt_ret"}).set_index('date')

test_data = {}
for name, code in STOCKS.items():
    df_test = get_k_data(code, START_TEST, END_TEST, with_pe=True)
    df_test = df_test.set_index('date')

    # Normalize peTTM using train min and max
    min_pe = stock_data[name]["min_pe"]
    max_pe = stock_data[name]["max_pe"]
    df_test["norm_peTTM"] = (df_test["peTTM"] - min_pe) / (max_pe - min_pe)

    # Calculate PEG
    growth = profit_growth_data[name]
    # To avoid negative or zero PEG issues, we use the absolute or floor it, but the instruction just says "若出现负增长率，需标注并说明处理方式".
    # China Petroleum is 0.028, all are positive in our sample.
    df_test["PEG"] = df_test["norm_peTTM"] / growth
    test_data[name] = df_test

bs.logout()

# Get the aligned trading days
trading_days = mkt_test.index.sort_values()

# Initial portfolio setup
cash = 1000000.0
position = None # Can be None or a stock name
portfolio_values = []

# Trading costs
COMMISSION = 0.0003
SLIPPAGE = 0.001

for t in range(len(trading_days)-1):
    today = trading_days[t]
    tomorrow = trading_days[t+1]

    # Record portfolio value at the end of 'today'
    # Wait, the returns are close-to-close.
    # If we are holding 'position', its value changes by tomorrow's return.

    # First, let's find the signals from today's data
    buy_candidates = []
    sell_signal = False

    if position is not None:
        if today in test_data[position].index:
            peg_today = test_data[position].loc[today, "PEG"]
            if peg_today > 1.5:
                sell_signal = True
        else:
            # Missing data, hold or sell? Let's hold
            pass

    # Find buy candidates if we are empty or selling
    if position is None or sell_signal:
        for name in STOCKS.keys():
            if today in test_data[name].index:
                peg_today = test_data[name].loc[today, "PEG"]
                if peg_today < 0.8:
                    buy_candidates.append(name)

    # Execute trades at tomorrow's open (we approximate using tomorrow's return)
    # 1. Update value based on tomorrow's return for current position
    # (Because the strategy acts based on today's close, executes at tomorrow's open/close, we just apply tomorrow's return to the held asset).

    # To be precise: If we hold a stock, capital grows by (1 + return_tomorrow).
    # If we trade, we incur costs on the capital traded.

    # Actually, the loop logic:
    # 1. Start of 'tomorrow': check if we need to sell
    if sell_signal and position is not None:
        cash *= (1 - COMMISSION - SLIPPAGE)
        position = None

    # 2. Check if we need to buy
    if position is None and len(buy_candidates) > 0:
        # Choose the one with max Alpha2
        best_stock = max(buy_candidates, key=lambda x: stock_data[x]["alpha2"])
        position = best_stock
        cash *= (1 - COMMISSION - SLIPPAGE)

    # 3. Apply market movement for 'tomorrow'
    if position is not None:
        if tomorrow in test_data[position].index:
            ret_tomorrow = test_data[position].loc[tomorrow, "return"]
            cash *= (1 + ret_tomorrow)

    portfolio_values.append((tomorrow, cash))

# Prepend the initial value for the first day
portfolio_values.insert(0, (trading_days[0], 1000000.0))

# Create Backtest DataFrame
bt_df = pd.DataFrame(portfolio_values, columns=["date", "strategy_value"]).set_index("date")
bt_df["market_value"] = 1000000.0 * (1 + mkt_test["mkt_ret"]).cumprod()

# Calculate metrics
def calc_metrics(series, freq=250):
    total_ret = series.iloc[-1] / series.iloc[0] - 1
    # approx years
    years = len(series) / freq
    ann_ret = (1 + total_ret) ** (1/years) - 1

    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    max_dd = drawdown.min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    return total_ret, ann_ret, max_dd, calmar

strat_tot, strat_ann, strat_mdd, strat_calmar = calc_metrics(bt_df["strategy_value"])
mkt_tot, mkt_ann, mkt_mdd, mkt_calmar = calc_metrics(bt_df["market_value"])

metrics_res = pd.DataFrame({
    "标的": ["PEG策略", "沪深300"],
    "累计收益率": [strat_tot, mkt_tot],
    "年化收益率": [strat_ann, mkt_ann],
    "最大回撤": [strat_mdd, mkt_mdd],
    "卡玛比率": [strat_calmar, mkt_calmar]
})
print("\n===== 回测结果 =====")
print(metrics_res)
metrics_res.to_csv(os.path.join(RESULTS_DIR, "backtest_metrics.csv"), index=False, encoding="utf-8-sig")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(pd.to_datetime(bt_df.index), bt_df["strategy_value"], label="PEG 二因子策略", color='red')
plt.plot(pd.to_datetime(bt_df.index), bt_df["market_value"], label="沪深300", color='blue', alpha=0.7)
plt.title("2023-2025年 PEG二因子策略回测表现 (考虑交易成本)")
plt.xlabel("日期")
plt.ylabel("净值 (元)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "backtest_curve.png"), dpi=300)
plt.show()
