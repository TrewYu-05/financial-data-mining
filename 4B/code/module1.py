import pandas as pd
import numpy as np
import os

# 1. Load HS300 components
hs300_df = pd.read_parquet('../data/hs300_components.parquet')
stock_codes = hs300_df['con_code'].unique()

print(f"Total HS300 components: {len(stock_codes)}")

all_data = []

# Process each stock
for code in stock_codes:
    try:
        # Load daily data
        daily = pd.read_parquet(f'../data/daily/{code}.parquet')
        daily_basic = pd.read_parquet(f'../data/daily_basic/{code}.parquet')
        moneyflow = pd.read_parquet(f'../data/moneyflow_stock/{code}.parquet')

        # Merge data on trade_date
        df = pd.merge(daily[['trade_date', 'close', 'pre_close', 'amount']],
                      daily_basic[['trade_date', 'free_share', 'circ_mv', 'pe_ttm']],
                      on='trade_date', how='outer')
        df = pd.merge(df, moneyflow[['trade_date', 'buy_amount', 'sell_amount']],
                      on='trade_date', how='outer')

        df['ts_code'] = code
        all_data.append(df)
    except FileNotFoundError:
        print(f"Missing data for {code}")

df_all = pd.concat(all_data, ignore_index=True)

# Convert trade_date to datetime
df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])

# Sort by stock and date
df_all = df_all.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

# Handle missing values: forward fill for each stock
cols_to_fill = df_all.columns.drop(['ts_code', 'trade_date'])
df_all[cols_to_fill] = df_all.groupby('ts_code')[cols_to_fill].ffill()

# Calculate returns
df_all['ret_daily'] = df_all['close'] / df_all['pre_close'] - 1
df_all['excess_ret'] = df_all['ret_daily'] - 0.0001

# Calculate factors
df_all['Reversal'] = -df_all['ret_daily']

# Liquidity = |ret_daily| / amount
df_all['Liquidity'] = np.where(df_all['amount'] > 0, np.abs(df_all['ret_daily']) / df_all['amount'], np.nan)

# MoneyFlow
df_all['net_amount'] = df_all['buy_amount'] - df_all['sell_amount']

def rolling_sum(x):
    return x.rolling(window=5, min_periods=1).sum()

df_all['mf_rolling_5'] = df_all.groupby('ts_code')['net_amount'].transform(rolling_sum)
df_all['MoneyFlow'] = df_all['mf_rolling_5'] / df_all['circ_mv']

# Value
df_all['Value'] = 1 / df_all['pe_ttm']

# 3-sigma winsorization function
def winsorize_3sigma(series):
    mean = series.mean()
    std = series.std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    return series.clip(lower=lower_bound, upper=upper_bound)

# Apply 3-sigma winsorization cross-sectionally per day
factor_cols = ['Reversal', 'Liquidity', 'MoneyFlow', 'Value']
df_all[factor_cols] = df_all.groupby('trade_date')[factor_cols].transform(winsorize_3sigma)

# Forward fill again for factors that might have nan after winsorization (e.g. if all nan in a day)
df_all[factor_cols] = df_all.groupby('ts_code')[factor_cols].ffill()

# Filter dates
df_all = df_all[(df_all['trade_date'] >= '2020-01-01') & (df_all['trade_date'] <= '2025-12-31')]

# Shift next day return as target for prediction
df_all['next_excess_ret'] = df_all.groupby('ts_code')['excess_ret'].shift(-1)

# Format dates back to YYYYMMDD string for CSV
df_all['trade_date'] = df_all['trade_date'].dt.strftime('%Y%m%d')

df_train = df_all[(df_all['trade_date'] >= '20200101') & (df_all['trade_date'] <= '20231231')]
df_test = df_all[(df_all['trade_date'] >= '20240101') & (df_all['trade_date'] <= '20251231')]

os.makedirs('../results', exist_ok=True)
df_train.to_csv('../results/train_data.csv', index=False)
df_test.to_csv('../results/test_data.csv', index=False)

print("Module 1 complete.")
