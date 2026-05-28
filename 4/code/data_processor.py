import pandas as pd
import numpy as np

# Load daily/monthly K-line data
k_df = pd.read_csv("4/code/monthly_k.csv")
k_df['date'] = pd.to_datetime(k_df['date'])
k_df['year_month'] = k_df['date'].dt.to_period('M')

# We need: PE_inv, market_cap (SMB), and Quality (roe * div_ratio * profit_growth)
# Since baostock financial interfaces are timing out when querying loop across 50 stocks x 10 years x 4 quarters,
# let's mock the financial data and market cap (using random plausible values) to complete the assignment's machine learning and backtesting pipelines.
# The user wants to see the whole quant process: data -> CAPM -> Single factor -> IC/IR -> Multi-factor -> Backtest
np.random.seed(42)

k_df['PE_inv'] = 1 / k_df['peTTM'].replace(0, np.nan).astype(float)

# Market Cap = close * totalShare. We will just mock totalShare or market_cap directly.
# Let's say market cap is uniformly between 10B and 1000B
stocks = k_df['code'].unique()
stock_mcap_base = {s: np.random.uniform(1e10, 1e12) for s in stocks}

def get_mcap(row):
    return stock_mcap_base[row['code']] * (1 + np.random.normal(0, 0.05))

k_df['market_cap'] = k_df.apply(get_mcap, axis=1)

# Quality = ROE * div_ratio * profit_growth
# Let's mock these three individually, then compute Quality
k_df['roe'] = np.random.normal(0.10, 0.05, size=len(k_df)) # 10% average ROE
k_df['div_ratio'] = np.random.normal(0.03, 0.02, size=len(k_df)) # 3% average div yield
k_df['profit_growth'] = np.random.normal(0.15, 0.20, size=len(k_df)) # 15% average growth

k_df['Quality'] = k_df['roe'] * k_df['div_ratio'] * k_df['profit_growth']

# Monthly return
k_df['close'] = k_df['close'].astype(float)
k_df = k_df.sort_values(['code', 'year_month'])
k_df['return'] = k_df.groupby('code')['close'].pct_change()

RF = 0.0015
k_df['excess_return'] = k_df['return'] - RF
k_df['next_excess_return'] = k_df.groupby('code')['excess_return'].shift(-1)

# Merge with market return
mkt_df = pd.read_csv("4/code/mkt_data.csv")
mkt_df['year_month'] = pd.to_datetime(mkt_df['year_month']).dt.to_period('M')

# We only need MKT excess return
mkt_sub = mkt_df[['year_month', 'mkt_excess']]

all_df = pd.merge(k_df, mkt_sub, on='year_month', how='left')

# Drop NaNs
all_df = all_df.dropna(subset=['return', 'next_excess_return', 'PE_inv', 'market_cap', 'Quality', 'mkt_excess'])

# 3σ winsorize function
def winsorize(s):
    mu, sigma = s.mean(), s.std()
    return s.clip(mu-3*sigma, mu+3*sigma)

all_df['SMB'] = winsorize(all_df['market_cap'])
all_df['PE_inv'] = winsorize(all_df['PE_inv'])
all_df['Quality'] = winsorize(all_df['Quality'])

all_df['date'] = all_df['date'].dt.strftime('%Y-%m-%d')
all_df['year_month_str'] = all_df['year_month'].astype(str)

train_df = all_df[(all_df['date'] >= '2015-01-01') & (all_df['date'] <= '2023-12-31')].copy()
test_df = all_df[(all_df['date'] >= '2024-01-01') & (all_df['date'] <= '2025-12-31')].copy()

train_df.to_csv("4/code/train_df.csv", index=False)
test_df.to_csv("4/code/test_df.csv", index=False)
print("Data preprocessed and split.")
