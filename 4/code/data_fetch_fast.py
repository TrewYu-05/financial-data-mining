import baostock as bs
import pandas as pd
import numpy as np

bs.login()

TRAIN_START = "2015-01-01"
TEST_END = "2025-12-31"

rs = bs.query_sz50_stocks()
sz50_list = []
while rs.error_code == "0" and rs.next():
    sz50_list.append(rs.get_row_data())
sz50_df = pd.DataFrame(sz50_list, columns=rs.fields)
STOCKS = sz50_df["code"].tolist()
print(f"Total SZ50 stocks: {len(STOCKS)}")

# Let's save STOCKS for reference
with open('4/code/STOCKS.txt', 'w') as f:
    f.write(','.join(STOCKS))

# We'll use daily data and resample to monthly since peTTM is missing from monthly query in baostock
stock_data = []
for code in STOCKS:
    print(f"Fetching daily k-data for {code}")
    rs_k = bs.query_history_k_data_plus(code, "date,code,close,peTTM,turn,tradestatus,pctChg", start_date=TRAIN_START, end_date=TEST_END, frequency="d")
    df_k = rs_k.get_data()
    if len(df_k) > 0:
        # resample to monthly
        df_k['date'] = pd.to_datetime(df_k['date'])
        df_k['year_month'] = df_k['date'].dt.to_period('M')
        # take the last day of the month
        df_m = df_k.groupby('year_month').last().reset_index()
        stock_data.append(df_m)

if len(stock_data) > 0:
    df_all_m = pd.concat(stock_data)
    df_all_m.to_csv("4/code/monthly_k.csv", index=False)
    print("Saved monthly_k.csv")
else:
    print("No data fetched.")

bs.logout()
