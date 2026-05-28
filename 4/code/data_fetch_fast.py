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

# monthly_k
stock_data = []
for code in STOCKS:
    # monthly frequency does not return peTTM, only daily. We must use daily.
    # To be safe, just do daily peTTM and close.
    rs_d = bs.query_history_k_data_plus(code, "date,code,close,peTTM", start_date=TRAIN_START, end_date=TEST_END, frequency="d")
    df = rs_d.get_data()
    if len(df) == 0:
        continue
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    df['close'] = df['close'].replace("", np.nan).astype(float)
    df['peTTM'] = df['peTTM'].replace("", np.nan).astype(float)
    df_m = df.groupby('year_month').last().reset_index()

    # calc return based on monthly closing
    df_m['return'] = df_m['close'].pct_change()
    stock_data.append(df_m)

monthly_k = pd.concat(stock_data)
monthly_k.to_csv("4/code/monthly_k.csv", index=False)
print("Finished monthly_k.csv")
bs.logout()
