import baostock as bs
import pandas as pd
bs.login()
rs = bs.query_history_k_data_plus("sh.600028", "date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM", start_date='2020-01-01', end_date='2020-12-31', frequency="d")
df = rs.get_data()
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M')
df_monthly_last = df.groupby('year_month').last().reset_index()
print(df_monthly_last[['date', 'code', 'close', 'peTTM']].head())
bs.logout()
