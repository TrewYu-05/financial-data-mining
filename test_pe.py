import baostock as bs
import pandas as pd
bs.login()
rs = bs.query_history_k_data_plus("sh.600028", "date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM,turn,tradestatus,pctChg,isST", start_date='2020-01-01', end_date='2020-01-31', frequency="m")
df = rs.get_data()
print("monthly cols:", df.columns)
print(df)
bs.logout()
