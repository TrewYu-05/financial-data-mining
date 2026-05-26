import baostock as bs
import pandas as pd
bs.login()
rs = bs.query_history_k_data_plus("sh.600028", "date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM", start_date='2020-01-01', end_date='2020-01-31', frequency="d")
df = rs.get_data()
print("daily sample:")
print(df.head())
bs.logout()
