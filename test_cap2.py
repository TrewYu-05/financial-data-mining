import baostock as bs
import pandas as pd
bs.login()
rs = bs.query_profit_data(code="sh.600028", year=2020, quarter=1)
profit_df = rs.get_data()
print("profit:", profit_df.columns)
rs2 = bs.query_history_k_data_plus("sh.600028", "date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM", start_date='2020-01-01', end_date='2020-01-05', frequency="d")
print("daily:", rs2.get_data().columns)
bs.logout()
