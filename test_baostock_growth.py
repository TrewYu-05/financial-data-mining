import baostock as bs
import pandas as pd
bs.login()
rs_growth = bs.query_growth_data(code="sh.600000", year=2020, quarter=1)
print("growth:", rs_growth.get_data().columns)
rs_profit = bs.query_profit_data(code="sh.600000", year=2020, quarter=1)
print("profit:", rs_profit.get_data().columns)
rs_daily = bs.query_history_k_data_plus("sh.600000", "date,code,close,peTTM,pbMRQ,psTTM,pcfNcfTTM,turn,tradestatus,pctChg,isST", start_date='2020-01-01', end_date='2020-01-31', frequency="d")
print("daily cols:", rs_daily.get_data().columns)
bs.logout()
