import baostock as bs
import pandas as pd
bs.login()
rs = bs.query_profit_data(code="sh.600028", year=2020, quarter=4)
print("profit 2020 Q4:", rs.get_data()[['pubDate', 'statDate', 'roeAvg']])
rs = bs.query_profit_data(code="sh.600028", year=2020, quarter=3)
print("profit 2020 Q3:", rs.get_data()[['pubDate', 'statDate', 'roeAvg']])
bs.logout()
