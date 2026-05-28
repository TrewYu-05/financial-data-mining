import baostock as bs
import pandas as pd
bs.login()
rs = bs.query_profit_data(code="sh.600028", year=2020, quarter=4)
print("totalShare 2020Q4:", rs.get_data()[['statDate', 'totalShare']])
rs = bs.query_profit_data(code="sh.600028", year=2020, quarter=3)
print("totalShare 2020Q3:", rs.get_data()[['statDate', 'totalShare']])
bs.logout()
