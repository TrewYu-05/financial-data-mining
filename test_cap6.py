import baostock as bs
import pandas as pd
bs.login()
rs = bs.query_profit_data(code="sh.600028", year=2020, quarter=1)
print("profit cols:", rs.get_data().columns)
rs2 = bs.query_growth_data(code="sh.600028", year=2020, quarter=1)
print("growth cols:", rs2.get_data().columns)
rs3 = bs.query_dividend_data(code="sh.600028", year="2020", yearType="report")
print("div cols:", rs3.get_data().columns)
bs.logout()
