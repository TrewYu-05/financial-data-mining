import baostock as bs
import pandas as pd
bs.login()
rs_profit = bs.query_profit_data(code="sh.600000", year=2020, quarter=1)
print("profit:", rs_profit.get_data())
rs_dividend = bs.query_dividend_data(code="sh.600000", year="2020", yearType="report")
print("dividend:", rs_dividend.get_data())
bs.logout()
