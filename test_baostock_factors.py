import baostock as bs
import pandas as pd
bs.login()
rs_profit = bs.query_profit_data(code="sh.600000", year=2020, quarter=1)
print("profit:", rs_profit.get_data()[['pubDate', 'statDate', 'roeAvg', 'totalShare']])
rs_growth = bs.query_growth_data(code="sh.600000", year=2020, quarter=1)
print("growth:", rs_growth.get_data()[['pubDate', 'statDate', 'YOYNI']])
rs_dividend = bs.query_dividend_data(code="sh.600000", year="2020", yearType="report")
print("dividend:", rs_dividend.get_data().columns)
bs.logout()
