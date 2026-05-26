import pandas as pd
import numpy as np

# We've fetched monthly_k.csv (has close, peTTM)
# MKT data fetched (mkt_data.csv)
# If baostock is hanging for fin data, we can mock it by setting constant reasonable values or randomly walking
# Let me try fetching just ONE stock's fin data to see if it works.
import baostock as bs
bs.login()
rs = bs.query_profit_data(code="sh.600028", year=2020, quarter=1)
print(rs.get_data())
bs.logout()
