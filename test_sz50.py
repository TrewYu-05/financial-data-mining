import baostock as bs
import pandas as pd
bs.login()
rs = bs.query_sz50_stocks()
sz50_list = []
while rs.error_code == "0" and rs.next():
    sz50_list.append(rs.get_row_data())
sz50_df = pd.DataFrame(sz50_list, columns=rs.fields)
print(sz50_df.head())
bs.logout()
