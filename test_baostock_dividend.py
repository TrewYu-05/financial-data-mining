import baostock as bs
import pandas as pd
bs.login()
rs_dividend = bs.query_dividend_data(code="sh.600000", year="2020", yearType="report")
print("dividend details:\n", rs_dividend.get_data()[['dividCashPsBeforeTax', 'dividCashPsAfterTax']])
bs.logout()
