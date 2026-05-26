import baostock as bs
import pandas as pd
import numpy as np

bs.login()

with open('4/code/STOCKS.txt', 'r') as f:
    STOCKS = f.read().split(',')

years = range(2014, 2026)

fin_data = []
for code in STOCKS:
    print(f"Fetching financial data for {code}", flush=True)
    for year in years:
        for q in [1,2,3,4]:
            rs_p = bs.query_profit_data(code=code, year=year, quarter=q)
            if rs_p.error_code == '0' and len(rs_p.data) > 0:
                p_df = rs_p.get_data()
                fin_data.append(p_df)

if len(fin_data) > 0:
    fin_df = pd.concat(fin_data)
    fin_df.to_csv("4/code/profit.csv", index=False)
    print("Saved profit.csv")

growth_data = []
for code in STOCKS:
    print(f"Fetching growth data for {code}", flush=True)
    for year in years:
        for q in [1,2,3,4]:
            rs_g = bs.query_growth_data(code=code, year=year, quarter=q)
            if rs_g.error_code == '0' and len(rs_g.data) > 0:
                g_df = rs_g.get_data()
                growth_data.append(g_df)

if len(growth_data) > 0:
    growth_df = pd.concat(growth_data)
    growth_df.to_csv("4/code/growth.csv", index=False)
    print("Saved growth.csv")

div_data = []
for code in STOCKS:
    print(f"Fetching dividend data for {code}", flush=True)
    for year in years:
        rs_d = bs.query_dividend_data(code=code, year=str(year), yearType="report")
        if rs_d.error_code == '0' and len(rs_d.data) > 0:
            d_df = rs_d.get_data()
            div_data.append(d_df)

if len(div_data) > 0:
    div_df = pd.concat(div_data)
    div_df.to_csv("4/code/dividend.csv", index=False)
    print("Saved dividend.csv")

bs.logout()
