import baostock as bs
import pandas as pd
import numpy as np

bs.login()

TRAIN_START = "2015-01-01"
TEST_END = "2025-12-31"

# 1. Fetch SZ50
rs = bs.query_sz50_stocks()
sz50_list = []
while rs.error_code == "0" and rs.next():
    sz50_list.append(rs.get_row_data())
sz50_df = pd.DataFrame(sz50_list, columns=rs.fields)
STOCKS = sz50_df["code"].tolist()
print(f"Total SZ50 stocks: {len(STOCKS)}")

# 2. Fetch Market index (monthly)
rs = bs.query_history_k_data_plus("sh.000001", "date,close", start_date=TRAIN_START, end_date=TEST_END, frequency="m")
mkt_df = rs.get_data()
mkt_df['close'] = mkt_df['close'].astype(float)
mkt_df['mkt_return'] = mkt_df['close'].pct_change()
mkt_df['mkt_excess'] = mkt_df['mkt_return'] - 0.0015
mkt_df['year_month'] = pd.to_datetime(mkt_df['date']).dt.to_period('M')

# We'll save it
mkt_df.to_csv("4/code/mkt_data.csv", index=False)
print("Finished saving market data")

# To prevent hanging, let's test fetching just one stock's fin data
years = range(2014, 2026)
quarters = [1, 2, 3, 4]

fin_data = []
for code in STOCKS:
    print(f"Fetching financial data for {code}", flush=True)
    for year in years:
        for q in quarters:
            # Profit (roeAvg, totalShare)
            rs_p = bs.query_profit_data(code=code, year=year, quarter=q)
            if rs_p.error_code == '0' and len(rs_p.data) > 0:
                p_df = rs_p.get_data()
                roe = p_df['roeAvg'].values[0]
                totalShare = p_df['totalShare'].values[0]
                pubDate_p = p_df['pubDate'].values[0]
            else:
                roe, totalShare, pubDate_p = np.nan, np.nan, np.nan

            # Growth (YOYNI)
            rs_g = bs.query_growth_data(code=code, year=year, quarter=q)
            if rs_g.error_code == '0' and len(rs_g.data) > 0:
                g_df = rs_g.get_data()
                yoyni = g_df['YOYNI'].values[0]
                pubDate_g = g_df['pubDate'].values[0]
            else:
                yoyni, pubDate_g = np.nan, np.nan

            # Use max pubDate as the effective date
            pubDate = pubDate_p if pd.notna(pubDate_p) else pubDate_g
            if pd.isna(pubDate) or pubDate == '':
                continue

            fin_data.append({
                'code': code,
                'pubDate': pubDate,
                'statDate': f"{year}-{(q*3):02d}-01", # approx
                'roe': float(roe) if roe and roe!="" else np.nan,
                'total_share': float(totalShare) if totalShare and totalShare!="" else np.nan,
                'profit_growth': float(yoyni) if yoyni and yoyni!="" else np.nan
            })

fin_df = pd.DataFrame(fin_data)
fin_df = fin_df.dropna(subset=['pubDate'])
fin_df = fin_df[fin_df['pubDate'] != '']
fin_df['pubDate'] = pd.to_datetime(fin_df['pubDate'])
fin_df = fin_df.sort_values(['code', 'pubDate'])
fin_df.to_csv("4/code/fin_data.csv", index=False)
print("Finished fetching financial data.")

# 4. Fetch daily K data for closing price, peTTM to calculate market cap
stock_data = []
for code in STOCKS:
    print(f"Fetching daily k-data for {code}")
    rs_k = bs.query_history_k_data_plus(code, "date,code,close,peTTM", start_date=TRAIN_START, end_date=TEST_END, frequency="d")
    df = rs_k.get_data()
    if len(df) == 0:
        continue
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = df['close'].replace("", np.nan).astype(float)
    df['peTTM'] = df['peTTM'].replace("", np.nan).astype(float)
    df['year_month'] = df['date'].dt.to_period('M')
    # get end of month
    df_m = df.groupby('year_month').last().reset_index()

    # Also fetch monthly exact return
    rs_m = bs.query_history_k_data_plus(code, "date,code,close", start_date=TRAIN_START, end_date=TEST_END, frequency="m")
    df_monthly_raw = rs_m.get_data()
    df_monthly_raw['date'] = pd.to_datetime(df_monthly_raw['date'])
    df_monthly_raw['year_month'] = df_monthly_raw['date'].dt.to_period('M')
    df_monthly_raw['close_m'] = df_monthly_raw['close'].replace("", np.nan).astype(float)
    df_monthly_raw['return'] = df_monthly_raw['close_m'].pct_change()

    df_merged = pd.merge(df_m, df_monthly_raw[['year_month', 'return']], on='year_month', how='left')
    stock_data.append(df_merged)

k_data_df = pd.concat(stock_data)
k_data_df.to_csv("4/code/k_data.csv", index=False)
print("Finished daily k-data fetch.")

# 5. Fetch dividend data for dividend ratio
div_data = []
for code in STOCKS:
    for year in years:
        rs_d = bs.query_dividend_data(code=code, year=str(year), yearType="report")
        if rs_d.error_code == '0' and len(rs_d.data) > 0:
            d_df = rs_d.get_data()
            for idx, row in d_df.iterrows():
                # dividCashPsBeforeTax is the cash dividend per share
                div_ps = row['dividCashPsBeforeTax']
                pub_date = row['dividPlanAnnounceDate'] if row['dividPlanAnnounceDate'] else row['dividAgmPumDate']
                if pub_date:
                    div_data.append({
                        'code': code,
                        'pubDate': pub_date,
                        'div_ps': float(div_ps) if div_ps and div_ps!="" else np.nan
                    })
div_df = pd.DataFrame(div_data)
div_df['pubDate'] = pd.to_datetime(div_df['pubDate'])
div_df = div_df.sort_values(['code', 'pubDate'])
div_df.to_csv("4/code/div_data.csv", index=False)
print("Finished fetching dividend data.")

bs.logout()
