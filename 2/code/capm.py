# ==============================
# CAPM 模型 Python 实现
# ==============================

import baostock as bs
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
import os

# 配置 matplotlib 以支持中文字符显示
plt.rcParams['font.sans-serif'] = [
    'SimHei', 
    'Microsoft YaHei',
]
plt.rcParams['axes.unicode_minus'] = False

# ============= 1. 登录 Baostock =============
lg = bs.login()
if lg.error_code != '0':
    print("login respond error_code:" + lg.error_code)
    print("login respond  error_msg:" + lg.error_msg)

# ============= 2. 数据获取函数 =============
def get_data(code, start, end):
    rs = bs.query_history_k_data_plus(
        code, "date,close",
        start_date=start, end_date=end,
        frequency="d", adjustflag="3"  # 前复权
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    df = pd.DataFrame(data_list, columns=rs.fields)
    df["close"] = df["close"].astype(float)
    df["return"] = df["close"].pct_change()  # 计算收益率
    df = df.dropna()

    # ======== 异常值处理：官方缩尾（代替3σ，一行搞定）========
    df["return"] = winsorize(df["return"], limits=[0.01, 0.01])
    return df

# ============= 3. CAPM 拟合函数 =============
def fit_capm(stock_ret, market_ret, Rf_daily):
    df = pd.DataFrame({"Ri": stock_ret, "Rm": market_ret}).dropna()
    df["Ri_ex"] = df["Ri"] - Rf_daily
    df["Rm_ex"] = df["Rm"] - Rf_daily

    X = sm.add_constant(df["Rm_ex"])
    model = sm.OLS(df["Ri_ex"], X).fit()
    alpha, beta = model.params
    return alpha, beta, model

# ============= 4. 参数设置（固定）=============
start_train = "2020-01-01"
end_train   = "2022-12-31"
start_test  = "2020-01-01"
end_test    = "2025-12-31"

Rf_year = 0.03
Rf_daily = (1 + Rf_year) ** (1/250) - 1

# 股票列表
stocks = {
    "贵州茅台": "sh.600519",
    "中国石油": "sh.601857",
    "五粮液": "sz.000858",
    "泸州老窖": "sz.000568",
    "招商银行": "sh.600036",
    "美的集团": "sz.000333"
}

# ============= 5. 获取市场数据（沪深300）=============
# baostock 中沪深300指数代码是 sh.000300
market = get_data("sh.000300", start_train, end_train)

# ============= 6. 循环计算所有股票的 α, β =============
result = []
for name, code in stocks.items():
    stock_df = get_data(code, start_train, end_train)

    # 按照日期对齐数据
    aligned_df = pd.merge(stock_df[['date', 'return']], market[['date', 'return']], on='date', suffixes=('_stock', '_market'))

    alpha, beta, model = fit_capm(aligned_df["return_stock"], aligned_df["return_market"], Rf_daily)
    p_value_alpha = model.pvalues.iloc[0]
    p_value_beta = model.pvalues.iloc[1]

    result.append([name, code, alpha, beta, p_value_alpha, p_value_beta])

# 输出结果并保存
res_df = pd.DataFrame(result, columns=["股票", "代码", "α(Alpha)", "β(Beta)", "Alpha_PValue", "Beta_PValue"])
res_df.to_csv("../results/capm_results10.csv", index=False, encoding='utf-8-sig')

print("===== CAPM 拟合结果 =====")
print(res_df.round(4))
best_idx = res_df["α(Alpha)"].idxmax()
best_stock = res_df.loc[best_idx, "股票"]
best_code = res_df.loc[best_idx, "代码"]
print("\nAlpha最高的股票是：", best_stock)

# ============= 7. 回测：持有 Alpha 最高股票 =============
# 获取回测数据
test_df = get_data(best_code, start_test, end_test)
market_test = get_data("sh.000300", start_test, end_test)

# 确保日期对齐计算指标
aligned_test = pd.merge(test_df[['date', 'return']], market_test[['date', 'return']], on='date', suffixes=('_stock', '_market'))

# 计算累计收益 (这里使用基准 100)
aligned_test["cum_ret_stock"] = (1 + aligned_test["return_stock"]).cumprod() * 100
aligned_test["cum_ret_market"] = (1 + aligned_test["return_market"]).cumprod() * 100

# 将计算结果赋回用于画图的日期序列
test_df = pd.merge(test_df, aligned_test[['date', 'cum_ret_stock']], on='date')
market_test = pd.merge(market_test, aligned_test[['date', 'cum_ret_market']], on='date')


# 计算指标 (使用小数收益率计算回测指标)
def calc_metrics(ret_series, cum_ret_series):
    total_return = (cum_ret_series.iloc[-1] / cum_ret_series.iloc[0]) - 1

    # 计算最大回撤
    running_max = cum_ret_series.cummax()
    drawdown = (cum_ret_series - running_max) / running_max
    max_drawdown = drawdown.min()

    # 卡玛比率 (年化收益率 / 最大回撤的绝对值)
    # 计算年化收益率, 假设2年
    annual_return = (1 + total_return) ** (1/2) - 1
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return total_return, max_drawdown, calmar_ratio

stock_tot_ret, stock_mdd, stock_calmar = calc_metrics(aligned_test["return_stock"], aligned_test["cum_ret_stock"])
market_tot_ret, market_mdd, market_calmar = calc_metrics(aligned_test["return_market"], aligned_test["cum_ret_market"])

metrics_df = pd.DataFrame({
    "标的": [best_stock, "沪深300"],
    "累积收益率": [stock_tot_ret, market_tot_ret],
    "最大回撤": [stock_mdd, market_mdd],
    "卡玛比率": [stock_calmar, market_calmar]
})

metrics_df.to_csv("../results/backtest_metrics10.csv", index=False, encoding='utf-8-sig')
print("\n===== 回测指标 =====")
print(metrics_df.round(4))


# 画图
plt.figure(figsize=(12, 5))
plt.plot(pd.to_datetime(test_df["date"]), test_df["cum_ret_stock"], label=f"{best_stock} 持仓收益")
plt.plot(pd.to_datetime(market_test["date"]), market_test["cum_ret_market"], label="沪深300")
plt.title(f"CAPM 理论Alpha最高股票 ({best_stock}) 10年回测")
plt.legend()
plt.xticks(rotation=45)
plt.ylabel("累计净值 (基准=100)")
plt.xlabel("日期")
plt.tight_layout()

# 确保文件夹存在 (脚本相对于自身运行的目录)
os.makedirs("../results", exist_ok=True)
plt.savefig("../results/backtest_plot10.png", dpi=300)
plt.savefig("../results/backtest_plot10.svg")
plt.show()

bs.logout()
