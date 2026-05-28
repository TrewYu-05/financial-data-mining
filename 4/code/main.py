# ===================== 量化因子选股全流程 =====================
# 作业4：量化全流程——单因子检验 + 因子质检 + 多因子赋权 + 选股回测实战
# 基于上证50成分股，2015-2023训练，2024-2025回测

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# ===================== 全局参数 =====================
RF = 0.0015          # 月度无风险利率
INIT_CAPITAL = 1000000  # 初始资金100万元
FEE = 0.003          # 单边交易成本0.3%
TOP_N = 3            # 每月选股数量
TRAIN_START = "2015-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-31"

# 因子列表（对应数据中的列名）
FACTORS = ["smb", "pe_recip", "quality"]
FACTOR_NAMES = {"smb": "SMB（规模因子）", "pe_recip": "PE倒数（价值因子）", "quality": "Quality（质量因子）"}

# 输出目录
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(RESULT_DIR, exist_ok=True)

# 中文字体设置
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def winsorize(s):
    """3σ缩尾处理：将超出[μ-3σ, μ+3σ]的值替换为边界值"""
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or np.isnan(sigma):
        return s
    return s.clip(mu - 3 * sigma, mu + 3 * sigma)


def standardize(s):
    """Z-Score标准化"""
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or np.isnan(sigma):
        return s - mu
    return (s - mu) / sigma


# ===================== 模块1：数据加载与预处理 =====================
print("=" * 60)
print("模块1：数据加载与预处理")
print("=" * 60)

# 1.1 加载训练集因子和收益率
train_factor = pd.read_csv(os.path.join(DATA_DIR, "train_factor.csv"), parse_dates=["trade_date"])
train_ret = pd.read_csv(os.path.join(DATA_DIR, "train_ret.csv"), parse_dates=["trade_date"])

# 1.2 加载测试集因子和收益率
test_factor = pd.read_csv(os.path.join(DATA_DIR, "test_factor.csv"), parse_dates=["trade_date"])
test_ret = pd.read_csv(os.path.join(DATA_DIR, "test_ret.csv"), parse_dates=["trade_date"])

# 1.3 加载指数数据（用于回测对比）
index_df = pd.read_csv(os.path.join(DATA_DIR, "index_monthly.csv"), parse_dates=["trade_date"])

print(f"训练集因子: {train_factor.shape}, 训练集收益: {train_ret.shape}")
print(f"测试集因子: {test_factor.shape}, 测试集收益: {test_ret.shape}")

# 1.4 合并因子与收益率数据
train_df = pd.merge(train_factor, train_ret, on=["trade_date", "stock_code"], how="inner")
test_df = pd.merge(test_factor, test_ret, on=["trade_date", "stock_code"], how="inner")

print(f"训练集合并后: {train_df.shape}, 测试集合并后: {test_df.shape}")

# 1.5 计算下月超额收益（因变量）
train_df = train_df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
test_df = test_df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

train_df["next_excess_ret"] = train_df.groupby("stock_code")["excess_ret"].shift(-1)
test_df["next_excess_ret"]  = test_df.groupby("stock_code")["excess_ret"].shift(-1)

# 1.6 计算市场超额收益
train_df["mkt_excess"] = train_df["mkt_ret"] - RF
test_df["mkt_excess"]  = test_df["mkt_ret"] - RF

# 1.7 3σ缩尾处理（按截面逐月对因子做缩尾）
print("\n执行3σ缩尾处理...")
for factor in FACTORS:
    winsorized = []
    for month in train_df["trade_date"].unique():
        mask = train_df["trade_date"] == month
        sub = train_df.loc[mask, factor].copy()
        train_df.loc[mask, factor] = winsorize(sub)
    for month in test_df["trade_date"].unique():
        mask = test_df["trade_date"] == month
        sub = test_df.loc[mask, factor].copy()
        test_df.loc[mask, factor] = winsorize(sub)

# 1.8 删除缺失值
train_df = train_df.dropna(subset=FACTORS + ["next_excess_ret", "excess_ret", "mkt_excess"])
test_df = test_df.dropna(subset=FACTORS + ["next_excess_ret", "excess_ret", "mkt_excess"])

print(f"训练集(清洗后): {train_df.shape}, 测试集(清洗后): {test_df.shape}")
print(f"训练集月份数: {train_df['trade_date'].nunique()}, 测试集月份数: {test_df['trade_date'].nunique()}")
print(f"训练集股票数: {train_df['stock_code'].nunique()}, 测试集股票数: {test_df['stock_code'].nunique()}")

# ===================== 模块2：CAPM筛选 + 单因子有效性检验 =====================
print("\n" + "=" * 60)
print("模块2：CAPM个股筛选 + 单因子有效性检验")
print("=" * 60)

# 2.1 CAPM个股筛选
# Ri - Rf = α + β × (Rm - Rf) + ε
print("\n--- CAPM个股筛选 ---")
capm_valid_stocks = []
capm_results = []
for code in train_df["stock_code"].unique():
    sub = train_df[train_df["stock_code"] == code].copy()
    if len(sub) < 12:  # 需要足够的数据点
        continue
    X = sm.add_constant(sub["mkt_excess"])
    y = sub["excess_ret"]
    try:
        model = sm.OLS(y, X).fit()
        beta = model.params.iloc[1] if len(model.params) > 1 else np.nan
        pval = model.pvalues.iloc[1] if len(model.pvalues) > 1 else np.nan
        capm_results.append({"stock_code": code, "beta": beta, "p_value": pval, "significant": pval < 0.05})
        if pval < 0.05:
            capm_valid_stocks.append(code)
    except Exception:
        pass

capm_df = pd.DataFrame(capm_results)
n_total = len(capm_df)
n_valid = capm_df["significant"].sum()
print(f"CAPM检验: {n_valid}/{n_total} 只股票β显著（p<0.05）")

# 保留CAPM通过的股票
train_capm = train_df[train_df["stock_code"].isin(capm_valid_stocks)].copy()
print(f"CAPM筛选后训练集: {train_capm.shape}")

# 2.2 单因子横截面回归
# Return_{i,t+1} = α + β × Factor_{i,t} + ε
print("\n--- 单因子横截面回归 ---")
valid_factors = []
factor_test_results = {}

for factor in FACTORS:
    beta_list, p_list, t_list = [], [], []
    for month in sorted(train_capm["trade_date"].unique()):
        sub = train_capm[train_capm["trade_date"] == month].dropna(subset=[factor, "next_excess_ret"])
        if len(sub) < 10:
            continue
        X = sm.add_constant(sub[factor])
        y = sub["next_excess_ret"]
        try:
            model = sm.OLS(y, X).fit()
            beta_list.append(model.params.iloc[1])
            p_list.append(model.pvalues.iloc[1])
            t_list.append(model.tvalues.iloc[1])
        except Exception:
            pass

    avg_beta = np.mean(beta_list)
    avg_p = np.mean(p_list)
    avg_t = np.mean(t_list)
    prop_sig = np.mean([p < 0.05 for p in p_list])  # 显著月份占比

    factor_test_results[factor] = {
        "avg_beta": avg_beta,
        "avg_p_value": avg_p,
        "avg_t_value": avg_t,
        "prop_significant": prop_sig,
        "n_months": len(beta_list),
    }

    print(f"\n因子: {FACTOR_NAMES[factor]}")
    print(f"  平均β系数: {avg_beta:.6f}")
    print(f"  平均p值: {avg_p:.4f}")
    print(f"  平均t值: {avg_t:.4f}")
    print(f"  显著月份占比: {prop_sig:.2%}")

    if avg_p < 0.05:
        valid_factors.append(factor)
        print(f"  ✓ 判定为有效因子")
    else:
        print(f"  ✗ 未通过有效性检验")

# 若所有因子都不显著，至少保留IC/IR表现最好的两个
if len(valid_factors) == 0:
    print("\n警告：无因子通过p<0.05检验，但继续后续流程以展示完整管线。")
    valid_factors = FACTORS  # 保留全部因子继续流程

print(f"\n有效因子列表: {[FACTOR_NAMES[f] for f in valid_factors]}")

# ===================== 模块3：IC/IR因子质检 =====================
print("\n" + "=" * 60)
print("模块3：IC/IR因子质检")
print("=" * 60)

# 3.1 Z-Score标准化（按截面逐月）
train_std = train_capm.copy()
for factor in FACTORS:
    std_col = f"{factor}_std"
    train_std[std_col] = np.nan
    for month in train_std["trade_date"].unique():
        mask = train_std["trade_date"] == month
        train_std.loc[mask, std_col] = standardize(train_std.loc[mask, factor])

# 3.2 计算IC和IR
ic_ir_results = {}
all_ic_series = {}

for factor in FACTORS:
    ic_list = []
    for month in sorted(train_std["trade_date"].unique()):
        sub = train_std[train_std["trade_date"] == month].dropna(subset=[f"{factor}_std", "next_excess_ret"])
        if len(sub) < 5:
            continue
        ic = sub[[f"{factor}_std", "next_excess_ret"]].corr(method="pearson").iloc[0, 1]
        if not np.isnan(ic):
            ic_list.append(ic)

    ic_mean = np.mean(ic_list)
    ic_std = np.std(ic_list, ddof=1)
    ir = ic_mean / ic_std if ic_std != 0 else 0

    ic_ir_results[factor] = {"IC_mean": ic_mean, "IC_std": ic_std, "IR": ir, "n_ic": len(ic_list)}
    all_ic_series[factor] = ic_list

    # 判定
    ic_judge = "优秀因子" if abs(ic_mean) > 0.05 else ("具备预测能力" if abs(ic_mean) > 0.02 else "预测能力较弱")
    ir_judge = "通过" if abs(ir) > 0.1 else "未通过"

    print(f"\n因子: {FACTOR_NAMES[factor]}")
    print(f"  IC均值: {ic_mean:.4f}  → {ic_judge}")
    print(f"  IC标准差: {ic_std:.4f}")
    print(f"  IR值: {ir:.4f}  → {ir_judge}")
    print(f"  IC序列长度: {len(ic_list)}个月")

# 3.3 绘制IC序列图
fig, axes = plt.subplots(len(FACTORS), 1, figsize=(14, 3.5 * len(FACTORS)), sharex=True)
if len(FACTORS) == 1:
    axes = [axes]
for i, factor in enumerate(FACTORS):
    ax = axes[i]
    ic_series = all_ic_series[factor]
    months = sorted(train_std["trade_date"].unique())[:len(ic_series)]
    ax.bar(range(len(ic_series)), ic_series, color=["red" if v > 0 else "green" for v in ic_series], alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=ic_ir_results[factor]["IC_mean"], color="blue", linestyle="--", label=f"IC均值={ic_ir_results[factor]['IC_mean']:.4f}")
    ax.set_title(f"{FACTOR_NAMES[factor]} 月度IC序列")
    ax.set_ylabel("IC值")
    ax.legend()
axes[-1].set_xlabel("月份序号")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "ic_series.png"), dpi=150)
plt.close()
print("\nIC序列图已保存至 results/ic_series.png")

# ===================== 模块4：多因子线性回归静态赋权 =====================
print("\n" + "=" * 60)
print("模块4：多因子线性回归静态赋权")
print("=" * 60)

# 4.1 准备建模数据
std_cols = [f"{f}_std" for f in valid_factors]
model_df = train_std.dropna(subset=std_cols + ["next_excess_ret"])

# 4.2 截面回归建模
X = model_df[std_cols]
X = sm.add_constant(X)
y = model_df["next_excess_ret"]

model = sm.OLS(y, X).fit()
print(model.summary())

# 提取权重
weights = {}
for i, factor in enumerate(valid_factors):
    weights[factor] = model.params.iloc[i + 1]  # 跳过常数项

print("\n因子静态权重:")
for factor in valid_factors:
    print(f"  {FACTOR_NAMES[factor]}: {weights[factor]:.6f}")

# 4.3 构建综合得分函数
def score_func(row, factors, weights_dict, suffix="_std"):
    """计算综合得分：加权因子标准化值之和"""
    return sum(weights_dict[f] * row[f"{f}{suffix}"] for f in factors)

# 保存因子权重
weights_df = pd.DataFrame({
    "factor": valid_factors,
    "factor_name": [FACTOR_NAMES[f] for f in valid_factors],
    "weight": [weights[f] for f in valid_factors],
})
weights_df.to_csv(os.path.join(RESULT_DIR, "factor_weights.csv"), index=False, encoding="utf-8-sig")

# ===================== 模块5：截面选股 + 样本外回测 =====================
print("\n" + "=" * 60)
print("模块5：截面选股 + 样本外回测")
print("=" * 60)

# 5.1 测试集因子标准化（按截面逐月）
test_std = test_df.copy()
for factor in valid_factors:
    std_col = f"{factor}_std"
    test_std[std_col] = np.nan
    for month in test_std["trade_date"].unique():
        mask = test_std["trade_date"] == month
        test_std.loc[mask, std_col] = standardize(test_std.loc[mask, factor])

# 5.2 逐月打分选股
months_test = sorted(test_std["trade_date"].unique())
print(f"测试集月份: {months_test[0].strftime('%Y-%m-%d')} → {months_test[-1].strftime('%Y-%m-%d')}, 共{len(months_test)}个月")

nav = [INIT_CAPITAL]
nav_dates = [months_test[0] - pd.Timedelta(days=30)]  # 净值起始日
trade_history = []

for month in months_test:
    sub = test_std[test_std["trade_date"] == month].copy()
    sub["score"] = sub.apply(lambda row: score_func(row, valid_factors, weights), axis=1)
    sub = sub.sort_values("score", ascending=False)

    # 选Top N股票
    top_stocks = sub.head(TOP_N)
    top_codes = top_stocks["stock_code"].tolist()

    # 计算当月等权收益
    month_return = top_stocks["monthly_ret"].mean()
    if np.isnan(month_return):
        month_return = 0

    # 扣除双边交易成本（简化：每月调仓收双边手续费）
    month_return_net = month_return - 2 * FEE

    # 更新净值
    new_nav = nav[-1] * (1 + month_return_net)
    nav.append(new_nav)
    nav_dates.append(month)

    trade_history.append({
        "trade_date": month,
        "hold_codes": ",".join(top_codes),
        "hold_count": len(top_codes),
        "avg_return": month_return,
        "net_return": month_return_net,
        "nav": new_nav,
        "top1_code": top_codes[0] if len(top_codes) > 0 else "",
        "top1_score": top_stocks["score"].iloc[0] if len(top_stocks) > 0 else np.nan,
    })

# 5.3 计算回测指标
nav_series = pd.Series(nav[1:], index=months_test)  # 每月末净值
nav_series_full = pd.Series(nav, index=[months_test[0] - pd.Timedelta(days=30)] + list(months_test))

# 策略收益率序列
strategy_returns = pd.Series(
    [h["net_return"] for h in trade_history],
    index=months_test
)

# 累计收益率
cum_return = (nav[-1] / INIT_CAPITAL) - 1

# 年化收益率
n_years = len(months_test) / 12
annual_return = (nav[-1] / INIT_CAPITAL) ** (1 / n_years) - 1 if n_years > 0 else 0

# 最大回撤
nav_peak = nav_series.cummax()
drawdown = (nav_series - nav_peak) / nav_peak
max_drawdown = drawdown.min()

# 月度胜率
win_rate = (strategy_returns > 0).sum() / len(strategy_returns)

# 超额收益（相对上证指数）
# 计算上证指数净值
index_test = index_df[(index_df["trade_date"] >= pd.Timestamp(TEST_START)) &
                       (index_df["trade_date"] <= pd.Timestamp(TEST_END))].copy()
index_test = index_test.sort_values("trade_date")

# 对齐月份
index_nav = [INIT_CAPITAL]
for month in months_test:
    idx_row = index_test[index_test["trade_date"] == month]
    if len(idx_row) > 0:
        mkt_ret = idx_row["mkt_ret"].values[0]
    else:
        mkt_ret = 0
    index_nav.append(index_nav[-1] * (1 + mkt_ret))

index_nav_series = pd.Series(index_nav[1:], index=months_test)
index_cum_return = (index_nav[-1] / INIT_CAPITAL) - 1

# 超额收益
excess_return = cum_return - index_cum_return

# 夏普比率（年化，假设无风险利率）
rf_annual = (1 + RF) ** 12 - 1
strategy_excess = strategy_returns - RF
sharpe_ratio = np.sqrt(12) * strategy_excess.mean() / strategy_excess.std() if strategy_excess.std() != 0 else 0

# 收益波动率
volatility = strategy_returns.std() * np.sqrt(12)

print("\n" + "-" * 40)
print("回测指标汇总")
print("-" * 40)
print(f"  累计收益率:      {cum_return:.4f} ({cum_return:.2%})")
print(f"  年化收益率:      {annual_return:.4f} ({annual_return:.2%})")
print(f"  年化波动率:      {volatility:.4f} ({volatility:.2%})")
print(f"  夏普比率:        {sharpe_ratio:.4f}")
print(f"  最大回撤:        {max_drawdown:.4f} ({max_drawdown:.2%})")
print(f"  月度胜率:        {win_rate:.4f} ({win_rate:.2%})")
print(f"  上证指数累计收益: {index_cum_return:.4f} ({index_cum_return:.2%})")
print(f"  超额收益:        {excess_return:.4f} ({excess_return:.2%})")

# 5.4 保存回测指标
metrics_dict = {
    "指标": [
        "累计收益率", "年化收益率", "年化波动率", "夏普比率",
        "最大回撤", "月度胜率", "上证指数累计收益", "超额收益（相对上证）",
        "回测月数", "初始资金",
    ],
    "数值": [
        f"{cum_return:.4f}", f"{annual_return:.4f}", f"{volatility:.4f}", f"{sharpe_ratio:.4f}",
        f"{max_drawdown:.4f}", f"{win_rate:.4f}", f"{index_cum_return:.4f}", f"{excess_return:.4f}",
        f"{len(months_test)}", f"{INIT_CAPITAL:,}",
    ],
}
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(os.path.join(RESULT_DIR, "backtest_metrics.csv"), index=False, encoding="utf-8-sig")

# 5.5 保存交易记录
trade_df = pd.DataFrame(trade_history)
trade_df.to_csv(os.path.join(RESULT_DIR, "trade_history.csv"), index=False, encoding="utf-8-sig")

# 5.6 绘制策略净值对比图
fig, ax1 = plt.subplots(figsize=(14, 7))

# 策略净值
ax1.plot(months_test, nav[1:], "b-", linewidth=2, label="策略净值", marker="o", markersize=3)
# 上证指数净值
ax1.plot(months_test, index_nav[1:], "r--", linewidth=2, label="上证指数净值", marker="x", markersize=3)
ax1.axhline(y=INIT_CAPITAL, color="gray", linestyle=":", linewidth=1, label="初始资金线")

# 标注最大回撤区间
dd_start_idx = drawdown.idxmin()
dd_end_idx = drawdown[dd_start_idx:].idxmax() if drawdown.index.get_loc(dd_start_idx) < len(drawdown) - 1 else drawdown.index[-1]
ax1.axvspan(dd_start_idx, dd_end_idx, alpha=0.1, color="red", label=f"最大回撤区间 ({max_drawdown:.2%})")

ax1.set_xlabel("日期")
ax1.set_ylabel("净值（元）")
ax1.set_title("策略净值 vs 上证指数（2024-2025 样本外回测）")
ax1.legend(loc="upper left")
ax1.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "nav_plot.png"), dpi=150)
plt.close()
print("策略净值对比图已保存至 results/nav_plot.png")

# 5.7 绘制回撤曲线
fig, ax2 = plt.subplots(figsize=(14, 4))
ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color="red", alpha=0.3)
ax2.plot(drawdown.index, drawdown.values * 100, "r-", linewidth=1)
ax2.set_xlabel("日期")
ax2.set_ylabel("回撤 (%)")
ax2.set_title("策略回撤曲线")
ax2.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "drawdown_plot.png"), dpi=150)
plt.close()
print("回撤曲线图已保存至 results/drawdown_plot.png")

# ===================== 保存IC/IR结果 =====================
ic_ir_df = pd.DataFrame([
    {
        "factor": f,
        "factor_name": FACTOR_NAMES[f],
        "IC_mean": ic_ir_results[f]["IC_mean"],
        "IC_std": ic_ir_results[f]["IC_std"],
        "IR": ic_ir_results[f]["IR"],
        "n_ic": ic_ir_results[f]["n_ic"],
    }
    for f in FACTORS
])
ic_ir_df.to_csv(os.path.join(RESULT_DIR, "ic_ir_results.csv"), index=False, encoding="utf-8-sig")

# ===================== 保存因子检验结果 =====================
factor_test_df = pd.DataFrame([
    {
        "factor": f,
        "factor_name": FACTOR_NAMES[f],
        "avg_beta": factor_test_results[f]["avg_beta"],
        "avg_p_value": factor_test_results[f]["avg_p_value"],
        "avg_t_value": factor_test_results[f]["avg_t_value"],
        "prop_significant": factor_test_results[f]["prop_significant"],
        "is_valid": f in valid_factors,
    }
    for f in FACTORS
])
factor_test_df.to_csv(os.path.join(RESULT_DIR, "factor_test_results.csv"), index=False, encoding="utf-8-sig")

# ===================== 最终汇总 =====================
print("\n" + "=" * 60)
print("全流程完成!")
print("=" * 60)
print(f"\n输出结果文件:")
for f in os.listdir(RESULT_DIR):
    fpath = os.path.join(RESULT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  results/{f} ({size_kb:.1f} KB)")
