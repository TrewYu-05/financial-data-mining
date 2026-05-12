# 导入库
import baostock as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import matplotlib.dates as mdates

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 确保图片保存目录存在
os.makedirs('1/results', exist_ok=True)

# ======== 1. 固定参数 ========
# 随机选择白酒行业1家上市公司，这里使用贵州茅台
stock_code = "sh.600519"
start_date = "2024-01-01"
end_date = "2025-12-31"
window = 10   # 10天滚动窗口

# ======== 任务1：数据获取 ========
print("正在登录Baostock...")
bs.login()

print(f"正在获取 {stock_code} 的数据...")
rs = bs.query_history_k_data_plus(
    stock_code,
    "date,close,volume",
    start_date=start_date, end_date=end_date,
    frequency="d", adjustflag="3" # 3：前复权
)

data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())

# 数据转换为DataFrame
df = pd.DataFrame(data_list, columns=rs.fields)

# 退出系统
bs.logout()

print("数据获取完成，正在进行数据清洗...")
# 数据清洗：转换数值类型、设置日期索引
df['close'] = pd.to_numeric(df['close'])
df['volume'] = pd.to_numeric(df['volume'])
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# ======== 任务2：时序特征构建 ========
print("正在构建时序特征...")
# 1. 计算【价格收益率】(价格变动)
df['price_return'] = df['close'].pct_change()
# 2. 计算【成交量收益率】(成交量变动)
df['vol_return'] = df['volume'].pct_change()
# 3. 构建标签：次日价格收益率（预测目标）
df['label'] = df['price_return'].shift(-1)

# 4. 构造10天滚动特征
for i in range(1, window+1):
    df[f'price_return_{i}'] = df['price_return'].shift(i)
    df[f'vol_return_{i}'] = df['vol_return'].shift(i)

# 5. 删除缺失值，划分特征X和标签y
df = df.dropna()

# 特征列：20个特征（10个价变动 + 10个量变动）
feature_cols = [f'price_return_{i}' for i in range(1, window+1)] + [f'vol_return_{i}' for i in range(1, window+1)]
X = df[feature_cols]
y = df['label']

print("特征构建完成，数据集形状:", df.shape)

# ======== 任务3：严格时序数据划分 ========
# 1. 2024年数据：用于模型训练与评估（前80%训练，后20%测试）
data_2024_mask = df.index < "2025-01-01"
X_2024 = X[data_2024_mask]
y_2024 = y[data_2024_mask]

# 按时间顺序划分2024年数据为训练集和测试集（80% / 20%）
split_idx = int(len(X_2024) * 0.8)
X_train_eval = X_2024.iloc[:split_idx]
y_train_eval = y_2024.iloc[:split_idx]
X_test_eval = X_2024.iloc[split_idx:]
y_test_eval = y_2024.iloc[split_idx:]

# 2. 2025年数据：用于量化回测
data_2025_mask = df.index >= "2025-01-01"
X_2025 = X[data_2025_mask]
y_2025 = y[data_2025_mask]
price_return_2025 = df.loc[data_2025_mask, 'label'] # 实际次日收益（用于回测中计算资金变动）
dates_2025 = df.loc[data_2025_mask].index

# ======== 任务4：三大模型训练 + 评估 ========
print("\n正在进行模型评估 (2024年 80%训练, 20%测试)...")

# 初始化模型
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LightGBM': LGBMRegressor(random_state=42, verbose=-1) # verbose=-1 抑制输出
}

# 1. 2024年数据拆分，评估模型性能
for name, model in models.items():
    model.fit(X_train_eval, y_train_eval)
    preds = model.predict(X_test_eval)
    rmse = np.sqrt(mean_squared_error(y_test_eval, preds))
    mae = mean_absolute_error(y_test_eval, preds)
    print(f"{name} 评估结果 - RMSE: {rmse:.6f}, MAE: {mae:.6f}")

print("\n正在使用全部2024年数据训练最终模型...")
# 2. 用全部2024年数据训练最终模型
trained_models = {}
for name, model in models.items():
    model.fit(X_2024, y_2024)
    trained_models[name] = model

# 对2025年数据进行预测
predictions_2025 = {}
for name, model in trained_models.items():
    predictions_2025[name] = model.predict(X_2025)

# 3. 输出2025年测试集RMSE、MAE
print("\n2025年测试集模型表现:")
for name in trained_models.keys():
    preds = predictions_2025[name]
    rmse = np.sqrt(mean_squared_error(y_2025, preds))
    mae = mean_absolute_error(y_2025, preds)
    print(f"{name} - RMSE: {rmse:.6f}, MAE: {mae:.6f}")

# ======== 任务5：量化回测 ========
print("\n正在进行量化回测...")

# 回测参数
initial_capital = 1_000_000
fee_rate = 0.0003

# 辅助函数：计算最大回撤
def calculate_max_drawdown(capital_curve):
    if len(capital_curve) == 0: return 0
    rolling_max = pd.Series(capital_curve).cummax()
    drawdowns = (pd.Series(capital_curve) - rolling_max) / rolling_max
    return drawdowns.min()

backtest_results = {}

# 计算基准收益（买入持有策略）
baseline_capital = [initial_capital]
current_baseline = initial_capital * (1 - fee_rate) # 第一天满仓买入
for i in range(len(y_2025)):
    current_baseline = current_baseline * (1 + y_2025.values[i])
    baseline_capital.append(current_baseline)
# 最后一个元素对应最后一天结束，但为了和下面的资本曲线对齐，我们切片
baseline_capital = baseline_capital[:-1]

for name, preds in predictions_2025.items():
    capital = initial_capital
    position = 0
    capital_curve = []

    trades = 0
    winning_trades = 0

    actual_returns = y_2025.values

    for i in range(len(preds)):
        # 结算上一日决策带来的收益
        if position == 1:
            capital = capital * (1 + actual_returns[i])

        capital_curve.append(capital)

        # 今日决策
        pred_return = preds[i]
        if pred_return > 0.005 and position == 0:
            # 买入
            capital = capital * (1 - fee_rate)
            position = 1
            trades += 1
            if actual_returns[i] > 0:
                winning_trades += 1

        elif pred_return < -0.005 and position == 1:
            # 卖出
            capital = capital * (1 - fee_rate)
            position = 0

    final_capital = capital_curve[-1]
    cumulative_return = (final_capital - initial_capital) / initial_capital
    max_drawdown = calculate_max_drawdown(capital_curve)
    win_rate = winning_trades / trades if trades > 0 else 0

    backtest_results[name] = {
        'capital_curve': capital_curve,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trades': trades
    }

    print(f"\n{name} 回测结果:")
    print(f"累计收益: {cumulative_return * 100:.2f}%")
    print(f"最大回撤: {max_drawdown * 100:.2f}%")
    print(f"预测胜率 (基于买入决策): {win_rate * 100:.2f}%")
    print(f"交易次数: {trades}")

# ======== 任务6：结果可视化 ========
print("\n正在生成并保存图表...")

# 1. 绘制模型预测值vs真实值对比图 (取部分数据以更清晰展示)
plt.figure(figsize=(15, 6))
plt.plot(dates_2025, y_2025.values, label='真实值', color='black', linewidth=2)
plt.plot(dates_2025, predictions_2025['Linear Regression'], label='Linear Regression 预测', alpha=0.7)
plt.plot(dates_2025, predictions_2025['Random Forest'], label='Random Forest 预测', alpha=0.7)
plt.plot(dates_2025, predictions_2025['LightGBM'], label='LightGBM 预测', alpha=0.7)
plt.title(f'{stock_code} 2025年收益率预测值与真实值对比')
plt.xlabel('日期')
plt.ylabel('收益率')
plt.legend()
plt.grid(True)
# 格式化日期显示
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('1/results/predictions_vs_true.png')
plt.close()

# 2. 绘制2025年三大模型回测资金收益曲线
plt.figure(figsize=(15, 6))
# 绘制基准
baseline_returns = [(x - initial_capital)/initial_capital * 100 for x in baseline_capital]
plt.plot(dates_2025, baseline_returns, label='Buy and Hold 基准', color='black', linewidth=2, linestyle='--')

# 绘制各模型
colors = ['blue', 'green', 'red']
for (name, results), color in zip(backtest_results.items(), colors):
    returns_curve = [(x - initial_capital)/initial_capital * 100 for x in results['capital_curve']]
    plt.plot(dates_2025, returns_curve, label=f'{name} 策略', color=color, linewidth=1.5)

plt.title(f'{stock_code} 2025年各模型回测累计收益曲线')
plt.xlabel('日期')
plt.ylabel('累计收益率 (%)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('1/results/cumulative_returns.png')
plt.close()

print("全部任务完成！图表已保存至 1/results/ 目录下。")
