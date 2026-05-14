"""
敏感性分析公共工具模块
提供参数化的实验执行函数，三个敏感性分析脚本共用此模块。
"""
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

# 按优先级尝试常见中文字体，适配不同操作系统
import matplotlib.font_manager as fm
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_font_candidates = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans SC']
_selected_font = 'DejaVu Sans'
for _f in _font_candidates:
    if _f in _available_fonts:
        _selected_font = _f
        break
plt.rcParams['font.sans-serif'] = [_selected_font, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
print(f"[font] 使用字体: {_selected_font}")


def calculate_max_drawdown(capital_curve):
    """计算最大回撤"""
    if len(capital_curve) == 0:
        return 0
    rolling_max = pd.Series(capital_curve).cummax()
    drawdowns = (pd.Series(capital_curve) - rolling_max) / rolling_max
    return drawdowns.min()


def run_sensitivity_experiment(config):
    """
    运行一个完整的敏感性分析实验（数据获取→特征构建→训练→回测→可视化）。

    Parameters
    ----------
    config : dict
        stock_code       : str  - 股票代码，如 'sh.600519'
        fetch_start_date : str  - 数据获取起始日期 'YYYY-MM-DD'
        fetch_end_date   : str  - 数据获取结束日期 'YYYY-MM-DD'
        train_end_date   : str  - 训练/回测分界日期，此日期之前为训练集
        backtest_label   : str  - 回测期标签（用于图表标题），如 '2025'
        window           : int  - 滚动窗口天数
        output_dir       : str  - 结果输出目录
        experiment_title : str  - 实验标题（用于图表）
    """
    stock_code = config['stock_code']
    fetch_start = config['fetch_start_date']
    fetch_end = config['fetch_end_date']
    train_end = config['train_end_date']
    backtest_label = config['backtest_label']
    window = config['window']
    output_dir = config['output_dir']
    exp_title = config['experiment_title']

    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # 1. 数据获取
    # ================================================================
    print(f"\n{'='*60}")
    print(f"敏感性分析实验: {exp_title}")
    print(f"股票: {stock_code}, 窗口: {window}天")
    print(f"数据范围: {fetch_start} ~ {fetch_end}, 训练截止: {train_end}")
    print(f"{'='*60}\n")

    print("正在登录 Baostock ...")
    bs.login()

    print(f"正在获取 {stock_code} 数据 ...")
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,close,volume",
        start_date=fetch_start, end_date=fetch_end,
        frequency="d", adjustflag="3"
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    df = pd.DataFrame(data_list, columns=rs.fields)
    bs.logout()

    print("数据获取完成，正在进行数据清洗 ...")
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # ================================================================
    # 2. 时序特征构建
    # ================================================================
    print("正在构建时序特征 ...")
    df['price_return'] = df['close'].pct_change()
    df['vol_return'] = df['volume'].pct_change()
    df['label'] = df['price_return'].shift(-1)

    for i in range(1, window + 1):
        df[f'price_return_{i}'] = df['price_return'].shift(i)
        df[f'vol_return_{i}'] = df['vol_return'].shift(i)

    df = df.dropna()

    feature_cols = (
        [f'price_return_{i}' for i in range(1, window + 1)]
        + [f'vol_return_{i}' for i in range(1, window + 1)]
    )
    X = df[feature_cols]
    y = df['label']

    print(f"特征构建完成，数据集形状: {df.shape}, 特征数: {len(feature_cols)}")

    # ================================================================
    # 3. 训练 / 回测划分
    # ================================================================
    train_mask = df.index < train_end
    X_train_all = X[train_mask]
    y_train_all = y[train_mask]

    backtest_mask = df.index >= train_end
    X_backtest = X[backtest_mask]
    y_backtest = y[backtest_mask]
    backtest_dates = df.loc[backtest_mask].index

    print(f"训练集: {X_train_all.shape[0]} 条, 回测集: {X_backtest.shape[0]} 条")

    # 训练集内 80/20 划分（用于初步评估）
    split_idx = int(len(X_train_all) * 0.8)
    X_train_eval = X_train_all.iloc[:split_idx]
    y_train_eval = y_train_all.iloc[:split_idx]
    X_test_eval = X_train_all.iloc[split_idx:]
    y_test_eval = y_train_all.iloc[split_idx:]

    # ================================================================
    # 4. 模型训练 & 初步评估
    # ================================================================
    print(f"\n--- 模型评估 (80%训练, 20%测试) ---")

    model_factories = {
        'Linear Regression': lambda: LinearRegression(),
        'Random Forest': lambda: RandomForestRegressor(
            n_estimators=50, max_depth=None, min_samples_split=2, random_state=42
        ),
        'LightGBM': lambda: LGBMRegressor(
            max_depth=3, learning_rate=0.1, n_estimators=200, random_state=42, verbose=-1
        ),
    }

    eval_metrics = {}
    for name, factory in model_factories.items():
        m = factory()
        m.fit(X_train_eval, y_train_eval)
        preds = m.predict(X_test_eval)
        rmse = np.sqrt(mean_squared_error(y_test_eval, preds))
        mae = mean_absolute_error(y_test_eval, preds)
        eval_metrics[name] = {'RMSE': rmse, 'MAE': mae}
        print(f"  {name:25s}  RMSE={rmse:.6f}  MAE={mae:.6f}")

    # 用全部训练数据重新训练
    print(f"\n--- 使用全部训练数据重新训练 ---")
    trained_models = {}
    for name, factory in model_factories.items():
        m = factory()
        m.fit(X_train_all, y_train_all)
        trained_models[name] = m

    # 回测期预测
    predictions_backtest = {}
    for name, model in trained_models.items():
        predictions_backtest[name] = model.predict(X_backtest)

    print(f"\n--- {backtest_label} 回测期预测误差 ---")
    backtest_metrics = {}
    for name in trained_models:
        preds = predictions_backtest[name]
        rmse = np.sqrt(mean_squared_error(y_backtest, preds))
        mae = mean_absolute_error(y_backtest, preds)
        backtest_metrics[name] = {'RMSE': rmse, 'MAE': mae}
        print(f"  {name:25s}  RMSE={rmse:.6f}  MAE={mae:.6f}")

    # ================================================================
    # 5. 量化回测
    # ================================================================
    print(f"\n--- 量化回测 ---")

    initial_capital = 1_000_000
    fee_rate = 0.0003

    backtest_results = {}

    # 基准：买入持有
    baseline_capital = [initial_capital]
    current = initial_capital * (1 - fee_rate)
    for ret in y_backtest.values:
        current *= (1 + ret)
        baseline_capital.append(current)
    baseline_capital = baseline_capital[:-1]

    for name, preds in predictions_backtest.items():
        capital = initial_capital
        position = 0
        capital_curve = []
        trades = 0
        winning_trades = 0
        actual_returns = y_backtest.values

        for i in range(len(preds)):
            if position == 1:
                capital *= (1 + actual_returns[i])
            capital_curve.append(capital)

            pred_return = preds[i]
            if pred_return > 0.005 and position == 0:
                capital *= (1 - fee_rate)
                position = 1
                trades += 1
                if actual_returns[i] > 0:
                    winning_trades += 1
            elif pred_return < -0.005 and position == 1:
                capital *= (1 - fee_rate)
                position = 0

        final_capital = capital_curve[-1]
        cumulative_return = (final_capital - initial_capital) / initial_capital
        max_drawdown = calculate_max_drawdown(capital_curve)
        max_return = (pd.Series(capital_curve).max() - initial_capital) / initial_capital
        win_rate = winning_trades / trades if trades > 0 else 0

        backtest_results[name] = {
            'capital_curve': capital_curve,
            'cumulative_return': cumulative_return,
            'max_return': max_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': trades,
        }

        print(f"\n  {name}")
        print(f"    累计收益: {cumulative_return * 100:+.2f}%")
        print(f"    最大回撤: {max_drawdown * 100:.2f}%")
        print(f"    预测胜率: {win_rate * 100:.2f}%")
        print(f"    交易次数: {trades}")

    # ================================================================
    # 6. 可视化
    # ================================================================
    print(f"\n--- 生成图表 ---")

    # 6.1 预测值 vs 真实值
    plt.figure(figsize=(15, 6))
    plt.plot(backtest_dates, y_backtest.values, label='真实值', color='black', linewidth=2)
    plt.plot(backtest_dates, predictions_backtest['Linear Regression'],
             label='Linear Regression', alpha=0.7)
    plt.plot(backtest_dates, predictions_backtest['Random Forest'],
             label='Random Forest', alpha=0.7)
    plt.plot(backtest_dates, predictions_backtest['LightGBM'],
             label='LightGBM', alpha=0.7)
    plt.title(f'{exp_title}\n{backtest_label} 收益率预测值与真实值对比')
    plt.xlabel('日期')
    plt.ylabel('收益率')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_true.png'), dpi=150)
    plt.close()

    # 6.2 累计收益曲线
    plt.figure(figsize=(15, 6))
    baseline_returns = [(x - initial_capital) / initial_capital * 100 for x in baseline_capital]
    plt.plot(backtest_dates, baseline_returns,
             label='Buy & Hold 基准', color='black', linewidth=2, linestyle='--')

    colors = ['#2196F3', '#4CAF50', '#F44336']
    for (name, results), color in zip(backtest_results.items(), colors):
        rets = [(x - initial_capital) / initial_capital * 100 for x in results['capital_curve']]
        plt.plot(backtest_dates, rets, label=f'{name}', color=color, linewidth=1.5)

    plt.title(f'{exp_title}\n{backtest_label} 各模型回测累计收益曲线')
    plt.xlabel('日期')
    plt.ylabel('累计收益率 (%)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'), dpi=150)
    plt.close()

    # 6.3 RMSE / MAE 柱状图
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    model_names = list(backtest_metrics.keys())
    rmse_vals = [backtest_metrics[m]['RMSE'] for m in model_names]
    mae_vals = [backtest_metrics[m]['MAE'] for m in model_names]

    ax[0].bar(model_names, rmse_vals, color=colors, alpha=0.7)
    ax[0].set_title(f'{exp_title}\n{backtest_label} RMSE 对比')
    ax[0].set_ylabel('RMSE')
    ax[0].tick_params(axis='x', rotation=15)

    ax[1].bar(model_names, mae_vals, color=colors, alpha=0.7)
    ax[1].set_title(f'{exp_title}\n{backtest_label} MAE 对比')
    ax[1].set_ylabel('MAE')
    ax[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_rmse_mae.png'), dpi=150)
    plt.close()

    # 6.4 收益 & 回撤柱状图
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    model_names = list(backtest_results.keys())
    final_rets = [backtest_results[m]['cumulative_return'] * 100 for m in model_names]
    max_rets = [backtest_results[m]['max_return'] * 100 for m in model_names]
    max_dds = [backtest_results[m]['max_drawdown'] * 100 for m in model_names]

    ax[0].bar(model_names, final_rets, color=colors, alpha=0.7)
    ax[0].set_title(f'{exp_title}\n最终收益率 (%)')
    ax[0].set_ylabel('%')
    ax[0].tick_params(axis='x', rotation=15)

    ax[1].bar(model_names, max_rets, color=colors, alpha=0.7)
    ax[1].set_title(f'{exp_title}\n最大收益率 (%)')
    ax[1].set_ylabel('%')
    ax[1].tick_params(axis='x', rotation=15)

    ax[2].bar(model_names, max_dds, color=colors, alpha=0.7)
    ax[2].set_title(f'{exp_title}\n最大回撤 (%)')
    ax[2].set_ylabel('%')
    ax[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_returns_drawdown.png'), dpi=150)
    plt.close()

    print(f"\n{exp_title} 全部图表已保存至 {output_dir}\n")

    return {
        'eval_metrics': eval_metrics,
        'backtest_metrics': backtest_metrics,
        'backtest_results': backtest_results,
        'config': config,
    }
