"""
敏感性分析实验 2b — 统计周期敏感性
===================================
控制变量：除训练/回测周期外，所有参数与基准实验保持一致。
  基准: 2024 训练 → 2025 回测
  实验: 2022 训练 → 2023 回测
  固定: 股票 sh.600519, window = 10 天
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensitivity_utils import run_sensitivity_experiment

config = {
    'stock_code': 'sh.600519',
    'fetch_start_date': '2022-01-01',       # ← 改变
    'fetch_end_date': '2023-12-31',         # ← 改变
    'train_end_date': '2023-01-01',         # ← 改变
    'backtest_label': '2023',               # ← 改变
    'window': 10,                            # 保持与基准一致
    'output_dir': 'results/sensitivity_period_2022_2023',
    'experiment_title': 'SA2b: 2022训练→2023回测',
}

if __name__ == '__main__':
    run_sensitivity_experiment(config)
