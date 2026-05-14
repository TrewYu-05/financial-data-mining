"""
敏感性分析实验 1 — 滑动窗口敏感性
===================================
控制变量：除窗口大小外，所有参数与基准实验保持一致。
  基准: window = 10 天
  实验: window = 20 天
  固定: 股票 sh.600519, 训练期 2024, 回测期 2025
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensitivity_utils import run_sensitivity_experiment

config = {
    'stock_code': 'sh.600519',
    'fetch_start_date': '2024-01-01',
    'fetch_end_date': '2025-12-31',
    'train_end_date': '2025-01-01',
    'backtest_label': '2025',
    'window': 20,                           # ← 唯一改变：10 → 20
    'output_dir': 'results/sensitivity_window20',
    'experiment_title': 'SA1: 窗口=20天 (2024训练→2025回测)',
}

if __name__ == '__main__':
    run_sensitivity_experiment(config)
