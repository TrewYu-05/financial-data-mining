import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# Load data from CSV (saved by module 1)
df = pd.read_csv('../results/train_data.csv')

# Drop rows with NaN in factors or next_excess_ret
factor_cols = ['Reversal', 'Liquidity', 'MoneyFlow', 'Value']
df = df.dropna(subset=factor_cols + ['next_excess_ret']).copy()

# 1. Z-Score Standardization
def z_score(series):
    return (series - series.mean()) / series.std()

df[factor_cols] = df.groupby('trade_date')[factor_cols].transform(z_score)
df = df.dropna(subset=factor_cols).copy()

# Save standardized data for later modules
df.to_csv('../results/train_data_std.csv', index=False)

# 2. Daily Cross-sectional Spearman IC
def calc_ic(group):
    # Calculate spearman correlation between each factor and next_excess_ret
    res = {}
    for col in factor_cols:
        res[col] = group[col].corr(group['next_excess_ret'], method='spearman')
    return pd.Series(res)

ic_series = df.groupby('trade_date')[factor_cols + ['next_excess_ret']].apply(calc_ic, include_groups=False).dropna()

# 3. Newey-West Significance Test, IC Mean, IC Std, IR
results = []
for col in factor_cols:
    ic = ic_series[col]
    mean_ic = ic.mean()
    std_ic = ic.std()
    ir = mean_ic / std_ic if std_ic != 0 else np.nan

    # Newey-West t-test
    # Regress IC on a constant
    if len(ic) > 1:
        model = sm.OLS(ic.values, np.ones(len(ic)))
        lags = int(4 * (len(ic)/100)**(2/9))
        nw_result = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        t_stat = nw_result.tvalues[0]
        p_val = nw_result.pvalues[0]
    else:
        t_stat, p_val = np.nan, np.nan

    results.append({
        'Factor': col,
        'IC Mean': mean_ic,
        'IC Std': std_ic,
        'IR': ir,
        't-stat (NW)': t_stat,
        'p-value (NW)': p_val
    })

ic_results_df = pd.DataFrame(results)

# 4. Collinearity Test (VIF)
def calc_vif(group):
    X = group[factor_cols].values
    vifs = []
    # If not enough features or variance is 0, VIF could fail
    if X.shape[0] > X.shape[1] + 1:
        X_df = pd.DataFrame(X, columns=factor_cols)
        X_df = sm.add_constant(X_df)
        try:
            for i in range(1, X_df.shape[1]):
                vifs.append(variance_inflation_factor(X_df.values, i))
        except:
            vifs = [np.nan] * len(factor_cols)
    else:
        vifs = [np.nan] * len(factor_cols)
    return pd.Series(vifs, index=factor_cols)

vif_series = df.groupby('trade_date')[factor_cols].apply(calc_vif, include_groups=False).dropna()
vif_mean = vif_series.mean()

vif_results_df = pd.DataFrame({'Factor': factor_cols, 'VIF Mean': vif_mean.values})

# Output results
ic_results_df.to_csv('../results/IC_IR_results.csv', index=False)
vif_results_df.to_csv('../results/VIF_results.csv', index=False)

print("IC/IR Results:")
print(ic_results_df)
print("\nVIF Results:")
print(vif_results_df)

print("Module 2 complete.")
