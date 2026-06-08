import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load standardized data
df = pd.read_csv('../results/train_data_std.csv')
factor_cols = ['Reversal', 'Liquidity', 'MoneyFlow', 'Value']

def single_factor_regression(group):
    res = {}
    y = group['next_excess_ret'].values
    for col in factor_cols:
        X = group[col].values
        # Add constant
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X).fit()
            # If factor collinearity or exact zero variance, pvalues might not exist
            res[col + '_beta'] = model.params[1]
            res[col + '_pval'] = model.pvalues[1]
        except:
            res[col + '_beta'] = np.nan
            res[col + '_pval'] = np.nan
    return pd.Series(res)

reg_results = df.groupby('trade_date').apply(single_factor_regression, include_groups=False).dropna()

summary_results = []
for col in factor_cols:
    beta_mean = reg_results[col + '_beta'].mean()
    pval_mean = reg_results[col + '_pval'].mean()
    summary_results.append({
        'Factor': col,
        'Beta Mean': beta_mean,
        'p-value Mean': pval_mean,
        'Linear Effective': pval_mean < 0.05
    })

summary_df = pd.DataFrame(summary_results)
summary_df.to_csv('../results/OLS_single_factor_results.csv', index=False)

print("Single Factor OLS Results:")
print(summary_df)

print("Module 3 complete.")
