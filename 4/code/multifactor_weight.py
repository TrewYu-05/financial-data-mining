import pandas as pd
import statsmodels.api as sm

train_df = pd.read_csv("4/code/train_df_std.csv")
with open('4/code/valid_factors.txt', 'r') as f:
    valid_factors = f.read().split(',')

std_cols = [f'{f}_std' for f in valid_factors]

# 2. 截面迴歸建模 (Cross-sectional regression for multi-factor static weighting)
X = train_df[std_cols]
X = sm.add_constant(X)
y = train_df['next_excess_return']

model = sm.OLS(y, X, missing='drop').fit()
weights = model.params.iloc[1:] # Exclude intercept

weights_df = pd.DataFrame({'Factor': valid_factors, 'Weight': weights.values})
weights_df.to_csv("4/code/factor_weights.csv", index=False)

print("Weights calculated:")
print(weights_df)
