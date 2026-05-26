import pandas as pd
import numpy as np

train_df = pd.read_csv("4/code/train_df_filtered.csv")
with open('4/code/valid_factors.txt', 'r') as f:
    valid_factors = f.read().split(',')

def standardize(s):
    if s.std() == 0: return s - s.mean()
    return (s - s.mean()) / s.std()

# Standardize section by section
standardized_list = []
for month in train_df['year_month_str'].unique():
    sub = train_df[train_df['year_month_str'] == month].copy()
    for f in valid_factors:
        sub[f'{f}_std'] = standardize(sub[f])
    standardized_list.append(sub)

train_df = pd.concat(standardized_list)
train_df.to_csv("4/code/train_df_std.csv", index=False)

ic_results = []
for f in valid_factors:
    ic_list = []
    for month in train_df['year_month_str'].unique():
        sub = train_df[train_df['year_month_str'] == month]
        if len(sub) > 2:
            # Pearson correlation
            ic = sub[[f'{f}_std', 'next_excess_return']].corr().iloc[0,1]
            if not np.isnan(ic):
                ic_list.append(ic)

    ic_mean = np.mean(ic_list)
    ic_std = np.std(ic_list)
    ir = ic_mean / ic_std if ic_std != 0 else 0
    ic_results.append({
        'Factor': f,
        'IC_mean': ic_mean,
        'IC_std': ic_std,
        'IR': ir
    })

ic_df = pd.DataFrame(ic_results)
ic_df.to_csv("4/code/ic_ir_results.csv", index=False)
print(ic_df)
