import pandas as pd
import numpy as np

# Load original data
train_f = pd.read_csv("4/data/train_factor.csv")
train_r = pd.read_csv("4/data/train_ret.csv")

# Merge
df = pd.merge(train_f, train_r, on=["trade_date", "stock_code"])

print("smb unique per date:")
print(df.groupby("trade_date")["smb"].nunique().value_counts())
