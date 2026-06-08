import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import joblib

# Load standardized data
df = pd.read_csv('../results/train_data_std.csv')
factor_cols = ['Reversal', 'Liquidity', 'MoneyFlow', 'Value']

# Sort by trade_date to ensure proper time series splitting
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.sort_values('trade_date').reset_index(drop=True)

X = df[factor_cols]
y = df['next_excess_ret']

# 5-Fold TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

print("Starting 5-Fold TimeSeries Cross Validation...")
fold = 1
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    # Silent parameter is deprecated, using callbacks or log_evaluation=False isn't strictly needed for small datasets but let's just use defaults
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    print(f"Fold {fold} - Best Iteration: {model.best_iteration_}")
    fold += 1

# Train final model on full training set
print("Training final model on full training set...")
final_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
final_model.fit(X, y)

# Save the model
joblib.dump(final_model, '../results/lgbm_model.pkl')

# Feature Importance
importance = pd.DataFrame({
    'Factor': factor_cols,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

importance.to_csv('../results/lgbm_feature_importance.csv', index=False)

print("\nFeature Importance:")
print(importance)

print("Module 4 complete.")
