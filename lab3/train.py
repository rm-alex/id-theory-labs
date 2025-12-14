import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

def get_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

df=pd.read_csv('lab3/processed_data.csv')

m, n = df.shape
idx = np.random.permutation(m)
P = 0.7

split = int(P * m)
train_df = df.iloc[idx[:split]]
test_df = df.iloc[idx[split:]]

X_train = train_df.drop(columns=['price'])
y_train = train_df['price']

X_test = test_df.drop(columns=['price'])
y_test = test_df['price']

model = RandomForestRegressor(
    max_depth=20,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred_model = model.predict(X_test)

rmse = get_rmse(y_test.values, y_pred_model)
print("Test RMSE:", rmse)

joblib.dump(model, 'lab3/model.joblib')

# from sklearn.model_selection import GridSearchCV
# rf = RandomForestRegressor(random_state=100, n_jobs=-1)
# param_rf = {'n_estimators': [100, 200], 
#             'max_depth': [10, 20, None], 
#             'min_samples_leaf': [1, 2], 
#             'min_samples_split': [2, 5], 
#             'max_features': ['sqrt', 'log2']}
# rf_grid = GridSearchCV(rf, 
#                        param_rf, 
#                        cv=4, 
#                        scoring='neg_root_mean_squared_error', 
#                        n_jobs=-1)
# rf_grid.fit(X_train, y_train)
# best_rf = rf_grid.best_estimator_
# print("Best RMSE:", -rf_grid.best_score_)
# print("Best params:", rf_grid.best_params_)
# y_pred_rf = best_rf.predict(X_test)
# rmse_rf = get_rmse(y_test.values, y_pred_rf)
# print("Test RMSE:", rmse_rf)