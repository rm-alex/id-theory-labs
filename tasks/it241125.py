import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os


df = pd.read_csv(os.path.abspath(os.path.join("tasks", "data.csv")))

X = df[["x_1","x_2","x_3","x_4","x_5","x_6","x_7"]]
y = df["y"]

model = LinearRegression()
model.fit(X, y)

print("Estimated parameters (coefficients):")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef}")

print("\nIntercept:", model.intercept_)

y_pred = model.predict(X)

print("\nR²:", r2_score(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))

residuals = y - y_pred
print("Residual mean:", residuals.mean())
print("Residual std:", residuals.std())