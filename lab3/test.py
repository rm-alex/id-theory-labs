import pandas as pd
import joblib

saved_model = joblib.load('model.joblib')
df_test = pd.read_csv('processed_test_data.csv')

y_test_pred = saved_model.predict(df_test)

price_table = pd.DataFrame({
    'index': range(len(y_test_pred)),
    'price': y_test_pred
})

price_table.to_csv('test_pred.csv', index=False)