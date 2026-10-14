import streamlit as st
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("simulated_financial_forecasting_data.csv")
X=df[["sales","market_indicator_1","market_indicator_2","gdp_growth","unemployment_rate","inflation_rate"]]
y=df["target_sales"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Save the model as a pickle file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

model.score(X_test, y_test)