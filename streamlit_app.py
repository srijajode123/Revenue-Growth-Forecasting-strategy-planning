import streamlit  as st
import pandas as pd
import pickle

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Revenue Forecasting App")

st.write("Provide the input features to predict the target sales:")

# Input fields for the features
sales = st.number_input("Sales", min_value=0.0, step=0.1)
market_indicator_1 = st.number_input("Market Indicator 1", min_value=0.0, step=0.1)
market_indicator_2 = st.number_input("Market Indicator 2", min_value=0.0, step=0.1)
gdp_growth = st.number_input("GDP Growth", min_value=-10.0, step=0.1)
unemployment_rate = st.number_input("Unemployment Rate", min_value=0.0, step=0.1)
inflation_rate = st.number_input("Inflation Rate", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
"sales": [sales],
        "market_indicator_1": [market_indicator_1],
        "market_indicator_2": [market_indicator_2],
        "gdp_growth": [gdp_growth],
        "unemployment_rate": [unemployment_rate],
        "inflation_rate": [inflation_rate]
    })

    # Make predictions
    prediction = model.predict(input_data)

    st.write(f"Predicted Target Sales: {prediction[0]:.2f}")  