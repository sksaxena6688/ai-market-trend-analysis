import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="AI Demand Forecasting", layout="wide")

st.title("AI-Based Product Demand Forecasting")
st.write("LSTM Neural Network based time-series demand prediction")

@st.cache_data
def load_data():
    df = pd.read_csv("../data/train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

df = load_data()

store_id = st.selectbox("Select Store", df["store_nbr"].unique())
family = st.selectbox("Select Product Family", df["family"].unique())

forecast_days = st.slider("Forecast Days", 7, 60, 30)

ts_df = df[
    (df["store_nbr"] == store_id) &
    (df["family"] == family)
][["date", "sales"]].set_index("date")

st.subheader("Historical Sales Data")
st.line_chart(ts_df)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sales = scaler.fit_transform(ts_df[["sales"]])

model = load_model("../model/lstm_demand_forecast.h5")

def forecast_future(model, data, days):
    window_size = 30
    last_window = data[-window_size:]
    predictions = []

    for _ in range(days):
        pred = model.predict(last_window.reshape(1, window_size, 1), verbose=0)
        predictions.append(pred[0, 0])
        last_window = np.append(last_window[1:], pred, axis=0)

    return np.array(predictions)

if len(scaled_sales) < 30:
    st.warning("Not enough data for forecasting (need at least 30 days).")
else:
    future_scaled = forecast_future(model, scaled_sales, forecast_days)
    future_sales = scaler.inverse_transform(future_scaled.reshape(-1, 1))

    future_dates = pd.date_range(
        start=ts_df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame(
        future_sales,
        index=future_dates,
        columns=["Predicted Sales"]
    )

    st.subheader("Forecasted Demand")
    st.line_chart(forecast_df)

    st.subheader("Forecast Values")
    st.dataframe(forecast_df.round(2))
