# AI for Market Trend Analysis Using LSTM

This project implements an AI-based system for analyzing and forecasting market trends using historical sales data.  
A Long Short-Term Memory (LSTM) neural network is used to learn temporal patterns and predict future demand.

## Dataset
The dataset used in this project is sourced from Kaggle and represents retail sales 
time-series data. The raw dataset is not included in the repository.

## Project Features
- Time-series market trend analysis
- LSTM neural network model
- Demand forecasting for future days
- Streamlit-based interactive deployment
- End-to-end AI pipeline

## Tech Stack
- Python
- Pandas, NumPy
- TensorFlow / Keras
- Matplotlib
- Streamlit

## Project Structure
- `notebooks/` – Data exploration and preprocessing
- `model/` – Trained LSTM model
- `app/` – Streamlit deployment
- `results/` – Output graphs and predictions
- `report/` – Final project report (PDF)

## How to Run
```bash
pip install -r requirements.txt
cd app
streamlit run streamlit_app.py

