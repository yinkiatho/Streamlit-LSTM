import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from algorithm import DualSMASignal
from algotrader import AlgoTrader

st.title('LSTM and SMA SPY Trading ALgorithm')

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

default_new_data = pd.read_csv("Datasets/SPY_input.csv",)

# Define the Streamlit app
st.title("Stock Price Prediction App")

# Upload a CSV file containing historical stock data
uploaded_file = st.multiselect("Choose stock ticker", ["SPY"])
default_training_data = pd.read_csv("Datasets/SPY_30years.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Overview")
    st.write(default_training_data)
    
    
    #Loading and running the algorithm
    algotrader = AlgoTrader()
    algotrader.load_training_data(default_training_data)
    algotrader.train_LSTM()
    #algotrader.load_visualisations()
    algotrader.load_algorithm(DualSMASignal())
    algotrader.run_mean_reversion_algorithm(default_new_data)
    total_profit, books = algotrader.tally_books()
    #algotrader.plot_results()
    
    
    # Display predictions and metrics
    st.subheader("Transaction History and Profit")
    st.write(pd.DataFrame(books))
    st.subheader("Total Profit")
    st.write(total_profit)

    # Plot the actual vs. predicted stock prices
    fig, ax = plt.subplots()
    ax.plot(books.Profit, label="Actual Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Profit")
    ax.set_title("Profit vs. Date")
    ax.legend()
    st.pyplot(fig)

    # Calculate profits or any other relevant metrics here
    # You can add more sections to display additional charts and tables

# Optionally, add a sidebar to customize model parameters or other settings
# st.sidebar.header("Model Configuration")
# model_parameter = st.sidebar.slider("Parameter Name", min_value, max_value, default_value)

# Add any other customization and enhancements you need for your specific application

