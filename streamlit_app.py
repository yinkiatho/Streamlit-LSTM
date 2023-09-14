import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from algorithm import DualSMASignal
from algotrader_streamlit import AlgoTrader

st.title('LSTM and SMA SPY Trading ALgorithm')

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Define the Streamlit app
st.title("Stock Price Prediction App")

# Upload a CSV file containing historical stock data
ticker = st.selectbox("Choose stock ticker", ("SPY", "AAPL", "MSFT", "GOOG"))
default_training_data = pd.read_csv(f"Datasets/{ticker}_30years.csv", index_col="Date")

if default_training_data is not None:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #df = pd.read_csv(uploaded_file)
    st.subheader("Data Overview")
    st.write(default_training_data)
    
    #Data Exploration
    st.subheader("Data Exploration")
    st.write(default_training_data.describe())
    
    #Data Visualization
    st.subheader("Data Visualization")
    
    #plot grids of line charts for open, close, high, and adjusted close prices
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)
    metrics = ['Close', 'Open', 'High', 'Adj Close']
    colors = ['red', 'green', 'blue', 'black']

    st.subheader("Line Chart of Prices Over Time")
    st.write(f"Examining the historical closing, opening, high and adjusted close price for {ticker} stock across the 30 year timeframe.")
    for i, m in enumerate(metrics):
    # plot the metrics with colors
        plt.subplot(2, 2, i+1)
        default_training_data[m].plot(color=colors[i])
        plt.ylabel(m + " Price")
        plt.xlabel(None)
        plt.title(f"{m} Price of {ticker}")

    plt.tight_layout()
    st.pyplot()
    
    # Plotting Volume Data
    st.subheader("Line Chart of Volume Over Time")
    st.write(f"Examining the historical volume and adv20 for {ticker} stock across the 30 year timeframe. Volume refers to the total number of shares or contracts traded for a particular financial asset (e.g., stocks, bonds, commodities) within a specific time frame, typically a trading day. ADV20 is the average trading volume of a financial asset over the previous 20 trading days and provides a smoothed view of {ticker}'s volume data.")
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)
    vol_metrics = ['Volume', 'adv20']

    for i, m in enumerate(vol_metrics, 1):
        plt.subplot(2, 2, i)
        default_training_data[m].plot()
        plt.ylabel(m)
        plt.xlabel(None)
        plt.title(f"{m} for {ticker}")

    plt.tight_layout()
    st.pyplot()
    
    # Plotting MAs
    st.subheader("SMA over Time")
    st.write(f"SMA provides a smoothed view of the price trend for {ticker} by creating a constantly updated average price. In this plot, SMA5, SMA20, SMA50, SMA252 along with its adjusted close price is plotted to visualise the trends.")
    metrics = ['Adj Close', 'SMA_5', 'SMA_20', 'SMA_50', 'SMA_252']
    default_training_data[metrics].plot(figsize=(15, 10), title=f"Different SMAs for {ticker}")
    plt.gca().set_facecolor('white')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot()


    
    
    # Load new data
    default_new_data = pd.read_csv(f"Datasets/{ticker}_input.csv")
    
    
    #Loading and running the algorithm
    algotrader = AlgoTrader(5, ticker)
    algotrader.streamlit_initilise()
    #algotrader.load_visualisations()
    algotrader.load_algorithm(DualSMASignal())
    algotrader.run_mean_reversion_algorithm(default_new_data)
    
    
    total_profit, books = algotrader.tally_books()
    #algotrader.plot_results()
    
    
    # Display predictions and metrics
    st.subheader("Transaction History and Profit")
    st.write(books)
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

