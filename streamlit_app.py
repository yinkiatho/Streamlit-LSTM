import os
import streamlit as st
import pandas as pd
import pandas_profiling
import numpy as np
from math import sqrt
from keras.models import load_model
from streamlit_pandas_profiling import st_profile_report
from streamlit_extras.metric_cards import style_metric_cards



import matplotlib.pyplot as plt
from algorithm import DualSMASignal
from algotrader_streamlit import AlgoTrader

st.set_page_config(
    page_title="LSTM Stock Price Prediction App",
    page_icon="✅",
    layout="wide",
)
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


def sharpe(account_values: np.array, risk_free_rate, annualize_coefficient):
  # this gets our pct_return in the array
  diff = np.diff(account_values, 1) / account_values[1:]
  # we'll get the mean and calculate everything
  # we multply the mean of returns by the annualized coefficient and divide by annualized std
  annualized_std = diff.std() * sqrt(annualize_coefficient)
  return (diff.mean() * annualize_coefficient - risk_free_rate) / annualized_std

if default_training_data is not None:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #df = pd.read_csv(uploaded_file)
    st.subheader("Data Overview")
    st.write("The data is loaded and displayed below")
    st.write(default_training_data)
    
    pr = default_training_data.profile_report()
    
    st_profile_report(pr)
    
    #Data Exploration
    st.subheader("Data Exploration")
    st.write("The data is explored by looking at the first 5 rows of the data, the shape of the data, the data types of each column, and the summary statistics of the data.")
    st.write(default_training_data.describe())
    
    #Data Visualization
    st.subheader("Data Visualization")
    
    #plot grids of line charts for open, close, high, and adjusted close prices
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)
    metrics = ['Close', 'Open', 'High', 'Adj Close']
    colors = ['red', 'green', 'blue', 'black']

    st.markdown("### Line Chart of Prices Over Time")
    st.write(f"Examining the historical closing, opening, high and adjusted close price for {ticker} stock across the 30 year timeframe.")
    for i, m in enumerate(metrics):
    # plot the metrics with colors
        plt.subplot(2, 2, i+1)
        default_training_data[m].plot(color=colors[i])
        plt.ylabel(m + " Price")
        plt.xlabel(None)
        plt.gca().set_facecolor('white')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f"{m} Price of {ticker}")

    plt.tight_layout()
    st.pyplot()
    
    # Plotting Volume Data
    st.markdown("### Line Chart of Volume Over Time")
    st.write(f"Examining the historical volume and adv20 for {ticker} stock across the 30 year timeframe. Volume refers to the total number of shares or contracts traded for a particular financial asset (e.g., stocks, bonds, commodities) within a specific time frame, typically a trading day. ADV20 is the average trading volume of a financial asset over the previous 20 trading days and provides a smoothed view of {ticker}'s volume data.")
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)
    vol_metrics = ['Volume', 'adv20']

    for i, m in enumerate(vol_metrics, 1):
        plt.subplot(2, 2, i)
        default_training_data[m].plot()
        plt.ylabel(m)
        plt.xlabel(None)
        plt.gca().set_facecolor('white')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f"{m} for {ticker}")

    plt.tight_layout()
    st.pyplot()
    
    # Plotting MAs
    st.markdown("### SMA over Time")
    st.write(f"SMA provides a smoothed view of the price trend for {ticker} by creating a constantly updated average price. In this plot, SMA5, SMA20, SMA50, SMA252 along with its adjusted close price is plotted to visualise the trends.")
    metrics = ['Adj Close', 'SMA_5', 'SMA_20', 'SMA_50', 'SMA_252']
    default_training_data[metrics].plot(figsize=(15, 10), title=f"Different SMAs for {ticker}")
    plt.gca().set_facecolor('white')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot()
    
    # Plotting Daily Returns
    st.markdown("### Investigating Returns")
    st.write(
        f"Daily returns and Monthly returns are plotted across the 30year time frame to visualise its trends, along with histograms to visualise the dispersion and central tendencies of each return metric.")
    
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)
    colors = ['red', 'blue']

    metrics = ['daily_return', 'monthly_return']
    for i, m in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        default_training_data[m].hist(bins=100, color=colors[i-1])
        plt.ylabel(m)
        plt.xlabel(None)
        plt.title(f"Histogram of {m} for {ticker}")
        
    plt.tight_layout()
    st.pyplot()


    for (i, m) in enumerate(metrics, 1):
        plt.figure(figsize=(15, 10))
        plt.ylabel(m)
        plt.title(f"Average {m} for {ticker}")
        plt.subplots_adjust(top=1.5, bottom=1.2)
        default_training_data[m].plot(figsize=(15, 10), color=colors[i-1])
        
    plt.tight_layout()
    st.pyplot()
    
    # Load new data
    default_new_data = pd.read_csv(f"Datasets/{ticker}_input.csv")
    
    
    #Loading and running the algorithm
    algotrader = AlgoTrader(5, ticker)
    algotrader.streamlit_initilise()
    #algotrader.load_visualisations()
    
    st.markdown("### LSTM Prediction on Test Data")
    test_x = algotrader.default_test[0]
    test_y = algotrader.default_test[1]

    # Predicting Testing Dataset
    test_predict = algotrader.model.predict(test_x)
    # print(test_x.shape)
    # print(test_predict)
    # test_predict = scaler.inverse_transform(test_predict)
    # test_labels = scaler.inverse_transform(test_y)
    plt.figure(figsize=(30, 10))
    plt.plot(test_y, label="actual")
    plt.plot(test_predict, label="prediction")
    plt.legend(fontsize=20)
    plt.grid(axis="both")
    plt.title("Actual Close Price and Predicted Price on test set", fontsize=25)
    st.pyplot()
    
    
    
    #tab1, tab2, tab3 = st.tabs(["SMA + LSTM Prediction on Input Data", "Mean Reversion Strategy on Input Data", "LSTM Strategy on Input Data"])
    tab2, tab3 = st.tabs([
                         "Mean Reversion Strategy on Input Data", "LSTM Strategy on Input Data"])
    #with tab1:
    #    st.subheader("SMA + LSTM Prediction on Input Data")
    #    st.write("Momentum trading strategy, whereby we long when SMA5 is greater than SMA20, coupled with its corresponding signal from the LSTM model. We short when SMA5 is less than SMA20, coupled with its corresponding signal from the LSTM model.")
    #    algotrader.load_algorithm(DualSMASignal())
    #    algotrader.run_sma_algorithm_two(default_new_data)

    #    total_profit, books = algotrader.tally_books()
        # algotrader.plot_results()

        # Display predictions and metrics
    #    st.subheader("Transaction History and Profit")
    #    st.write(books)
    #    st.subheader("Total Profit")
    #    st.write(total_profit)

        # Plot the actual vs. predicted stock prices
    #    fig, ax = plt.subplots(figsize=(8, 6))
    #    ax.plot(books.Date, books.Profit, label="Profit Level ")
    #    ax.set_xlabel("Date")
    #    ax.set_ylabel("Profit")
    #    ax.set_title("Profit vs. Date")
    #    ax.legend()
    #    st.pyplot(fig)
        
        
    with tab2:
    
        st.subheader("Mean Reversion Strategy on Input Data")
        st.write("Classic 3 Day Mean Reversion Strategy, longing stock when close price falls below the 3 day moving average and shorting when it rises above the 3 day moving average.")
        algotrader = AlgoTrader(5, ticker)
        algotrader.streamlit_initilise()
        algotrader.load_algorithm(DualSMASignal())
        algotrader.run_mean_reversion_algorithm(default_new_data)
    
        total_profit, books = algotrader.tally_books()
        # algotrader.plot_results()

        # Display predictions and metrics
        st.markdown("### Transaction History and Profit")
        st.dataframe(books)
        #st.subheader("Total Profit")
        #st.write(total_profit)
        
        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Total Profit ＄",
            value=round(total_profit),
            #delta=round(total_profit) - 10,
        )

        kpi2.metric(
            label="CAGR %",
            value=round((((total_profit + 1000000)/(1000000)) ** (1.0/(175/365)) - 1)*100, 2),
            #delta=-10 + count_married,
        )
        
        account_values = np.array(books.Profit)
        

        kpi3.metric(
            label="Sharpe Ratio",
            value=f"{sharpe(account_values, 0.04, 252):.2f}",
            delta= round((sharpe(account_values, 0.04, 252) - 1.25), 2)
        )
        
        style_metric_cards()

        st.markdown("### Profit vs. Date")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(books.Date, books.Profit, label="Profit Level ")
        ax.set_xlabel("Date")
        ax.set_ylabel("Profit")
        ax.set_title("Profit vs. Date")
        ax.legend()
        st.pyplot(fig)
        
    with tab3:
    
        st.subheader("LSTM Strategy on Input Data")
        st.write("Long-Short Intra-Day strategy - For the prediction for each stock on the next day, if the prediction is positive, we buy the stock at the open price and sell the stock at close price in the same day. If the prediction is negative, we short-sell the stock at the open price and close out the short-sell at the close price.")
        algotrader = AlgoTrader(5, ticker)
        algotrader.streamlit_initilise()
        algotrader.load_algorithm(DualSMASignal())
        algotrader.run_lstm_algorithm(default_new_data)

        total_profit, books = algotrader.tally_books()
        # algotrader.plot_results()

        # Display predictions and metrics
        st.subheader("Transaction History and Profit")
        st.dataframe(books)
        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Total Profit ＄",
            value=round(total_profit),
            #delta=round(total_profit) - 10,
        )

        kpi2.metric(
            label="CAGR %",
            value=round((((total_profit + 1000000)/(1000000)) ** (1.0/(175/365)) - 1)*100, 2),
            # delta=-10 + count_married,
        )

        account_values = np.array(books.Profit)

        kpi3.metric(
            label="Sharpe Ratio",
            value=f"{sharpe(account_values, 0.04, 252):.2f}",
            delta=round((sharpe(account_values, 0.04, 252) - 1.25), 2)
        )
        
        style_metric_cards()

        st.markdown("### Profit vs. Date")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(books.Date, books.Profit, label="Profit Level ")
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

