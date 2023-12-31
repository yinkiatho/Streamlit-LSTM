# LSTM and SMA SPY Trading Algorithm - Stock Price Prediction App
================

This Streamlit application is designed to help you explore and analyze the historical stock price data of various stocks using a combination of LSTM (Long Short-Term Memory) for stock price prediction and SMA (Simple Moving Average) for technical analysis. You can also assess the performance of a mean reversion strategy based on the LSTM predictions.

# LSTM Model
- Long-Short Term Memory Model build with tensorflow and keras.
- Built with time-series cross validation method as well as grid searching for parameter optimization
- Predicts next day Close price with window=5 past close prices
- ref. algotrader_streamlit.py
  

# Getting Started
To get started with the app, follow these simple steps:

1. Run https://yinkiatalgo.streamlit.app
2. Choose Stock Ticker: Select desired ticker from the dropdown menu to load the historical stock price data for the stock.


## Features:
- Data Overview
  
- Data Visualization: Explore various data visualizations to gain insights into stock's historical stock performance.
  
- Line Chart of Prices Over Time: Visualize historical closing, opening, high, and adjusted close prices for SPY stock over a 30-year timeframe.
  
- Line Chart of Volume Over Time: Examine the historical volume and ADV20 (average trading volume over the previous 20 trading days) for SPY stock over the same period. This helps you understand trading volume trends.
  
- SMA over Time: View Simple Moving Averages (SMA) of different periods (SMA5, SMA20, SMA50, SMA252) along with the adjusted close price to identify price trends.
  
- Investigating Returns: Analyze daily and monthly returns over the 30-year timeframe. Histograms are provided to visualize the distribution and central tendencies of these returns.
  
- Technical Analysis: Dive into technical indicators to assess stock's trading conditions.
 
- Stochastic Oscillator: Analyze the Stochastic Oscillator indicator.
  
- Relative Strength Index (RSI) & Commodity Channel Index (CCI): Examine RSI and CCI indicators.


### Data Overview:
This section provides a brief overview of the loaded data, including the number of rows, data types of each column, and summary statistics.
![alt text](https://github.com/yinkiatho/Streamlit-LSTM/blob/main/Pictures/data_overview.png)


### Data Visualisation:
![alt text](https://github.com/yinkiatho/Streamlit-LSTM/blob/main/Pictures/data_exploration.png)


### Line Chart of Prices Over Time: Visualize historical closing, opening, high, and adjusted close prices for SPY stock over a 30-year timeframe.
- Visualize historical closing, opening, high, and adjusted close prices for SPY stock over a 30-year timeframe.
![alt text](https://github.com/yinkiatho/Streamlit-LSTM/blob/main/Pictures/data_viz1.png)


### Line Chart of Volume over Time and Different SMAs over Time
- Examine the historical volume and ADV20 (average trading volume over the previous 20 trading days) for SPY stock over the same period. This helps you understand trading volume trends.
- View Simple Moving Averages (SMA) of different periods (SMA5, SMA20, SMA50, SMA252) along with the adjusted close price to identify price trends
![alt text](https://github.com/yinkiatho/Streamlit-LSTM/blob/main/Pictures/data_linechart2.png)


### Investigating Returns
Analyze daily and monthly returns over the 30-year timeframe. Histograms are provided to visualize the distribution and central tendencies of these returns.
![alt text](https://github.com/yinkiatho/Streamlit-LSTM/blob/main/Pictures/returns.png)


## Technical Analysis
 
### Stochastic Oscillator: Analyze the Stochastic Oscillator indicator.
![alt text](https://github.com/yinkiatho/Streamlit-LSTM/blob/main/Pictures/stochastic.png)

### Relative Strength Index (RSI) & Commodity Channel Index (CCI): Examine RSI and CCI indicators.
![alt text](https://github.com/yinkiatho/Streamlit-LSTM/blob/main/Pictures/stochastic.png)




# LSTM Prediction on Test Data: Explore the LSTM-based stock price prediction on the test data and assess its accuracy.

### LSTM Test Results
![alt text](https://github.com/yinkiatho/Streamlit-LSTM/blob/main/Pictures/lstm.png)

Metrics and Features:
- Transaction History and Profit: Review the transaction history and calculate the following performance metrics:

- Total Profit: The total profit generated by the trading strategy.

- Compound Annual Growth Rate (CAGR): The annualized rate of return.

- Sharpe Ratio: A measure of risk-adjusted return.


## Mean Reversion Strategy
- Calculate 3 day rolling closing price mean
- Compare with current closing price
- Long if > , short if <

![alt text](https://github.com/yinkiatho/Streamlit-LSTM/blob/main/Pictures/mean_reversion.png)


## LSTM Intra-Day Strategy
- Predict next day closing price
- If predicted next day closing price > today predicted closing price: buy at open and close position at close price next day
- If predicted next day closing price < today predicted closing price: short at open and close position at close price next day

![alt text](https://github.com/yinkiatho/Streamlit-LSTM/blob/main/Pictures/lstm_trading.png)


**How to run this demo**
To run this Streamlit app locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required Python packages listed in the requirements.txt file using pip install -r requirements.txt.
3. Run the app using the following command: 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
streamlit run app.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will launch the app in your web browser, allowing you to interact with the data and explore the trading algorithm's performance.

Feel free to contribute, provide feedback, or report any issues related to this app. Happy exploring and trading!
