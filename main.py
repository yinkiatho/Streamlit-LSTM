from algotrader import AlgoTrader
from DualSMASignal import DualSMASignal

import pandas as pd
import numpy as np

default_new_data = pd.read_csv("Datasets/SPY_input.csv",)
new_row = pd.DataFrame(default_new_data.iloc[0, :]).transpose()
print(new_row)
default_data = pd.read_csv("Datasets/SPY_raw.csv",
                           index_col="Date", parse_dates=True)


if __name__ == '__main__':
    algotrader = AlgoTrader(window=5, ticker="SPY")
    print("Loading training data")
    algotrader.load_training_data()
    print("Training LSTM")
    algotrader.train_LSTM()
    #algotrader.load_visualisations()
    algotrader = AlgoTrader(window=5, ticker="AAPL")
    print("Loading training data")
    algotrader.load_training_data()
    print("Training LSTM")
    algotrader.train_LSTM()
    #algotrader.load_visualisations()
    algotrader = AlgoTrader(window=5, ticker="MSFT")
    print("Loading training data")
    algotrader.load_training_data()
    print("Training LSTM")
    algotrader.train_LSTM()
    #algotrader.load_visualisations()
    algotrader = AlgoTrader(window=5, ticker="GOOG")
    print("Loading training data")
    algotrader.load_training_data()
    print("Training LSTM")
    algotrader.train_LSTM()
    #algotrader.load_visualisations()
    
    
