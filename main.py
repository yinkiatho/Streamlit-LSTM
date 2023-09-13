from algotrader import AlgoTrader
from algorithm import DualSMASignal

import pandas as pd
import numpy as np

default_new_data = pd.read_csv("Datasets/SPY_input.csv",)
new_row = pd.DataFrame(default_new_data.iloc[0, :]).transpose()
print(new_row)


if __name__ == '__main__':
    algotrader = AlgoTrader()
    print("Loading training data")
    algotrader.load_training_data()
    print("Training LSTM")
    algotrader.train_LSTM()
    #algotrader.load_visualisations()
    print("Loading algorithm")
    algotrader.load_algorithm(DualSMASignal())
    print("Running algorithm")
    algotrader.run_mean_reversion_algorithm(default_new_data)
    print("Tallying books")
    algotrader.tally_books()
    algotrader.plot_results()
    
