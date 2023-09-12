from algotrader import AlgoTrader
from algorithm import DualSMASignal

import pandas as pd
import numpy as np

default_new_data = pd.read_csv("RNN Trading/Datasets/SPY_input.csv",)
new_row = pd.DataFrame(default_new_data.iloc[0, :]).transpose()
print(new_row)


if __name__ == '__main__':
    algotrader = AlgoTrader()
    algotrader.load_training_data()
    algotrader.train_LSTM()
    #algotrader.load_visualisations()
    algotrader.load_algorithm(DualSMASignal())
    algotrader.run_mean_reversion_algorithm(default_new_data)
    algotrader.tally_books()
    algotrader.plot_results()
    
