from __future__ import division
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, MaxPooling1D, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class DualSMASignal():
    
    def __init__(self, window=5, data=None):
        self.algorithm = None
        self.model = None
        self.window = window
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.default_data = data
        self.default_test = None
        
    def generate_sma_signals(self, row_data=None):
        print("Generating SMA signals")
        print(row_data)
        signal = (row_data['SMA_20'] > row_data['SMA_252'])[0]
        date = (row_data['SMA_20'] > row_data['SMA_252']).index
        #Idea if SMA 5 > SMA 20, buy, else sell
        return [signal, date]
        
    def generate_lstm_signals(self, row_data):
        
        return
    
    def generate_dual_signals(self, data=None):
        return self.generate_sma_signals(data) and self.generate_lstm_signals(data)
        
    
