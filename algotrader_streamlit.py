from __future__ import division
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
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
from algorithm import DualSMASignal

default_data = pd.read_csv("Datasets/SPY_raw.csv",
                           index_col="Date", parse_dates=True)

default_new_data = pd.read_csv("Datasets/SPY_input.csv",)


class AlgoTrader():

    def __init__(self, window, ticker):
        
        self.models_dict = {
            
            "SPY": load_model('model_SPY.keras'),
            "MSFT": load_model('model_MSFT.keras'),
            "AAPL": load_model('model_AAPL.keras'),
            "GOOG": load_model('model_GOOG.keras')
        }
        self.algorithm = None
        self.model = None
        self.window = window
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.ticker = ticker
        self.default_data = pd.read_csv(f"Datasets/{ticker}_30years.csv",
                                        index_col="Date", parse_dates=True)
        self.default_test = []
        self.past_close = None
        self.default_past_close = None
        self.books = pd.DataFrame(
            columns=["Date", "Action", "Price", "Quantity"])
        self.cash = 1000000

    def load_lstm_model(self):
        #path = "/model_" + self.ticker + ".keras"
        #print(path)
        try:
        #    self.model = load_model(path)
            self.model = self.models_dict[self.ticker]
            print("Model loaded")

        except:
            print("Model not found, training model")
            self.train_LSTM()
            
    def streamlit_initilise(self):
        # Load model, build test_data and aggregations
        self.load_lstm_model()
        
        # Create a dataset of only the close column
        df = self.default_data['Close']
        dataset = df.values.reshape(-1, 1)

        # Scale dataset
        scaled_data = self.scaler.fit_transform(dataset)

        train, test = scaled_data[0:int(
            len(dataset) * 0.7), :], scaled_data[int(len(dataset) * 0.7):, :]

        train_x, train_y = self.reconstruct_data(train, 5)
        test_x, test_y = self.reconstruct_data(test, 5)
        self.default_test = [test_x, test_y]
        
        # Predicting Testing Dataset
        test_predict = self.model.predict(test_x)

        # print(test_predict.shape)
        predictions = self.scaler.inverse_transform(test_predict)
        test_y_real = self.scaler.inverse_transform(test_y.reshape(-1, 1))

        # Create dataset of actual close price and predicted close price
        final_df = pd.DataFrame({'Actual Close Price': test_y_real.flatten(
        ), 'Predicted Close Price': predictions.flatten()})
        # print(final_df)

        final_df.index = self.default_data[int(
            len(self.default_data) * 0.7)+5: -1].index

        self.past_close = final_df
        self.default_past_close = final_df
        self.default_test = [test_x, test_y]
        
        
        


    def load_algorithm(self, algorithm):
        self.algorithm = algorithm

    def load_training_data(self, startdate=datetime.datetime(1993, 1, 1),
                           enddate=datetime.datetime(2023, 1, 1)):

        GetData = yf.download(self.ticker, start=startdate, end=enddate)
        df = pd.DataFrame(GetData)
        # df.index = df.index.strftime('%Y-%m-%d')
        df = df.sort_index()

        # Build Columns
        df = df.assign(
            SMA_5=df['Close'].rolling(5).mean(),
            SMA_20=df['Close'].rolling(20).mean(),
            SMA_50=df['Close'].rolling(50).mean(),
            SMA_252=df['Close'].rolling(252).mean(),
            daily_return=(df['Close'].pct_change() * 100).round(2),
            monthly_return=(df['Close'].pct_change(30) * 100).round(2),
            adv20=df["Volume"].rolling(20).mean(),
            VWAP=(df['Volume'] * df['Close']).cumsum() / df['Volume'].cumsum()
        )
        # Add in log returns
        df['log_returns'] = np.log(df['Close']) - np.log(df['Close'].shift(1))

        # add in monthly volatility
        df['volatility_30'] = df['log_returns'].rolling(30).std() * np.sqrt(30)
        df['volatility_60'] = df['log_returns'].rolling(60).std() * np.sqrt(60)
        df['annual_volatility'] = df['log_returns'].rolling(
            252).std() * np.sqrt(252)

        # add in change in monthly vriation rapp
        df["rapp"] = df["Close"].shift(-21).divide(df["Close"])
        df.dropna(inplace=True)

        self.default_data = df

    def train_LSTM(self, window=5, n_splits=5):

        # if self.model is not None:
        #    print("Model already trained")
        #    return

        def model_lstm(window, features):
            model = Sequential()
            model.add(LSTM(150, return_sequences=True,
                           input_shape=(window, features)))
            model.add(LSTM(75, return_sequences=False))
            model.add(Dense(30))
            model.add(Dense(1))
            model.compile(loss='mse', optimizer='adam')

            return model

        # Create a dataset of only the close column
        df = self.default_data['Close']
        dataset = df.values.reshape(-1, 1)

        # Scale dataset
        scaled_data = self.scaler.fit_transform(dataset)

        train, test = scaled_data[0:int(
            len(dataset) * 0.7), :], scaled_data[int(len(dataset) * 0.7):, :]

        train_x, train_y = self.reconstruct_data(train, 5)
        test_x, test_y = self.reconstruct_data(test, 5)
        self.default_test = [test_x, test_y]

        tscv = TimeSeriesSplit(n_splits=n_splits)

        epochs = [50, 100]  # [50, 100]
        batch_sizes = [12, 24]  # [12, 24]
        results = {
            "epoch": [],
            "batch_size": [],
            "score": []
        }
        for batch_size in batch_sizes:
            for epoch in epochs:
                score_tracking = []
                for train_index, val_index in tscv.split(train):

                    model = model_lstm(window, 1)
                    train_data, val_data = train[:max(
                        train_index)], train[max(train_index):]

                    train_data_x, train_data_y = self.reconstruct_data(
                        train_data, 5)
                    val_data_x, val_data_y = self.reconstruct_data(val_data, 5)
                    # print(train_data_x.shape, train_data_y.shape)

                    train_data_x = np.reshape(
                        train_data_x, (train_data_x.shape[0], train_data_x.shape[1], 1))
                    val_data_x = np.reshape(
                        val_data_x, (val_data_x.shape[0], val_data_x.shape[1], 1))

                    model.fit(train_data_x, train_data_y, epochs=epoch, batch_size=batch_size,
                              validation_data=(val_data_x, val_data_y),
                              verbose=0, callbacks=[], shuffle=False)

                    test_loss = model.evaluate(test_x, test_y)
                    score_tracking.append(test_loss)

                Mean_Squared_Error = np.mean(score_tracking)
                # Append to dataframe
                results["epoch"].append(epoch)
                results["batch_size"].append(batch_size)
                results["score"].append(Mean_Squared_Error)

        # Get Best Params individual
        params = pd.DataFrame(results).sort_values(
            by=['score'], ascending=True, ignore_index=True)
        # params.head(10)

        # Constructing Final Model using best params
        model = model_lstm(window, 1)
        train_x = np.reshape(train_x, (train_x.shape[0],  train_x.shape[1], 1))
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
        model.fit(
            train_x, train_y, epochs=int(params.iloc[0, 0]), batch_size=int(params.iloc[0, 1]), verbose=2)

        self.model = model
        model.save("model_" + self.ticker + ".keras")

        # Predicting Testing Dataset
        # Predicting Testing Dataset
        test_predict = model.predict(test_x)

        # print(test_predict.shape)
        predictions = self.scaler.inverse_transform(test_predict)
        test_y_real = self.scaler.inverse_transform(test_y.reshape(-1, 1))

        # Create dataset of actual close price and predicted close price
        final_df = pd.DataFrame({'Actual Close Price': test_y_real.flatten(
        ), 'Predicted Close Price': predictions.flatten()})
        # print(final_df)

        final_df.index = self.default_data[int(
            len(self.default_data) * 0.7)+5: -1].index

        self.past_close = final_df
        self.default_past_close = final_df
        self.default_test = [test_x, test_y]
        # print(self.past_close)

    def reconstruct_data(self, data, n=1):
        data = pd.DataFrame(data)
        x, y = [], []
        for i in range(len(data) - n - 1):
            x.append(data.iloc[i:(i + n), :])
            # append close price
            y.append(data.iloc[i + n, 0])

        return (np.array(x), np.array(y))

    def reaggregate_data(self):
        # Build Columns
        df = self.default_data
        df = df.astype({"Close": float})
        df = df.assign(
            SMA_5=df['Close'].rolling(5).mean(),
            SMA_20=df['Close'].rolling(20).mean(),
            SMA_50=df['Close'].rolling(50).mean(),
            SMA_252=df['Close'].rolling(252).mean(),
            daily_return=(df['Close'].pct_change() * 100).round(2),
            monthly_return=(df['Close'].pct_change(30) * 100).round(2),
            adv20=df["Volume"].rolling(20).mean(),
            VWAP=(df['Volume'] * df['Close']).cumsum() / df['Volume'].cumsum()
        )
        # Add in log returns
        # print(df.info())
        df['log_returns'] = np.log(df['Close']) - np.log(df['Close'].shift(1))

        # add in monthly volatility
        df['volatility_30'] = df['log_returns'].rolling(30).std() * np.sqrt(30)
        df['volatility_60'] = df['log_returns'].rolling(60).std() * np.sqrt(60)
        df['annual_volatility'] = df['log_returns'].rolling(
            252).std() * np.sqrt(252)

        # add in change in monthly vriation rapp
        df["rapp"] = df["Close"].shift(-21).divide(df["Close"])
        # df.dropna(inplace=True)

        self.default_data = df

    def load_visualisations(self):
        test_x = self.default_test[0]
        test_y = self.default_test[1]

        # Predicting Testing Dataset
        test_predict = self.model.predict(test_x)
        # print(test_x.shape)
        # print(test_predict)
        # test_predict = scaler.inverse_transform(test_predict)
        # test_labels = scaler.inverse_transform(test_y)
        plt.figure(figsize=(30, 10))
        plt.plot(test_y, label="actual")
        plt.plot(test_predict, label="prediction")
        plt.legend(fontsize=20)
        plt.grid(axis="both")
        plt.title("Actual close price and pedicted one on test set", fontsize=25)
        plt.show()

    def add_new_row(self, row_data):
        row_data.index = row_data['Date']
        data_temp = pd.concat(
            [self.default_data, row_data.drop(columns=['Date'])], axis=0).iloc[1:, :]
        # data_temp.iloc[-1].index = row_data['Date']
        self.default_data = data_temp
        # print(data_temp)
        self.reaggregate_data()
        print("New row added")

    # Predict adds in rows and reaggregates data

    def predict(self, row_data):
        print("Predicting new row")
        # print(row_data)
        # row_data is a dataframe of one row
        self.add_new_row(row_data)
        data_temp = self.default_data
        # Create a dataset of only the close column

        df = data_temp['Close']
        dataset = df.values.reshape(-1, 1)

        # Scale dataset
        scaled_data = self.scaler.fit_transform(dataset)

        train, test = scaled_data[0:int(
            len(dataset) * 0.7), :], scaled_data[int(len(dataset) * 0.7):, :]

        # train_x, train_y = self.reconstruct_data(train, 5)
        test_x, test_y = self.reconstruct_data(test, 5)

        # Set new default_test_data
        self.default_test = [test_x, test_y]

        # Predict new data
        test_predict = self.scaler.inverse_transform(
            self.model.predict(test_x))
        # print(test_predict)
        return (test_predict[-1][0])

    def load_algorithm(self, algorithm):
        self.algorithm = algorithm

    def run_sma_algorithm(self, input_data=default_new_data):

        df = default_data
        past_close = self.default_past_close
        books = pd.DataFrame(
            columns=["Date", "Action", "Price", "Quantity"])
        date = None
        close = 0
        long, short = 0, 0

        for i in range(len(input_data)):

            # Read row_data
            row_data = pd.DataFrame(input_data.iloc[i, :]).transpose()
            print(f"Row {i}: ")
            # print(row_data)
            date = row_data.iloc[0, 0]
            close = row_data.iloc[0, 3]
            # print(f"Date: {row_data.iloc[0, 0]}")
            # print(f"Close Price: {row_data.iloc[0, 3]}")

            # Add predicted close price into default_data
            predicted_close = self.predict(row_data)
            print(pd.DataFrame(self.default_data.iloc[-1:, :]))
            signal = self.algorithm.generate_sma_signals(
                pd.DataFrame(self.default_data.iloc[-1:, :]))

            if predicted_close > past_close.iloc[-1, 1] and signal[0]:
                # Buy stock
                print(
                    f"Buying SPY500 on {date}, buying at {close}")
                self.cash -= close
                new_row = pd.DataFrame(
                    {"Date": date, "Action": "Buy", "Price": close, "Quantity": 1}, index=[0])
                books = pd.concat([books, new_row], axis=0, ignore_index=True)

                print(f"Current holdings: {books}")

                # add to past_close
                new_row = pd.DataFrame(
                    {"Actual Close Price": close, "Predicted Close Price": predicted_close}, index=[0])
                past_close = pd.concat(
                    [past_close, new_row], axis=0, ignore_index=True)

                long += 1

            elif predicted_close < past_close.iloc[-1, 1] and not signal[0]:
                # Sell stock
                print(
                    f"Selling SPY500 on {date}, selling at {close}")
                self.cash += close

                new_row = pd.DataFrame(
                    {"Date": date, "Action": "Sell", "Price": close, "Quantity": 1}, index=[0])
                books = pd.concat([books, new_row], axis=0, ignore_index=True)

                print(f"Current holdings: {books}")

                # add to past_close
                new_row = pd.DataFrame(
                    {"Actual Close Price": close, "Predicted Close Price": predicted_close}, index=[0])
                past_close = pd.concat(
                    [past_close, new_row], axis=0, ignore_index=True)

                short += 1

        # Clear holdings at the end of the day
        if long > short:
            for i in range(long - short):
                new_row = pd.DataFrame(
                    {"Date": date, "Action": "Sell", "Price": close, "Quantity": 1}, index=[0])
                books = pd.concat([books, new_row], axis=0, ignore_index=True)
                self.cash += close
        elif long < short:
            for i in range(short - long):
                new_row = pd.DataFrame(
                    {"Date": date, "Action": "Buy", "Price": close, "Quantity": 1}, index=[0])
                books = pd.concat([books, new_row], axis=0, ignore_index=True)
                self.cash -= close

        self.books = books
        self.past_close = past_close

        print("Trading completed")
        print(f"Final holdings: {books}")

    def run_mean_reversion_algorithm(self, input_data=default_new_data):

        df = default_data
        past_close = self.default_past_close
        books = pd.DataFrame(
            columns=["Date", "Action", "Price", "Quantity"])

        date = None
        close = 0
        long, short = 0, 0

        for i in range(len(input_data)):

            # Read row_data
            row_data = pd.DataFrame(input_data.iloc[i, :]).transpose()
            print(f"Row {i}: ")
            # print(row_data)
            date = row_data.iloc[0, 0]
            close = row_data.iloc[0, 3]
           # print(f"Date: {row_data.iloc[0, 0]}")
            # print(f"Close Price: {row_data.iloc[0, 3]}")
            long, short = 0, 0

            # Add predicted close price into default_data
            # predicted_close = self.predict(row_data)

            self.add_new_row(row_data)
            # print(pd.DataFrame(self.default_data.iloc[-1:, :]))
            # signal = self.algorithm.generate_sma_signals(
            #    pd.DataFrame(self.default_data.iloc[-1:, :]))

            # Execute Mean-Reversion Strategy
            if close < past_close.iloc[-1:-4:-1, 0].mean():
                # Buy stock
                print(
                    f"Buying SPY500 on {date}, buying at {close}")
                self.cash -= close
                new_row = pd.DataFrame(
                    {"Date": date, "Action": "Buy", "Price": close, "Quantity": 1}, index=[0])
                books = pd.concat([books, new_row], axis=0, ignore_index=True)

                # print(f"Current holdings: {books}")

                # add to past_close
                new_row = pd.DataFrame(
                    {"Actual Close Price": close, "Predicted Close Price": 0}, index=[0])
                past_close = pd.concat(
                    [past_close, new_row], axis=0, ignore_index=True)

                long += 1

            elif close < past_close.iloc[-1:-4:-1, 0].mean():
                # Sell stock
                print(
                    f"Selling SPY500 on {date}, selling at {close}")
                self.cash += close

                new_row = pd.DataFrame(
                    {"Date": date, "Action": "Sell", "Price": close, "Quantity": 1}, index=[0])
                books = pd.concat([books, new_row], axis=0, ignore_index=True)

                # print(f"Current holdings: {books}")

                # add to past_close
                new_row = pd.DataFrame(
                    {"Actual Close Price": close, "Predicted Close Price": 0}, index=[0])
                past_close = pd.concat(
                    [past_close, new_row], axis=0, ignore_index=True)

                short += 1

        # Clear holdings at the end of the day
        if long > short:
            for i in range(long - short):
                new_row = pd.DataFrame(
                    {"Date": date, "Action": "Sell", "Price": close, "Quantity": 1}, index=[0])
                books = pd.concat([books, new_row], axis=0, ignore_index=True)
                self.cash += close
        elif long < short:
            for i in range(short - long):
                new_row = pd.DataFrame(
                    {"Date": date, "Action": "Buy", "Price": close, "Quantity": 1}, index=[0])
                books = pd.concat([books, new_row], axis=0, ignore_index=True)
                self.cash -= close

        self.books = books
        self.past_close = past_close

        print("Trading completed")
        print(f"Final holdings: {books}")

    def tally_books(self):
        books = self.books
        total_profit = 0
        for i in range(len(books)):
            if books.iloc[i, 1] == "Buy":
                total_profit -= books.iloc[i, 2]
            else:
                total_profit += books.iloc[i, 2]

        books['Profit'] = 0
        position = 0

        for index, row in books.iterrows():
            if row['Action'] == 'Buy':
                position -= float(row['Quantity'] * row['Price'])
            elif row['Action'] == 'Sell':
                position += float(row['Quantity'] * row['Price'])
            books.at[index, 'Profit'] = position

        self.books = books
        print(f"Total profit: {total_profit}")

        return [total_profit, books]

    def plot_profit(self):
        books = self.books
        books.index = books['Date']

        books['Profit'].plot(figsize=(20, 10))
        plt.title('Profit over time', fontsize=20)
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Profit', fontsize=20)
        plt.grid(axis='both')
        plt.show()
