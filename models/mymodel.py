import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from utility.fetch_ import download_stock_data,nifty50_tickers_ns

def stock_prediction(ticker=nifty50_tickers_ns, start_date="2024-01-01", end_date="2024-12-31", download_bool=0, interval=["1d", "1mo"], filename=["time_series.csv", "time_series_monthly.csv"], use_second_model=False):
    mymodelprediction_df =[]
    for j in range(len(ticker)):
        dd, dm = download_stock_data(download_bool,ticker[j], start_date, end_date, interval, filename)
        df = dd.sort_values('Date')

        # Feature selection
        data = df[['Open', 'High', 'Low']]

        # Scaling the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Preparing training data
        X_train = []
        y_train_high = []
        y_train_low = []

        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i-60:i, 0])  # Open prices
            y_train_high.append(scaled_data[i, 1])  # High price
            y_train_low.append(scaled_data[i, 2])   # Low price

        X_train = np.array(X_train)
        y_train_high = np.array(y_train_high)
        y_train_low = np.array(y_train_low)

        # Reshaping for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Building the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=25))
        model.add(Dense(units=2))  # Two outputs for High and Low prices

        # Compile the model
        optimizer = Adam(learning_rate=0.0001) if use_second_model else 'adam'
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)

        # Train the model
        model.fit(X_train, np.column_stack((y_train_high, y_train_low)), epochs=500 if use_second_model else 10, batch_size=32, callbacks=[early_stopping] if use_second_model else None)

        # Get today's open price and previous data
        todays_open = df.iloc[-1]['Open']
        previous_data = df[['Open', 'High', 'Low']].values[-60:]

        # Predict High and Low
        test_data = np.vstack((previous_data, [todays_open, 0, 0]))  # Replace NaN with 0 for scaling
        test_data_scaled = scaler.transform(test_data)
        test_data_scaled_input = test_data_scaled[-60:, 0].reshape(1, -1, 1)

        predicted_high_low_scaled = model.predict(test_data_scaled_input)

        # Inverse scaling to get actual predicted values
        dummy_row = np.zeros((1, 3))
        dummy_row[0, 0] = todays_open
        dummy_row[0, 1] = predicted_high_low_scaled[0][0]
        dummy_row[0, 2] = predicted_high_low_scaled[0][1]

        predicted_high_low = scaler.inverse_transform(dummy_row)

        predicted_high = predicted_high_low[0][1]
        predicted_low = predicted_high_low[0][2]
        mymodelprediction_df.append([ticker[j], predicted_high, predicted_low])
        print(f"{j} tick:{ticker[j]} | Predicted High: {predicted_high} | Predicted Low: {predicted_low}")
        # Save predictions to CSV
    mymodelprediction = pd.DataFrame(mymodelprediction_df, columns=['Ticker', 'Predicted High', 'Predicted Low'])
    mymodelprediction.to_csv('./database/mymodelprediction.csv', index=False)
    return model, scaler, df, predicted_high, predicted_low

