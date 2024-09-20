import numpy as np
import pandas as pd
from utility.fetch_ import download_stock_data, nifty50_tickers_ns

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros((n, y.shape[1]))  # Two outputs: Next_High and Next_Low
    for epoch in range(epochs):
        predictions = X.dot(weights)
        errors = predictions - y
        
        # Calculate gradients
        gradients = X.T.dot(errors) / m
        
        # Update weights
        weights -= learning_rate * gradients
        
        if epoch % 100 == 0:
            mse = np.mean(errors**2)
            print(f'Epoch {epoch}: MSE = {mse}')
    
    return weights

def predict(ticker="tatasteel.ns", learning_rate=0.01, epochs=1000, start_date="2024-01-01", download_bool=0, end_date="2024-12-31", interval=["1d", "1mo"], filename=["time_series.csv", "time_series_monthly.csv"]):
        linearregression = []   
        dd, dm = download_stock_data(download_bool, ticker, start_date, end_date, interval, filename)

        data = dd
        # Create next day high and low columns
        data['Next_High'] = data['High'].shift(-1)
        data['Next_Low'] = data['Low'].shift(-1)
        data = data[:-1]  # Drop the last row with NaN values

        # Features and labels
        X = data[['Open', 'High', 'Low', 'Close']].values
        y = data[['Next_High', 'Next_Low']].values

        # Normalize the features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_normalized = (X - X_mean) / X_std

        # Normalize the target variables
        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0)
        y_normalized = (y - y_mean) / y_std

        # Train the model
        weights = gradient_descent(X_normalized, y_normalized, learning_rate, epochs)

        # Predict the next day's high and low
        last_day_normalized = X_normalized[-1].reshape(1, -1)
        predicted_next_normalized = last_day_normalized.dot(weights)

        # Reverse normalization
        predicted_next = predicted_next_normalized * y_std + y_mean

        print(f'Predicted Next Day High: {predicted_next[0][0]} | Actual Last Day High: {dd["High"].iloc[-1]}')
        print(f'Predicted Next Day Low: {predicted_next[0][1]} | Actual Last Day Low: {dd["Low"].iloc[-1]}')
        linearregression.append([f"{ticker:20}", f"{predicted_next[0][0]:.2f}", dd['High'].iloc[-1], f"{predicted_next[0][1]:.2f}", dd['Low'].iloc[-1]])
    
        
        
        return weights, y_mean, y_std 

def predict_today(ticker=nifty50_tickers_ns, learning_rate=0.01, epochs=50000, start_date="2024-01-01", download_bool=0, end_date="2024-12-31", interval=["1d", "1mo"], filename=["time_series.csv", "time_series_monthly.csv"]):
    # Get the model weights and normalization parameters
    linearregression = []
    for j in range(len(ticker)):
        dd, dm = download_stock_data(download_bool, ticker[j], start_date, end_date, interval, filename)

        weights, y_mean, y_std = predict(ticker[j], learning_rate, epochs, start_date, download_bool, end_date, interval, filename)
        yesterday = len(dd)-2
        # Prepare the input for prediction
        last_day_data = np.array([[dd['Open'][yesterday],  dd['High'][yesterday], dd['Low'][yesterday], dd['Close'][yesterday]]])
        
        # Normalize the input
        last_day_mean = np.array([dd['Open'][yesterday], dd['High'][yesterday], dd['Low'][yesterday], dd['Close'][yesterday]]).mean()
        last_day_std = np.array([dd['Open'][yesterday], dd['High'][yesterday], dd['Low'][yesterday], dd['Close'][yesterday]]).std(ddof=0)
   
        last_day_normalized = (last_day_data - last_day_mean) / last_day_std if last_day_std != 0 else last_day_data

        # Predict using the provided weights
        predicted_next_normalized = last_day_normalized.dot(weights)

        # Reverse normalization
        predicted_next = predicted_next_normalized * y_std + y_mean
        linearregression.append([ticker[j],predicted_next[0][0], dd['High'][len(dd)-1] ,predicted_next[0][1],dd['Low'][len(dd)-1]])

    linearregression_df = pd.DataFrame(linearregression, columns=["ticker", "predicted_next_day_high", "actual_next_day_high", "predicted_next_day_low", "actual_next_day_low"])
    linearregression_df.to_csv("./database/linear_regression.csv", index=False)
    print(f"CSV of linear regression './database/linear_regression.csv' saved successfully.")
