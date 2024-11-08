import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utility.fetch_ import download_stock_data , nifty50_tickers_ns
import os
def calculate_indicators(stock_data):
    # Calculate simple moving averages
    stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()

    # Calculate RSI
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    return stock_data

def stock_prediction(ticker=nifty50_tickers_ns, start_date="2024-01-01", end_date="2024-12-31", download_bool=0, interval=["1d", "1mo"], filename=["time_series.csv", "time_series_monthly.csv"]):
    for j in range(len(ticker)):
        # Download stock data
        dd, dm = download_stock_data(download_bool,ticker[j], start_date, end_date, interval, filename)
        stock_data = dd

        # Feature Engineering: Calculate technical indicators
        stock_data = calculate_indicators(stock_data)
        
        # Lag features (previous day's close and open)
        stock_data['Prev_Close'] = stock_data['Close'].shift(1)
        stock_data['Prev_Open'] = stock_data['Open'].shift(1)

        # Drop NaN values
        stock_data.dropna(inplace=True)
        
        # Define features and labels for regression
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50', 'RSI', 'Prev_Close', 'Prev_Open']
        X = stock_data[features]
        y = stock_data['Close']  # Predicting the actual closing price
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Regressor to predict stock closing prices
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict future closing prices
        stock_data['Predicted_Close'] = model.predict(X)

        # Define conditions for Buy/Sell logic based on predicted price
        stock_data['Signal'] = None
        stock_data['Target_Price'] = None

        for i in range(1, len(stock_data)):
            if stock_data['Predicted_Close'].iloc[i] > stock_data['Close'].iloc[i]:  # Buy signal
                stock_data.at[stock_data.index[i], 'Signal'] = 'Buy'
                stock_data.at[stock_data.index[i], 'Target_Price'] = stock_data['Close'].iloc[i] * 1.10  # 10% profit target
            elif stock_data['Predicted_Close'].iloc[i] < stock_data['Close'].iloc[i]:  # Sell signal
                stock_data.at[stock_data.index[i], 'Signal'] = 'Sell'
                stock_data.at[stock_data.index[i], 'Target_Price'] = stock_data['Close'].iloc[i] * 0.90  # 10% stop-loss during downturn

        # Save the generated signals and data to a CSV file
        if not os.path.exists(f"./trade_reports"):
            os.makedirs(f"./trade_reports")
        stock_data.to_csv(f"./trade_reports/{ticker[j]}.csv", index=False)
    print("Storying At ./trade_reports")
    return stock_data

# Example usage
# result_data = stock_prediction(ticker=["TATAMOTORS.NS"], start_date="2024-01-01", end_date="2024-12-31")
