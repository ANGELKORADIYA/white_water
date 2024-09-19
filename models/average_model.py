import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utility.fetch_ import download_stock_data

def calculate_percentage_changes(data):
    """Calculate percentage changes from high and low prices."""
    data['High-High Change'] = 0.0
    data['Low-Low Change'] = 0.0
    data['High-Low Change'] = 0.0
    for i in range(1, len(data)):
        data['High-High Change'] = (data['High'] - data['High'].shift(1)) / data['High'].shift(1) * 100
        data['Low-Low Change'] = (data['Low'] - data['Low'].shift(1)) / data['Low'].shift(1) * 100
        data['High-Low Change'] = (data['High'] - data['Low']) / data['Low'] * 100
    #     data.loc[i, 'High-High Change'] = (data.loc[i, 'High'] - data.loc[i-1, 'High']) / data.loc[i-1, 'High'] * 100
    #     data.loc[i, 'Low-Low Change'] = (data.loc[i, 'Low'] - data.loc[i-1, 'Low']) / data.loc[i-1, 'Low'] * 100
    #     data.loc[i, 'High-Low Change'] = (data.loc[i, 'High'] - data.loc[i, 'Low']) / data.loc[i, 'Low'] * 100

    return data

def average_percentage_change(data, start_date):
    """filter and return average"""
    start_date = pd.to_datetime(start_date)
    filtered_data = data[data['Date'] >= start_date]
    filtered_data = calculate_percentage_changes(filtered_data)
    
    avg_high_high_change = filtered_data['High-High Change'].mean()
    avg_low_low_change = filtered_data['Low-Low Change'].mean()
    avg_high_low_change = filtered_data['High-Low Change'].mean()
    
    return filtered_data, avg_high_high_change, avg_low_low_change, avg_high_low_change

def plot_percentage_changes(data, start_date , show_bool=0):
    """Plot the distribution and trend of percentage changes."""
    filtered_data, avg_high_high_change, avg_low_low_change, avg_high_low_change = average_percentage_change(data, start_date)
    
    # Print the average percentage changes
    if show_bool:
        print(f"Average High-High Percentage Change from {start_date}: {avg_high_high_change:.2f}%")
        print(f"Average Low-Low Percentage Change from {start_date}: {avg_low_low_change:.2f}%")
        print(f"Average High-Low Percentage Change from {start_date}: {avg_high_low_change:.2f}%")
        
        # Show total frequency of percentage changes
        print(f"Total Frequency of High-High Changes: {filtered_data['High-High Change'].count()}")
        print(f"Total Frequency of Low-Low Changes: {filtered_data['Low-Low Change'].count()}")
        print(f"Total Frequency of High-Low Changes: {filtered_data['High-Low Change'].count()}")
        
        # Plot the trend of percentage changes over time
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_data['Date'], filtered_data['High-High Change'], marker='o', linestyle='-', markersize=2, label='High-High Change')
        plt.plot(filtered_data['Date'], filtered_data['Low-Low Change'], marker='x', linestyle='--', markersize=2, label='Low-Low Change')
        plt.plot(filtered_data['Date'], filtered_data['High-Low Change'], marker='^', linestyle='-.', markersize=2, label='High-Low Change')
        plt.title('Percentage Change Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Percentage Change')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot the distribution of percentage changes
        plt.figure(figsize=(12, 6))
        sns.histplot(filtered_data['High-High Change'], bins=30, kde=True, color='blue', label='High-High Change')
        sns.histplot(filtered_data['Low-Low Change'], bins=30, kde=True, color='orange', label='Low-Low Change')
        sns.histplot(filtered_data['High-Low Change'], bins=30, kde=True, color='green', label='High-Low Change')
        plt.title('Distribution of Percentage Changes')
        plt.xlabel('Percentage Change')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()


def calculate_buying_price(data_dd, data_dm, start_date):
    """Calculate the expected buying price based on historical data and percentage changes."""
    # Get average percentage changes from both datasets
    _, avg_high_high_dd, avg_low_low_dd, avg_high_low_dd = average_percentage_change(data_dd, start_date)
    _, avg_high_high_dm, avg_low_low_dm, avg_high_low_dm = average_percentage_change(data_dm, start_date)
    
    # Calculate average percentage change across datasets
    avg_high_high = (avg_high_high_dd + avg_high_high_dm) / 2
    avg_low_low = (avg_low_low_dd + avg_low_low_dm) / 2
    avg_high_low = (avg_high_low_dd + avg_high_low_dm) / 2
    
    # Assume current month's average price
    avg_close_dd = data_dd[data_dd['Date'].dt.to_period('M') == pd.to_datetime('today').strftime('%Y-%m')]['Close'].mean()
    avg_close_dm = data_dm[data_dm['Date'].dt.to_period('M') == pd.to_datetime('today').strftime('%Y-%m')]['Close'].mean()
    avg_close = (avg_close_dd + avg_close_dm) / 2
    
    # Calculate target buy price based on desired profit margin (e.g., 10%)
    target_profit_percentage = 10
    target_profit_multiplier = 1 + target_profit_percentage / 100
    expected_buy_price = avg_close / target_profit_multiplier
    
    return expected_buy_price, avg_high_high, avg_low_low, avg_high_low

def expected_buy_price(show_bool=0 ,start_date="2024-01-01", download_bool=0,ticker=["tatasteel.ns"],  end_date="2024-12-31", interval=["1d","1mo"] , filename=["time_series.csv","time_series_monthly.csv"]):
    trade_history = []
    for i in range(len(ticker)):
        dd , dm = download_stock_data(download_bool,ticker[i], start_date, end_date, interval, filename)
        plot_percentage_changes(dd, start_date, show_bool)
        plot_percentage_changes(dm, start_date,show_bool)

        expected_buy_price, avg_high_high, avg_low_low, avg_high_low = calculate_buying_price(dd, dm, start_date)

        # Filter data for the current month
        current_month = pd.to_datetime('today').strftime('%Y-%m')
        dd_current = dd[dd['Date'].dt.to_period('M') == current_month]
        dm_current = dm[dm['Date'].dt.to_period('M') == current_month]

        # Calculate average price for the current month
        avg_low_dd = dd_current['Low'].mean()
        avg_close_dd = dd_current['Close'].mean()
        avg_low_dm = dm_current['Low'].mean()
        avg_close_dm = dm_current['Close'].mean()

        # Define target profit
        target_profit_percentage = 10
        target_profit_multiplier = 1 + target_profit_percentage / 100

        # Calculate expected buy prices for 10% profit
        expected_buy_price_dd = avg_close_dd / target_profit_multiplier
        expected_buy_price_dm = avg_close_dm / target_profit_multiplier

        # Print the results
        trade_history.append([f"{ticker[i]:20}", f"{expected_buy_price:.2f}", f"{expected_buy_price_dd:.2f}", f"{expected_buy_price_dm:.2f}",f"{avg_high_high:.2f}%", f"{avg_low_low:.2f}%", f"{avg_high_low:.2f}%"])

        # Visualization
        if show_bool:
            plt.figure(figsize=(14, 7))

            # Plot DD
            plt.subplot(1, 2, 1)
            plt.hist(dd_current['Close'], bins=20, alpha=0.7, color='blue')
            plt.axvline(expected_buy_price_dd, color='red', linestyle='--', label='Target Buy Price')
            plt.title('DD Current Month Close Prices')
            plt.xlabel('Close Price')
            plt.ylabel('Frequency')
            plt.legend()

            # Plot DM
            plt.subplot(1, 2, 2)
            plt.hist(dm_current['Close'], bins=20, alpha=0.7, color='green')
            plt.axvline(expected_buy_price_dm, color='red', linestyle='--', label='Target Buy Price')
            plt.title('DM Current Month Close Prices')
            plt.xlabel('Close Price')
            plt.ylabel('Frequency')
            plt.legend()

            plt.tight_layout()
            plt.show()
    trade_history_df= pd.DataFrame(trade_history,columns=["ticker", "expected_buy_price", "expected_buy_price_dd", "expected_buy_price_dm","avg_high_high", "avg_low_low", "avg_high_low"])
    trade_history_df.to_csv("./database/average_buy.csv", index=False)
    print(f"CSV of buy signal './database/average_buy.csv' saved successfully.")
    return trade_history_df

def alert_from_average(ticker=["tatasteel.ns"], show_bool=0, download_bool=0, start_date="2024-01-01", end_date="2024-12-31", interval=["1d", "1mo"], filename=["time_series.csv", "time_series_monthly.csv"], period="2d"):
    print("wait...")
    print("10 seconds for model to execute")
    data = []
    average_buy = expected_buy_price(show_bool, start_date, download_bool, ticker, end_date, interval, filename)
    for i in range(len(ticker)):
        dd, dm = download_stock_data(download_bool, ticker[i], start_date, end_date, interval, filename)
        
        today = pd.Timestamp('today').normalize()
        
        if period == "1w":  # For last week (7 days)
            last_period = today - pd.Timedelta(days=7)
        else:  # Default is today and yesterday (2 days)
            last_period = today - pd.Timedelta(days=2)

        # Filter the data for the given time range
        filtered_data = dd[dd['Date'].dt.normalize().between(last_period, today)]
        
        trigger_count = 0
        expected_buy = float(average_buy[average_buy['ticker'].str.contains(ticker[i])]['expected_buy_price'].iloc[0])
        
        for j in range(len(filtered_data)):
            low = filtered_data.iloc[j]['Low']
            high = filtered_data.iloc[j]['High']
            
            # Check if expected buy price is between Low and High
            if low <= expected_buy <= high:
                trigger_count += 1
        
        # Add the data for this ticker to the list
        data.append({
            'Ticker': ticker[i],
            'Expected_Buy_Price': expected_buy,
            'Trigger_Count': trigger_count
        })
    
    # Convert the data to a DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv('./database/buy_signal_alerts.csv', index=False)
    print("CSV of buy signal './database/buy_signal_alerts.csv' saved successfully.")

def show_diff():
    dd,dm = download_stock_data()
    for i in range(2,9):
        start_date = pd.to_datetime(f"2024-0{i}-01")
        filtered_data_month = dm[dm['Date'] < start_date]
        filtered_data_day = dd[dd['Date'] < start_date]
        start_date = pd.to_datetime(f"2024-0{i-1}-01")
        filtered_data_month = filtered_data_month[filtered_data_month['Date'] >=start_date]
        filtered_data_day = filtered_data_day[filtered_data_day['Date'] >=start_date]
        print(start_date,filtered_data_day['High'].mean()-dm['High'][i])

