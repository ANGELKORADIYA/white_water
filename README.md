# Whiter Water - stock market Prediction & Analysis

# Description

analysis of stock market from previous market data.This analysis helps traders and investors identify favorable buying opportunities by analyzing historical stock data.

not use any ai algo but use some basic math to predict the stock market

# Installation

### python3
### pip install pandas numpy matplotlib yfinance mplfinance

### for use models : pip install pandas numpy matplotlib yfinance mplfinance scikit-learn tensorflow keras


# How to run
### python nifty50.py
### usemodel.ipynb  

# Documentation
- nifty50.py
    - trade_history(ticker=nifty50_tickers_ns)
        - trade_history.csv
        
    - trade_history_monthly(dataset="trade_history",ticker=nifty50_tickers_ns)
        - mon_trade_history.csv

    - trade_history_monthly(ticker=nifty50_tickers_ns) 
        - trade_history_2.csv
        - mon_trade_history_2.csv

    @ trade_history generate a reportss in ./database/{company_name} corresponding folder
        

    - alert_from_average(ticker=nifty50_tickers_ns)
        - generates a reportss in ./database/average_buy.csv
        - generates a reportss in ./database/buy_signal_alerts.csv

- usemodel.ipynb
    - clf.py
        - classification_report , RandomForestClassifier
        - generates a reportss in ./database/trade_reports folder

    - mymodel.py
        - LSTM
        - generates a reportss in ./database/mymodelprediction.csv


- fetch_.download_stock_data - download and preprocess csv
- checkingswingtrading.trade_history_2 - trade history with buy and sell signals
- checkingswingtrading.trade_history_monthly - trade history with buy and sell signals
- average_model.alert_from_average - alert from average