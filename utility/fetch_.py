import yfinance as yf
import pandas as pd
import os

nifty50_tickers_ns = [
    "ADANIENT.ns",  # Adani Enterprises Ltd.
    "ADANIPORTS.ns",  # Adani Ports and SEZ Ltd.
    "APOLLOHOSP.ns",  # Apollo Hospitals Enterprise Ltd.
    "ASIANPAINT.ns",  # Asian Paints Ltd.
    "AXISBANK.ns",  # Axis Bank Ltd.
    "BAJFINANCE.ns",  # Bajaj Finance Ltd.
    "BAJAJFINSV.ns",  # Bajaj Finserv Ltd.
    "BHARTIARTL.ns",  # Bharti Airtel Ltd.
    "BIOCON.ns",  # Biocon Ltd.
    "BPCL.ns",  # Bharat Petroleum Corporation Ltd.
    "BRITANNIA.ns",  # Britannia Industries Ltd.
    "CIPLA.ns",  # Cipla Ltd.
    "COALINDIA.ns",  # Coal India Ltd.
    "DIVISLAB.ns",  # Divi's Laboratories Ltd.
    "DLF.ns",  # DLF Ltd.
    "DRREDDY.ns",  # Dr. Reddy's Laboratories Ltd.
    "EICHERMOT.ns",  # Eicher Motors Ltd.
    "GRASIM.ns",  # Grasim Industries Ltd.
    "HCLTECH.ns",  # HCL Technologies Ltd.
    # "HDFC.ns"
    "HDFCBANK.ns",  # HDFC Bank Ltd.
    "HDFCLIFE.ns",  # HDFC Life Insurance Company Ltd.
    "HEROMOTOCO.ns",  # Hero MotoCorp Ltd.
    "HINDALCO.ns",  # Hindalco Industries Ltd.
    "HINDUNILVR.ns",  # Hindustan Unilever Ltd.
    "ICICIBANK.ns",  # ICICI Bank Ltd.
    "ITC.ns",  # ITC Ltd.
    "JSWSTEEL.ns",  # JSW Steel Ltd.
    "KOTAKBANK.ns",  # Kotak Mahindra Bank Ltd.
    "LT.ns",  # Larsen & Toubro Ltd.
    "M&M.ns",  # Mahindra & Mahindra Ltd.
    "MARUTI.ns",  # Maruti Suzuki India Ltd.
    "NTPC.ns",  # NTPC Ltd.
    "ONGC.ns",  # Oil & Natural Gas Corporation Ltd.
    "POWERGRID.ns",  # Power Grid Corporation of India Ltd.
    "RELIANCE.ns",  # Reliance Industries Ltd.
    "SBILIFE.ns",  # SBI Life Insurance Company Ltd.
    "SBIN.ns",  # State Bank of India
    "SHREECEM.ns",  # Shree Cement Ltd.
    "SIEMENS.ns",  # Siemens Ltd.
    "SUNPHARMA.ns",  # Sun Pharmaceutical Industries Ltd.
    "TATAMOTORS.ns",  # Tata Motors Ltd.
    "TATAPOWER.ns",  # Tata Power Company Ltd.
    "TATASTEEL.ns",  # Tata Steel Ltd.
    "TECHM.ns",  # Tech Mahindra Ltd.
    "TITAN.ns",  # Titan Company Ltd.
    "ULTRACEMCO.ns",  # UltraTech Cement Ltd.
    "UPL.ns",  # UPL Ltd.
    "WIPRO.ns",  # Wipro Ltd.
    "YESBANK.ns",  # Yes Bank Ltd.
    "ZOMATO.ns"  # Zomato Ltd.
]


def read_and_preprocess_csv(filepath):
    df = pd.read_csv(filepath)
    # df.rename(index={0: 'Date', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close',5:'Adj Close', 6: 'Volume'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', ascending=True, inplace=True)
    return df

def download_stock_data(download_bool=0, ticker="tatasteel.ns", start_date="2024-01-01", end_date="2024-12-31", interval=["1d", "1mo"], filename=["time_series.csv", "time_series_monthly.csv"]):
    folder_path = f"./database/{ticker}/"
    
    if download_bool == 0:
        try:
            dd = read_and_preprocess_csv(folder_path + filename[0])
            dm = read_and_preprocess_csv(folder_path + filename[1])
            return dd, dm
        except FileNotFoundError:
            return download_stock_data(1, ticker, start_date, end_date, interval, filename)
    
    data = {}
    os.makedirs(folder_path, exist_ok=True)

    for i, freq in enumerate(interval):
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=freq )
        if stock_data.empty:
            return exit()
        
        stock_data.to_csv(folder_path + filename[i])
        print(f"Data saved to {filename[i]}")
        
        if freq not in ["1d", "1mo"]:
            data[freq] = stock_data
    
    if interval == ["1d", "1mo"]:
        dd = read_and_preprocess_csv(folder_path + filename[0])
        dm = read_and_preprocess_csv(folder_path + filename[1])
        return dd, dm
    
    return data


def range_between(start, end, checkstart, checkend):
    if start > end or checkstart > checkend:
        return None
    
    return [max(start, checkstart), min(end, checkend)] if checkstart <= end and checkend >= start else None
