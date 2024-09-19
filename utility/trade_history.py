import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utility.fetch_ import download_stock_data

def trade_history(ticker=["tatasteel.ns"],show_bool=0,download_bool=0, start_date="2024-01-01", end_date="2024-12-31", interval=["1d","1mo"] , filename=["time_series.csv","time_series_monthly.csv"]):
    for j in range(len(ticker)):
        dd , dm = download_stock_data(download_bool,ticker[j], start_date, end_date, interval, filename)
        buy_price = []
        trade_history = []
        for i in range(1, len(dm)):
            prev_month = float(dm.iloc[i-1]['Close'])  # old - low
            threshold = prev_month * 1.1

            curr_year, curr_month = pd.to_datetime(dm.iloc[i]['Date']).year, pd.to_datetime(dm.iloc[i]['Date']).month
            next_month_data = dd[(dd['Date'].dt.year == curr_year) & (dd['Date'].dt.month == curr_month)]
        
            for _, row in next_month_data.iterrows():
                if row['Low'] <= prev_month: # old - threshold
                    buy_price.append([row['Date'], row['Low']])
                    # print(f"Buy signal on {row['Date']} with a low of {row['Low']}")
                    pass
                if len(buy_price) > 0:
                    for buy in buy_price:
                        if row['High'] >= buy[1] * 1.1:
                            buy_date = buy[0]
                            buy_val = buy[1]
                            sell_date = row['Date']
                            sell_val = row['High']
                            months_diff = (sell_date.year - buy_date.year) * 12 + (sell_date.month - buy_date.month)
                            trade_history.append([buy_date, sell_date, f"{buy_val:.2f}", f"{sell_val:.2f}", f"{months_diff:.2f}"])
                            # print(f"Sell signal on {row['Date']} with a high of {row['High']}. Holding period: {months_diff} months.")
                            buy_price.remove(buy)
                            break

        trade_history_df = pd.DataFrame(trade_history, columns=['Buy Date', 'Sell Date', 'Buy Price', 'Sell Price', 'Holding Period (Months)'])

        # Insert empty rows where the month ends - not working
        # for i in range(1,len(trade_history_df)):
        #     curr_month = pd.to_datetime(trade_history_df.iloc[i]['Buy Date']).month
        #     prev_month = pd.to_datetime(trade_history_df.iloc[i - 1]['Buy Date']).month
        #     if curr_month != prev_month:
        #         empty_row = pd.Series([np.nan] * trade_history_df.shape[1], index=trade_history_df.columns)
        #         trade_history_df = pd.concat([trade_history_df.iloc[:i+1], empty_row.to_frame().T, trade_history_df.iloc[i+1:]], ignore_index=True)

        trade_history_df.to_csv("./database/"+ticker[j]+'/trade_history.csv',index=False)
        print(f"Exported trade history to ./database/{ticker[j]}/trade_history.csv")

        if not trade_history_df.empty and show_bool:
            plt.figure(figsize=(10, 6))
            
            plt.plot(dd['Date'], dd['Close'], label="Daily Close Price", color="gray", alpha=0.5)
            plt.scatter(pd.to_datetime(trade_history_df['Buy Date']), trade_history_df['Buy Price'], label='Buy', marker='^', color='green', s=100)
            plt.scatter(pd.to_datetime(trade_history_df['Sell Date']), trade_history_df['Sell Price'], label='Sell', marker='v', color='red', s=100)
            
            plt.title('Buy and Sell Signals with Holding Period')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()