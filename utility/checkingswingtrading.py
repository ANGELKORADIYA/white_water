import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from utility.fetch_ import download_stock_data
def trade_history_2(show_bool=0,download_bool=0,ticker=["tatasteel.ns"], start_date="2024-01-01", end_date="2024-12-31", interval=["1d","1mo"] , filename=["time_series.csv","time_series_monthly.csv"]):
    for i in range(len(ticker)):
        dd,dm = download_stock_data(download_bool,ticker[i], start_date, end_date, interval, filename)

        trade_history = []

        # Iterate through monthly data (dm) to check the previous month's close for buying
        for i in range(1, len(dm)):
            prev_month_close = float(dm.iloc[i-1]['Close'])  # Previous month's close value
            
            # Extract the year and month for the current month
            curr_year, curr_month = pd.to_datetime(dm.iloc[i]['Date']).year, pd.to_datetime(dm.iloc[i]['Date']).month
            
            # Filter daily data for the current month in dd
            curr_month_data = dd[(dd['Date'].dt.year == curr_year) & (dd['Date'].dt.month == curr_month)]
            
            # Check for buy signals in the daily data for the current month
            buy_made = False
            for _, row in curr_month_data.iterrows():
                # If buy price is found in daily data
                if row['Low'] <= prev_month_close:
                    buy_date = row['Date']
                    buy_price = prev_month_close
                    target_sell_price = buy_price * 1.1  # Target sell price (10% higher)
                    buy_made = True
                    # print(f"Buy signal on {buy_date} at price: {buy_price}")
                    
                    # After buying, look for the sell signal in the same month
                    curr_month_data= dd[dd['Date']>pd.to_datetime(f"{curr_year}-{curr_month}-01")]
                    for _, sell_row in curr_month_data.iterrows():
                        if sell_row['High'] >= target_sell_price:
                            sell_date = sell_row['Date']
                            sell_price = target_sell_price
                            
                            # Calculate holding period in months
                            months_diff = (sell_date.year - buy_date.year) * 12 + (sell_date.month - buy_date.month)
                            
                            # Append the trade to history
                            trade_history.append([buy_date, sell_date, f"{buy_price:.2f}", f"{sell_price:.2f}", f"{months_diff:.2f}"])
                            # print(f"Sell signal on {sell_date} at price: {sell_price}. Holding period: {months_diff} months.")
                            
                            # Exit after selling
                            break

        # Convert trade history into DataFrame
        trade_history_df = pd.DataFrame(trade_history, columns=['Buy Date', 'Sell Date', 'Buy Price', 'Sell Price', 'Holding Period (Months)'])

        # Plot Buy and Sell Prices
        if not trade_history_df.empty and show_bool:
            plt.figure(figsize=(10, 6))
            
            # Plot all daily close prices
            plt.plot(dd['Date'], dd['Close'], label="Daily Close Price", color="gray", alpha=0.5)
            
            # Highlight buy and sell points
            plt.scatter(pd.to_datetime(trade_history_df['Buy Date']), trade_history_df['Buy Price'], label='Buy', marker='^', color='green', s=100)
            plt.scatter(pd.to_datetime(trade_history_df['Sell Date']), trade_history_df['Sell Price'], label='Sell', marker='v', color='red', s=100)
            
            plt.title('Buy and Sell Signals with Holding Period')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()

        # Save trade history to CSV
        trade_history_df.to_csv("./database/"+ticker[i]+'/trade_history_2.csv', index=False)
        print(f"Exported trade history to ./database/{ticker[i]}/trade_history_2.csv")




def trade_history_monthly(dataset = "trade_history_2",show_bool=0,download_bool=0,ticker=["tatasteel.ns"], start_date="2024-01-01", end_date="2024-12-31", interval=["1d","1mo"] , filename=["time_series.csv","time_series_monthly.csv"]):
    for i in range(len(ticker)):
        trade_history_df = pd.DataFrame()
        try:
            trade_history_df=pd.read_csv(f'./database/{ticker[i]}/{dataset}.csv')
        except:
            print(f"no file found thats why im creating ./database/{ticker[i]}/trade_history_2.csv")
            trade_history_2(show_bool,download_bool,ticker[i], start_date, end_date, interval, filename)
            trade_history_df = pd.read_csv(f'./database/{ticker[i]}/trade_history_2.csv')
            dataset = "trade_history_2"
        trade_history_df['Buy Month'] = pd.to_datetime(trade_history_df['Buy Date']).dt.to_period('M')
        # Group by the Buy Month and calculate the total number of trades and profit for each month
        monthly_trades = trade_history_df.groupby('Buy Month').agg(
            Total_Trades=('Buy Date', 'count'), 
            Average_Month=('Holding Period (Months)', lambda x:(x.sum()/len(x))),
            Total_Profit=('Sell Price', lambda x: (x.sum() - trade_history_df.loc[x.index, 'Buy Price'].sum()))
        ).reset_index()

        # Save monthly trade statistics to CSV for further analysis
        monthly_trades.to_csv("./database/"+ticker[i]+'/mon_'+dataset+'.csv', index=False)
        print(f"Exported trade history to ./database/{ticker[i]}/mon_"+dataset+".csv")
        # Optionally, plot the number of trades and profit per month
        if show_bool:
            plt.figure(figsize=(10, 6))

            # Plot the number of trades per month
            plt.subplot(2, 1, 1)
            plt.bar(monthly_trades['Buy Month'].astype(str), monthly_trades['Total_Trades'], color='blue')
            plt.title('Number of Trades per Month')
            plt.ylabel('Total Trades')
            plt.xticks(rotation=45)

            # Plot the total profit per month
            plt.subplot(2, 1, 2)
            plt.bar(monthly_trades['Buy Month'].astype(str), monthly_trades['Total_Profit'], color='green')
            plt.title('Total Profit per Month')
            plt.ylabel('Total Profit')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()













# dalmon= np.array(dm)
# daldal= np.array(dd)
# def range_between(start, end, checkstart, checkend):
#     """
# range_between(start, end, checkstart, checkend) -> same in both (must be start < end)  <br>
# plot_candle(data,start,end) -> plots candle infex start to end for data<br>
# print_info(previous , current) -> prints info <br>
# profit_2d(final ,giverange) -> returns profit (giverange is [start,end]) using average of giverange <br>
# directive_analysis(direction) -> returns directions count like upup , downdown ,... <br> 
# pl5topl15_info() -> prints - previous low * 1.05 - buy ; previous high * 1.15 - sell <br>
# pl5topl15(daily from reverse in table = daily index , monthly index - 2 = last - 1 = last second) example : 11 ,3 - pl5topl15(len(daldal)-11,1)
# """
#     if start > end:
#         return None
#     if checkstart > checkend:
#         return None
    
#     if start <= checkstart <= end and start <= checkend <= end:
#         return [checkstart, checkend]
#     elif start <= checkstart <= end and checkend > end:
#         return [checkstart, end]
#     elif checkstart <start and checkend >= end:
#         return [start, end]
#     elif checkstart < start and start< checkend < end:
#         return [start, checkend]
#     else:
#         return None

# def findMonthandDailyIndex(month,bool_show=0):
#     diff=0
#     for i in range(len(dalmon)-1,0,-1):
#         if dalmon[i][0] <=month:
#             diff = i
#             break
#     curr = 0 
#     next = len(daldal) 
#     prev = dalmon[diff] 
#     for i in range(len(daldal)):
#         if daldal[i][0]>prev[0]:
#             curr=i
#             break
#     for i in range(len(daldal)):
#         if daldal[i][0]>dalmon[diff+1][0]:
#             next=i
#             break
#     if bool_show !=0:
#         bool_show(diff,curr,next)
#     return [diff,curr ,next]

# def plot_candle(data,start,end):
#     df = pd.DataFrame(data[start:end])
#     df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
#     df["Date"] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M')
#     df.set_index('Date', inplace=True)
#     style = "yahoo"
#     fig, ax = plt.subplots(figsize=(10, 5))
#     mpf.plot(df, type='candle', style=style, 
#             volume=False, ax=ax, ylabel='Price',
#             xrotation=45, datetime_format='%Y-%m-%d')
#     ax.set_facecolor('#101010')
#     fig.patch.set_facecolor('xkcd:black')
#     ax.grid(True, color='black', linestyle='--', linewidth=0.5)
#     ax.tick_params(axis='x', colors='white')  # For x-axis labels
#     ax.tick_params(axis='y', colors='white')  # For y-axis labels
#     plt.tight_layout()
#     plt.show()
    
# def direction_analysis(choose):
#     count =0
#     for i in range(1,len(dalmon)):
#         dalx = dalmon[i-1][4]
#         daly = dalmon[i-1][1] 
#         dalnx = dalmon[i][4] 
#         dalny= dalmon[i][1]  
#         match choose:
#             case 1:
#                 if dalx <daly and dalnx < dalny:
#                     count+=1
#             case 2:
#                 if dalx >daly and dalnx > dalny:
#                     count+=1
#             case 3:
#                 if dalx <daly and dalnx > dalny:
#                     count+=1
#             case 4:
#                 if dalx >daly and dalnx < dalny:
#                     count+=1
#     print(count,len(dalmon) , (count/len(dalmon))*100)

# def profit_2d(final ,giverange):
#     if type(giverange) != type([0,1]):
#         return None
#     avgrange = (giverange[0] + giverange[1])/2
#     return ((final - avgrange)/avgrange)*100

# def profit_(final ,giverange):
#     if giverange == 0:
#         return None
#     return ((final - giverange)/giverange)*100

# def print_info(previous , current):
#     print("date:",previous[0],"open:", previous[1] , "high:", previous[2] , "low:", previous[3] , "close:", previous[4])
#     print("date:",current[0],"open:", current[1] , "high:", current[2] , "low:", current[3] , "close:", current[4])


# # previous low * 1.05 - buy ; previous high * 1.15 - sell
# def pl5topl15_info():
#     count =0
#     for i in range(1, len(dalmon)):
#         same_range = range_between(dalmon[i][3],dalmon[i][2],dalmon[i-1][3]*1.05,dalmon[i-1][3]*1.15)
#         if same_range != None:
#             profit = ((same_range[1]-same_range[0])/same_range[0])*100
#             if profit >= 5:
#                 if dalmon[i][0]>"2022-02-21":
#                     print(dalmon[i][0],profit,same_range)
#                 count += 1
#     print(count,len(dalmon) , (count/len(dalmon))*100) 

# def pl5topl15(diff,curr,next):
#     i= 0
#     count =0
#     prev = dalmon[diff]
#     buyval =0
#     print(daldal[curr],prev)
#     for i in range(curr,next):
#         same_range =  range_between(prev[3],prev[3]*1.05,daldal[i][3],daldal[i][2])
#         if same_range != None:
#             print("buy",daldal[i][0],same_range)
#             buyval= same_range[0]
#             salval = same_range[0]*1.1
#             for j in range(i+1,next):
#                 same_range =  range_between(salval,salval,daldal[j][3],daldal[j][2]) 
#                 if same_range != None:
#                     print("sell",daldal[i][0],same_range , profit_(salval,buyval))
#                     count+=1
#                     break #ok
#     print(count,i)
    
# # previous low - buy ; previous high or  9-10% - sell ->
# def plph_info(dalnp):
#     count =0
#     for i in range(1,len(dalnp)):
#         dalx = dalnp[i-1][3]
#         daly = dalnp[i-1][2] 
#         dalnx = dalnp[i][3]
#         dalny= dalnp[i][2]
#         same_range = range_between(dalx,daly,dalnx,dalny)
#         if same_range != None:
#             profit = ((same_range[1]-same_range[0])/same_range[0])*100
#             if profit >= 5:
#                 if dalnp[i][0]>"2022-02-21":
#                     print(dalnp[i][0],int(profit),"%",profit_(dalx,same_range[0]) ,profit_2d(same_range[0],[dalnx,dalny]))
#                 count += 1
#     print(count,len(dalnp) , (count/len(dalnp))*100)
    
# def plph(diff,curr,next):
#     count =0
#     prev = dalmon[diff]
#     buyval =0
#     print(daldal[curr],prev)
#     for i in range(curr,next):
#         same_range =  range_between(prev[3],prev[3]*1.05,daldal[i][3],daldal[i][2])  
#         if same_range != None:
#                 print("buy",daldal[i][0],same_range)
#                 buyval = same_range[0]
#                 salval = same_range[0]*1.06
#                 for j in range(i+1,next):
#                     same_range =  range_between(salval,salval,daldal[j][3],daldal[j][2])
#                     if same_range != None:
#                         print("sell",daldal[j][0],same_range,profit_(salval,buyval))
#                         count+=1
#                         break #ok
#     print(count,i)
    
# def clch(diff,curr,next):
#     prev=dalmon[diff]
#     buycount=0
#     count=0
#     print("low :")
#     buyval = prev[3]
#     print(daldal[curr],prev,buyval)
#     for i in range(curr,next):
#         same_range = range_between(buyval, buyval, daldal[i][3],daldal[i][2])
#         if same_range != None:
#             print("buy",daldal[i][0],same_range)
#             buycount+=1
#         same_range = range_between(daldal[i][3], daldal[i][2], buyval*1.1,buyval*1.1)
#         if same_range != None:
#             print("sell",daldal[i][0],same_range)
#             count+=1
#     print(buycount,count)
#     print("open :")
#     buyval = prev[1]
#     print(daldal[curr],prev,buyval)
#     for i in range(curr,next):
#         same_range = range_between(buyval, buyval, daldal[i][3],daldal[i][2])
#         if same_range != None:
#             print("buy",daldal[i][0],same_range)
#             buycount+=1
#         same_range = range_between(daldal[i][3], daldal[i][2], buyval*1.1,buyval*1.1)
#         if same_range != None:
#             print("sell",daldal[i][0],same_range)
#             count+=1
#     print(buycount,count)