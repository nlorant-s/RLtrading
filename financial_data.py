''' 
Using yfinance to get stock data
https://pypi.org/project/yfinance/
'''
import yfinance as yf

# get stock data on Apple Inc.
apple_stock = yf.Ticker("AAPL")
# get all historical data
apple_stock_data = apple_stock.history(period="max")
# get historical data for the last 5 days
apple_stock_data_last_5_days = apple_stock.history(period="5d")
print(apple_stock_data_last_5_days["Close"])

'''
Using Alpacas API to get stock data
https://docs.alpaca.markets/docs/about-market-data-api/
'''

