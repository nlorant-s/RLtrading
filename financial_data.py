import yfinance as yf
import alpaca_trade_api as alpaca

def get_data(timestep):
    # use API

    apple = alpaca.Ticker("AAPL") # get stock data on Apple Inc.
    # get open, high, low, close
    # Volume
    # Moving Average
    # RSI
    # P/E Ratio
    # Average True Range
    # D/E Ratio
    # Revenue Growth rate
    
    # return an array of this data

'''
Using Alpacas API to get stock data
https://docs.alpaca.markets/docs/about-market-data-api/

Using yfinance to get stock data
https://pypi.org/project/yfinance/
'''