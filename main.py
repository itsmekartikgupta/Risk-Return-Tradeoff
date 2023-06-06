import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Define the NASDAQ-100 index and its components
portfolio_ticker = "^NDX"
stocks = ["META", "AMZN", "AAPL", "NFLX", "GOOGL", "BABA", "NVDA", "TSLA"]

# Define the date range for the historical data
start_date = "2018-01-01"
end_date = "2022-01-01"

# Retrieve historical stock price data for the NASDAQ-100 index and the individual stocks
data = yf.download([portfolio_ticker] + stocks, start=start_date, end=end_date, period="1d")["Adj Close"]

# Extract the NASDAQ-100 index data and individual stock data
portfolio_data = data[portfolio_ticker]
individual_stocks_data = data[stocks]

# Calculate the daily returns of the NASDAQ-100 index and the individual stocks
portfolio_returns = portfolio_data.pct_change().dropna()
individual_returns = individual_stocks_data.pct_change().dropna()

# Calculate the volatility (standard deviation) of the NASDAQ-100 index and the individual stocks
volatility_portfolio = portfolio_returns.std() * np.sqrt(252)  # Multiply by sqrt(252) for annual volatility
volatility_individual = individual_returns.std() * np.sqrt(252)

# Calculate the average annual returns
annual_returns_portfolio = portfolio_returns.mean() * 252  # 252 trading days in a year
annual_returns_individual = individual_returns.mean() * 252

# Perform linear regression between returns and risk
slope, intercept, r_value, p_value, std_err = linregress(volatility_individual, annual_returns_individual)

# Create a scatter plot to compare the returns of the individual stocks with the volatility
plt.scatter(volatility_individual, annual_returns_individual, label="Individual Stocks")
plt.scatter(volatility_portfolio, annual_returns_portfolio, color="red", label="Portfolio (NASDAQ-100)")
plt.xlabel("Volatility")
plt.ylabel("Annual Returns")
plt.title("Annual Returns vs. Volatility: Individual Stocks vs. Portfolio (NASDAQ-100)")

# Add the regression line
x = np.linspace(min(volatility_individual), max(volatility_individual), 100)
y = intercept + slope * x
plt.plot(x, y, color="black", linestyle="--", label="Regression Line")

# Add annotations for the stock names
for i, stock in enumerate(stocks):
    plt.annotate(stock, (volatility_individual[i] , annual_returns_individual[i]),textcoords="offset points", xytext=(5, 5), ha='right', size = 8)

plt.legend()
plt.show()
