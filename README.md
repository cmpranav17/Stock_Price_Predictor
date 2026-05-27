Stock Price Predictor

I always wondered if you could teach a machine to read the market. Not perfectly,
but well enough to spot a trend. That question turned into this project.

You type a company name — Apple, Nvidia, Tesla, anything — and it fetches two
years of real market data, trains a model on it, and tells you where the price
might go tomorrow.


What it does

Type a company name or ticker symbol into the search bar. The app resolves it
automatically, so you never need to know that Google is GOOGL or that Meta
used to be Facebook.

It then pulls historical price data, engineers features like moving averages
and volatility, trains a Ridge Regression model, and gives you a next-day
price prediction with a full signal breakdown.


What you see on screen

A live price, a predicted next-day price, a trend direction, an accuracy score,
two charts showing actual vs predicted prices and moving average bands, and a
signal panel showing whether the stock is bullish or bearish based on the model.


The bug that taught me the most

My first model was using Linear Regression with no feature scaling. It looked
accurate because it was essentially memorising recent prices rather than learning
patterns. After some research I switched to Ridge Regression with StandardScaler,
which penalises over-complexity and generalised far better on unseen data.

The model now hits R² of 0.922 on test data it has never seen. That means it
explains 92% of price movement correctly, with an average error of about $3.37
on a $300 stock.


Built with

Python, yfinance, Scikit-learn, Streamlit, Matplotlib, Pandas, NumPy


How to run it yourself

git clone https://github.com/cmpranav17/Stock_Price_Predictor.git
cd Stock_Price_Predictor
pip install -r requirements.txt
streamlit run app.py


What I learned

Time-series data is different from regular tabular data. You cannot shuffle it
before splitting or you leak future information into the training set. Feature
engineering matters more than model choice. And overfitting is sneaky — a model
can look perfect on trainin
