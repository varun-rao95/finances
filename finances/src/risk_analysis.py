from matplotlib import ticker
import numpy as np
import pandas as pd
import yfinance as yf
import sqlite3
import os
import statsmodels.api as sm

DB_NAME = "mint_transactions.db"
EXCLUDE_TICKERS = ["FIT", "FXAIX"]


def get_price_for_ticker(
    ticker,
    start_date="2020-07-17",
    end_date="2015-01-20",
    append_to_csv=False,
    overwrite_csv=False,
):
    start_date = "2020-07-17"
    end_date = "2025-01-20"
    data = yf.download(ticker, start=start_date, end=end_date)
    data["Daily Return"] = data["Adj Close"].pct_change()
    data["Cumulative Return"] = (1 + data["Daily Return"]).cumprod()

    if append_to_csv:
        original_df = pd.read_csv(f"../data/{ticker}.csv", index_col=0)
        pd.concat([original_df, data], axis=0).to_csv(f"../data/{ticker}.csv")
    if overwrite_csv:
        data.to_csv(f"../data/{ticker}.csv")
    return data


def get_listed_tickers(portfolio):
    return np.delete(
        portfolio.ticker.unique(),
        np.where(np.isin(portfolio.ticker.unique(), EXCLUDE_TICKERS)),
    )


def compute_portfolio_shares(portfolio):
    all_dates = pd.date_range(portfolio.date.min(), "2025-01-20")  # to current day
    listed_tickers = get_listed_tickers(portfolio)

    portfolio["date"] = pd.to_datetime(portfolio["date"])
    portfolio["adjusted_shares"] = portfolio["num_shares"] * portfolio["buy"].map(
        {1: 1, 0: -1}
    )
    pivot = portfolio.pivot_table(
        index="date",
        columns="ticker",
        values="adjusted_shares",
        aggfunc="sum",
        fill_value=0,
    )
    cumulative_shares = (
        pivot.cumsum()[listed_tickers].reindex(all_dates, method="ffill").fillna(0)
    )
    return cumulative_shares


def load_and_clean_prices(ticker):
    # TODO: relative path
    price = pd.read_csv(f"../data/{ticker}.csv", index_col=0)["Adj Close"]

    cleaned_series = price[2:]
    cleaned_series.index = pd.to_datetime(cleaned_series.index)
    cleaned_series.name = price.iloc[0]

    return cleaned_series


def load_and_clean_returns(ticker):
    # TODO: relative path
    returns = pd.read_csv(f"../data/{ticker}.csv", index_col=0)["Daily Return"]

    cleaned_series = returns[2:]
    cleaned_series.index = pd.to_datetime(cleaned_series.index)
    cleaned_series.name = returns.iloc[0]

    return cleaned_series


def load_prices_for_all_tickers(portfolio):
    all_dates = pd.date_range(portfolio.date.min(), "2025-01-20")  # to current day
    listed_tickers = get_listed_tickers(portfolio)
    # TODO: get latest price for latest day
    prices = (
        pd.concat([load_and_clean_prices(ticker) for ticker in listed_tickers], axis=1)
        .reindex(all_dates, method="ffill")
        .fillna(0)
        .apply(pd.to_numeric)
    )
    return prices


def load_returns_for_all_tickers(portfolio):
    all_dates = pd.date_range(portfolio.date.min(), "2025-01-20")  # to current day
    listed_tickers = get_listed_tickers(portfolio)
    returns = (
        pd.concat([load_and_clean_returns(ticker) for ticker in listed_tickers], axis=1)
        .reindex(all_dates)
        .fillna(0)
        .apply(pd.to_numeric)
    )
    return returns


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, DB_NAME)
    conn = sqlite3.connect(db_path)

    portfolio = pd.read_sql_query("SELECT * FROM portfolio", conn)
    cumulative_shares = compute_portfolio_shares(portfolio)
    prices = load_prices_for_all_tickers(portfolio)

    portfolio_value = cumulative_shares * prices
    daily_weights = portfolio_value.div(portfolio_value.sum(axis=1), axis=0)

    daily_returns = load_returns_for_all_tickers(portfolio)
    daily_returns.columns = portfolio_value.columns

    portfolio_ret = (daily_weights * daily_returns).sum(axis=1)
    portfolio_ret.name = "Portfolio_Return"

    sp500 = load_and_clean_returns("^GSPC")
    # 4) Subtract a constant Rf from each day if you want, e.g. 5% annual => 0.05/252
    daily_rf = 0.05 / 252
    portfolio_excess = portfolio_ret - daily_rf
    sp500_excess = sp500 - daily_rf

    # 5) Regression
    df_reg = pd.DataFrame(
        {
            "portfolio_excess": portfolio_excess,
            "sp500_excess": sp500_excess,
        }
    ).dropna()

    X = sm.add_constant(df_reg["sp500_excess"])
    y = df_reg["portfolio_excess"]
    print("------------------- REGULARIZED -------------------")
    model = sm.OLS(y, X).fit_regularized()
    print(model.params)
    print("------------------- OLS -------------------")
    model = sm.OLS(y, X).fit()
    print(model.params)
    print(model.t_test([1, 0]))


# alpha ~ model.params["const"]
# beta ~ model.params["sp500_excess"]

if __name__ == "__main__":
    main()
