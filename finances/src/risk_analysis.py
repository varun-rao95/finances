import numpy as np
import pandas as pd
import yfinance as yf
from db import get_conn
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time

DB_NAME = "mint_transactions.db"
EXCLUDE_TICKERS = ["FIT"]


def safe_download(ticker, start, end, retries=5, delay=5):
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end, auto_adjust=False)
            if not data.empty:
                return data
            else:
                print(f"[{ticker}] Empty DataFrame. Retry {attempt + 1}/{retries}...")
        except Exception as e:
            print(f"[{ticker}] Attempt {attempt + 1} failed: {e}")
        time.sleep(delay)
    print(f"[{ticker}] Failed after {retries} retries.")
    return pd.DataFrame()


def get_price_for_ticker(
    ticker,
    start_date="2017-12-11",
    end_date="2025-01-24",
    append_to_csv=False,
    overwrite_csv=False,
):
    start_date = "2020-07-17"
    end_date = "2025-04-29"
    data = safe_download(ticker, start=start_date, end=end_date, delay=30)
    data["Daily Return"] = data["Adj Close"].pct_change()
    data["Cumulative Return"] = (1 + data["Daily Return"]).cumprod()

    if append_to_csv:
        original_df = pd.read_csv(f"../data/{ticker}.csv", index_col=0, header=[0, 1])
        original_df.index = pd.to_datetime(original_df.index)
        pd.concat([original_df, data], axis=1).sort_index().to_csv(
            f"../data/{ticker}.csv"
        )
    if overwrite_csv:
        data.to_csv(f"../data/{ticker}.csv")
    return data


def get_listed_tickers(portfolio):
    return np.delete(
        portfolio.ticker.unique(),
        np.where(np.isin(portfolio.ticker.unique(), EXCLUDE_TICKERS)),
    )


def compute_portfolio_shares(portfolio):
    all_dates = pd.date_range(portfolio.date.min(), "2025-04-29")  # to current day
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
    all_dates = pd.date_range(portfolio.date.min(), "2025-04-29")  # to current day
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
    all_dates = pd.date_range(portfolio.date.min(), "2025-04-29")  # to current day
    listed_tickers = get_listed_tickers(portfolio)
    returns = (
        pd.concat([load_and_clean_returns(ticker) for ticker in listed_tickers], axis=1)
        .reindex(all_dates)
        .fillna(0)
        .apply(pd.to_numeric)
    )
    return returns


def rolling_beta(portfolio, market, window=30):
    betas, indices = [], []
    for start in range(len(portfolio) - window + 1):
        end = start + window
        X = sm.add_constant(market[start:end])
        y = portfolio[start:end]
        model = sm.OLS(y, X).fit()
        betas.append(model.params["sp500_excess"])
        indices.append(portfolio.index[end - 1])
    return pd.Series(betas, index=indices)


def main():
    with get_conn() as conn:
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
    ).dropna()  # should only drop first day- not a big deal

    X = sm.add_constant(df_reg["sp500_excess"])
    y = df_reg["portfolio_excess"]
    print("------------------- OLS -------------------")
    print(f"===== Date Range: {portfolio.date.min()} to 2025-04-29 =====")
    model = sm.OLS(y, X).fit()
    print(model.params)
    print(model.t_test([1, 0]))
    print(model.t_test([0, 1]))

    rolling_betas = rolling_beta(
        df_reg["portfolio_excess"], df_reg["sp500_excess"]
    )  # 30-day windows
    rolling_betas.name = "Rolling_Beta"
    rolling_betas.plot(title="Rolling Beta (30-day window)")
    plt.show()

    # R_portfolio = alpha + beta_1 * R_market + beta_2 * R_market, lagged + epsilon
    sp500_excess_lagged = df_reg["sp500_excess"].shift(1)  # test different lagged
    sp500_excess_lagged.name = "sp500_excess_lagged"
    X = pd.concat(
        [df_reg["sp500_excess"], sp500_excess_lagged],
        axis=1,
    ).dropna()
    y = df_reg["portfolio_excess"][len(df_reg["portfolio_excess"]) - len(X) :]
    X = sm.add_constant(X)

    print("------------------- w/ Lagged SP500 -------------------")
    model = sm.OLS(y, X).fit()
    print(model.params)
    print(model.t_test([1, 0, 0]))
    print(model.t_test([0, 1, 0]))
    print(model.t_test([0, 0, 1]))


# alpha ~ model.params["const"]
# beta ~ model.params["sp500_excess"]

if __name__ == "__main__":

    main()
