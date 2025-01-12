import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom, gamma, lognorm, norm
import statsmodels.api as sm

TXNS_FILE = "doordash.csv"
DATA_DIR = os.path.join("..", "data")


def load_data(filter_by=None):
    """
    Load the transactions data from the CSV file and filters by optional date.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, TXNS_FILE))
    # Convert date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    if filter_by is not None:
        if filter_by.startswith(">="):
            filter_by = filter_by.lstrip(">=")
            df = df[df["Date"] >= filter_by]
        elif filter_by.startswith("<="):
            filter_by = filter_by.lstrip("<=")
            df = df[df["Date"] <= filter_by]
        else:
            filter_by = filter_by.lstrip("=")
            df = df[df["Date"] == filter_by]

    return df


def plot_spend(df, monthly=True, weekly=False, daily=False, plot=True):
    if sum([monthly, weekly, daily]) != 1:
        raise ValueError("Exactly one of monthly, weekly, or daily must be True.")

    if monthly:
        # Calculate monthly spend
        df["month"] = df["Date"].dt.month
        spend = df.groupby("month")["Amount"].sum().reset_index()
        xlabel = "Montly Spend"

    elif weekly:
        # Calculate weekly spend
        df["week"] = df["Date"].dt.isocalendar().week
        spend = df.groupby("week")["Amount"].sum().reset_index()
        xlabel = "Weekly Spend"

    else:
        spend = df.grpuopby("Date")["Amount"].sum().reset_index()
        xlabel = "Daily Spend"

    # Plot spend distribution
    if not plot:
        return spend

    plt.figure(figsize=(10, 6))
    plt.hist(spend["Amount"], bins=20, alpha=0.7, color="blue")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(xlabel + " Distribution")
    plt.show()

    return spend


# Define a function to calculate likelihood
def spend_likelihood(amount, spend):
    mean_spend = spend["Amount"].mean()
    std_spend = spend["Amount"].std()

    return norm.pdf(amount, loc=mean_spend, scale=std_spend)


def sophis_spend_likelihood(amount, model):
    return model.pdf(amount)


def get_txns_per_day(df, colname="Transactions"):
    df_transactions_per_day = df.groupby("Date").size().reset_index(name=colname)
    return df_transactions_per_day


def find_best_fit_model(df, df_txns_per_day, colname="Transactions"):
    # Fit a Poisson distribution
    poisson_model = poisson.fit(df_txns_per_day[colname])

    # Fit a Negative Binomial distribution
    nbinom_model = nbinom.fit(df["transactions_per_day"])

    # Fit a Gamma distribution
    gamma_model = gamma.fit(df["transaction_amounts"])

    # Fit a Log-Normal distribution
    lognorm_model = lognorm.fit(df["transaction_amounts"])

    models = {
        "Poisson": poisson_model,
        "Negative Binomial": nbinom_model,
        "Gamma": gamma_model,
        "Log-Normal": lognorm_model,
    }

    best_model_name = min(models, key=lambda x: models[x].aic)
    model = models[best_model_name]
    print()
    print(f"Best model: {best_model_name} with AIC: {model.aic}")
    print(", ".join(sorted(models, key=lambda x: x[1].aic)))

    return model


def simulate_spend(model, n=1000, plot=False):
    samples = model.rvs(size=n)

    # Plot simulated spend distribution
    plt.hist(samples, bins=20, alpha=0.7, color="blue")
    plt.xlabel("Spend")
    plt.ylabel("Frequency")
    plt.title("Simulated Spend Distribution")
    plt.show()

    return samples


def compare_distribution_for_two_time_periods(df1, df2):
    pass


def estimate_budget_for_confidence_level(model, confidence_level=0.95):
    # Calculate the budget for a given confidence level
    budget = model.ppf(confidence_level)
    print("Estimated budget for a given confidence level:", budget)

    return budget


def main():
    df_2024 = load_data(filter_by=">=2024")
    df_2023 = load_data(filter_by="=2023")

    # OPTIONAL: here 2023 data is pretty trashy, some venmo + paypal shopping transactions appear here to be mixed w/ doordash
    # Calculate IQR (Interquartile Range)
    Q1 = df_2023["Amount"].quantile(0.25)
    Q3 = df_2023["Amount"].quantile(0.75)
    IQR = Q3 - Q1

    # Remove outliers
    df_2023_filtered = df_2023[
        ~(
            (df_2023["Amount"] < (Q1 - 1.5 * IQR))
            | (df_2023["Amount"] > (Q3 + 1.5 * IQR))
        )
    ]

    # Combine filtered 2023 data with 2024 data
    df = pd.concat([df_2023_filtered, df_2024])

    # Plot daily spend distribution
    daily_spend = plot_spend(df, monthly=False, weekly=False, daily=True, plot=True)

    # Plot weekly spend distribution
    weekly_spend = plot_spend(df, monthly=False, weekly=True, daily=False, plot=True)

    # Plot monthly spend distribution
    monthly_spend = plot_spend(df, monthly=True, weekly=False, daily=False, plot=True)

    # answer likelihood to spend amount in week
    print(
        f"Likelihood to spend ${100} this week: {spend_likelihood(100, weekly_spend):.4f}"
    )
    # answer likelihood to spend amount in week
    print(
        f"Likelihood to spend ${1000} this month: {spend_likelihood(1000, monthly_spend):.4f}"
    )

    # TODO: do the same with txns per week and per month
    df_txns_per_day = get_txns_per_day(df)
    model = find_best_fit_model(df, df_txns_per_day)

    # Fit a Gamma-Poisson model
    gamma_poisson_model = sm.GammaPoisson(df_txns_per_day["Transactions"], df["Amount"])

    # Fit a Compound Poisson model
    compound_poisson_model = sm.CompoundPoisson(
        df_txns_per_day["Transactions"], df["Amount"], dist="gamma"
    )

    # Fit a Tweedie model
    tweedie_model = sm.Tweedie(df_txns_per_day["Transactions"], df["Amount"], p=1.5)


if __name__ == "__main__":
    main()
