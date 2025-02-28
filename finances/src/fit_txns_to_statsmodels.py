import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma, norm, poisson, nbinom, lognorm
import statsmodels.api as sm
import statsmodels
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

TXNS_FILE = "spend_txns.csv"
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


def plot_spend(df, monthly=False, weekly=False, daily=False, plot=True):
    if sum([monthly, weekly, daily]) != 1:
        raise ValueError(
            "Exactly one of monthly, weekly, or daily must be True. Pass it in."
        )

    if monthly:
        # Calculate monthly spend
        df["month"] = df["Date"].dt.to_period("M")
        spend = df.groupby("month")["Amount"].sum().reset_index()
        xlabel = "Montly Spend"

    elif weekly:
        # Calculate weekly spend
        df["week"] = df["Date"].dt.isocalendar().week
        spend = df.groupby("week")["Amount"].sum().reset_index()
        xlabel = "Weekly Spend"

    else:
        spend = df.groupby("Date")["Amount"].sum().reset_index()
        xlabel = "Daily Spend"

    # Plot spend distribution
    if not plot:
        return spend

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=spend, x="Amount", kde=True, stat="density")
    n = len(spend)
    for patch in ax.patches:
        # Bin width and density from the patch.
        bin_width = patch.get_width()
        density = patch.get_height()

        # Calculate the count from density: count = density * n * bin_width.
        count = density * n * bin_width

        # Determine the center of the bin.
        bin_center = patch.get_x() + bin_width / 2

        # Place the count label above the patch.
        ax.text(
            bin_center,
            density,
            f"{int(np.round(count))}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(xlabel + " Distribution")

    return spend


def analyze_daily_patterns(spend_df, plot_cv_search=False):
    data = spend_df["Amount"].values[:, None]  # Reshape data to (N, 1) for KDE fitting

    # Set up GridSearchCV for bandwidth selection
    bandwidths = np.linspace(0.1, 30, 120)
    params = {"bandwidth": bandwidths}
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"), params, cv=5
    )  # 5-fold cross-validation
    grid.fit(data)

    # Extract the bandwidths and corresponding mean test scores (log-likelihoods)
    mean_log_likelihood = grid.cv_results_["mean_test_score"]
    bandwidths_tested = grid.cv_results_[
        "param_bandwidth"
    ].data  # Access the array of bandwidth values

    # Plot the log-likelihood across bandwidth values
    if plot_cv_search:
        plt.figure(figsize=(8, 5))
        plt.plot(bandwidths_tested, mean_log_likelihood, marker="o")
        plt.xlabel("Bandwidth")
        plt.ylabel("Mean Log-Likelihood")
        plt.title("Cross-Validated Log-Likelihood vs. Bandwidth")
        plt.grid(True)
        plt.show()

    # Fit the KDE with the best bandwidth
    kde = gaussian_kde(
        data.reshape(1, -1),
        bw_method=10 / np.std(data),  # 10 is way more spiky so good for bucketing
    )

    # Create an array of evaluation points for the KDE
    x = np.linspace(min(data), max(data), 1000)
    x = x.T  # Reshape to (1, 1000)

    # Evaluate the KDE at the points in x
    kde_values = kde.evaluate(x)

    # Find the local minima (troughs) to define bucket boundaries
    minima = argrelextrema(kde_values, np.less)[0]  # Local minima indices

    kde = gaussian_kde(data.reshape(1, -1))
    bw_default = kde.factor * np.std(data)
    print("Default bandwidth:", bw_default)

    best_bandwidths = [grid.best_params_["bandwidth"], 20, 10, 100, 90]
    # [2.0, 5.0, 8.0, bw_default]
    print(f"Optimal bandwidth: {best_bandwidths}")

    bw_adjust = [
        best_bandwidth / bw_default for best_bandwidth in best_bandwidths
    ]  # assuming you know or compute bw_default
    print(f"Optimal bw_adjust: {bw_adjust}")

    plt.figure(figsize=(15, 10))

    # Plot the histogram as background
    sns.histplot(
        data=spend_df, x="Amount", stat="density", color="lightgrey", alpha=0.5
    )

    # Plot KDE lines for each bw_adjust value
    for b_adjust in bw_adjust:
        sns.kdeplot(
            data=spend_df,
            x="Amount",
            bw_adjust=b_adjust,
            label=f"bw_adjust={b_adjust:.2f}",
        )

    plt.title("Daily Spend Distribution with Various bw_adjust Values")
    plt.xlabel("Amount")
    plt.ylabel("Density")
    plt.legend()

    return x[0, minima]  # bucket thresholds


# Define a function to calculate likelihood
def spend_likelihood(amount, spend):
    mean_spend = spend["Amount"].mean()
    std_spend = spend["Amount"].std()

    return norm.sf(amount, loc=mean_spend, scale=std_spend)


def sophis_spend_likelihood(amount, model, mean_txn_value=100):
    print(f"Mean txn value: {mean_txn_value:.2f}")
    k_threshold = int(amount / mean_txn_value)
    print(f"Num transactions to achieve {amount:.2f}: {k_threshold}")
    if isinstance(
        model, statsmodels.discrete.discrete_model.NegativeBinomialResultsWrapper
    ):
        predicted_mean_nb = model.predict()[0]
        alpha = model.scale

        n_param = 1.0 / alpha
        p_param = n_param / (n_param + predicted_mean_nb)

        p_spend_over_nb = nbinom.sf(k_threshold, n_param, p_param)
        print(
            f"Probability (Negative Binomial) of having more than {k_threshold} transactions: {p_spend_over_nb:.4f}"
        )
        return p_spend_over_nb
    elif isinstance(
        model, statsmodels.discrete.discrete_model.NegativeBinomialResultsWrapper
    ):
        predicted_mean_poisson = model.predict()[0]
        p_spend_over_poisson = poisson.sf(k_threshold, predicted_mean_poisson)
        print(
            f"Probability (Poisson) of having more than {k_threshold} transactions: {p_spend_over_poisson:.4f}"
        )
        return p_spend_over_poisson
    else:
        return 1 - model.cdf(amount)


def sophis_spend_likelihood_monte_carlo(
    amount, bucket_models, avg_bucket_spend, num_simulations=10_000
):
    """
    Monte Carlo simulation of spend likelihood using the best-fit models for each bucket.
    Returns the likelihood of spending more than `amount`.
    """
    simulated_spend = simulate_daily_spend(
        bucket_models, avg_bucket_spend, num_simulations
    )
    if isinstance(amount, list):  # TODO: iterable
        probability = [np.mean(simulated_spend > spend) for spend in amount]
        for spend, prob in zip(amount, probability):
            print(f"Probability of spending more than {spend:.2f}: {prob:.4f}")
    else:
        probability = np.mean(simulated_spend > amount)
        print(f"Probability of spending more than {amount:.2f}: {probability:.4f}")
    return probability


def get_txns_per_day(df, colname="Transactions"):
    df_transactions_per_day = df.groupby("Date").size().reset_index(name=colname)
    df_transactions_per_day.index = df_transactions_per_day["Date"]
    new_index = pd.date_range(
        start=df_transactions_per_day.index.min(),
        end=df_transactions_per_day.index.max(),
        freq="D",
    )
    df_transactions_per_day = df_transactions_per_day.reindex(new_index, fill_value=0)
    df_transactions_per_day["Date"] = df_transactions_per_day.index

    return df_transactions_per_day


def find_best_fit_model(df, df_txns_per_day, colname="Transactions"):
    # Fit a Poisson distribution
    poisson_model_results = sm.Poisson(
        df_txns_per_day[colname], np.ones(len(df_txns_per_day))
    ).fit(disp=0)

    # Fit a Negative Binomial distribution
    nbinom_model_results = sm.NegativeBinomial(
        df_txns_per_day[colname], np.ones(len(df_txns_per_day))
    ).fit(disp=0)

    # Fit a Gamma distribution
    gamma_model_results = sm.GLM(
        df["Amount"],
        sm.add_constant(np.ones(len(df))),
        family=sm.families.Gamma(),
    ).fit(disp=0)

    # Fit a Log-Normal distribution
    lognorm_model_results = sm.OLS(
        np.log(df["Amount"]), sm.add_constant(np.ones(len(df)))
    ).fit(disp=0)

    models = {
        "Poisson": poisson_model_results,
        "Negative Binomial": nbinom_model_results,
        "Gamma": gamma_model_results,
        "Log-Normal": lognorm_model_results,
    }

    best_model_name = min(
        models.items(),
        key=lambda x: x[1].aic,
    )
    model_results = best_model_name[1]
    print()
    print(f"Best model: {best_model_name[0]} with AIC: {best_model_name[1].aic}")

    return model_results


def simulate_spend(model, n=1000, plot=False):
    samples = model.rvs(size=n)

    # Plot simulated spend distribution
    plt.hist(samples, bins=20, alpha=0.7, color="blue")
    plt.xlabel("Spend")
    plt.ylabel("Frequency")
    plt.title("Simulated Spend Distribution")
    plt.show()

    return samples


def simulate_daily_spend(bucket_models, avg_bucket_spend, num_simulations=10_000):
    """
    Monte Carlo simulation of daily spend using the best-fit models
    for each bucket of transaction amounts.
    """
    daily_spends = np.zeros(num_simulations)

    for bucket, model_results in bucket_models.items():
        # Extract parameters based on model type
        if isinstance(
            model_results,
            statsmodels.discrete.discrete_model.NegativeBinomialResultsWrapper,
        ):
            # Negative Binomial model
            predicted_mean_nb = model_results.predict()[0]
            alpha = model_results.scale

            n_param = 1.0 / alpha
            p_param = n_param / (n_param + predicted_mean_nb)

            # Simulate the daily transaction counts for this bucket
            counts = nbinom.rvs(n_param, p_param, size=num_simulations)
            spend_for_bucket = counts * avg_bucket_spend[bucket]

        elif isinstance(
            model_results, statsmodels.discrete.discrete_model.PoissonResultsWrapper
        ):
            # Poisson model
            predicted_mean_poisson = model_results.predict()[0]

            # Simulate the daily transaction counts for this bucket
            counts = poisson.rvs(predicted_mean_poisson, size=num_simulations)
            spend_for_bucket = counts * avg_bucket_spend[bucket]

        elif isinstance(
            model_results, statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        ):
            # Gamma model
            shape = model_results.params[
                0
            ]  # For Gamma, the shape is typically the intercept
            scale = model_results.scale  # Scale parameter from GLM

            # Simulate the daily transaction counts for this bucket (assume the count is Poisson distributed)
            spend_for_bucket = gamma.rvs(shape, scale=scale, size=num_simulations)

        elif isinstance(
            model_results, statsmodels.regression.linear_model.RegressionResultsWrapper
        ):
            # Log-Normal model
            mean_log = model_results.params[0]  # Intercept
            std_log = model_results.bse[0]  # Standard error from OLS

            # Simulate the daily transaction amounts for this bucket
            spend_for_bucket = lognorm.rvs(
                std_log, scale=np.exp(mean_log), size=num_simulations
            )

        # Add to the total daily spend
        daily_spends += spend_for_bucket

    return daily_spends


def compare_distribution_for_two_time_periods(df1, df2):
    pass


def estimate_budget_for_confidence_level(model, confidence_level=0.95):
    # Calculate the budget for a given confidence level
    budget = model.ppf(confidence_level)
    print("Estimated budget for a given confidence level:", budget)

    return budget


def remove_outliers(df, iqr_multiplier=1.5, right_tail_only=False):
    # Calculate IQR (Interquartile Range)
    Q1 = df["Amount"].quantile(0.25)
    Q3 = df["Amount"].quantile(0.75)
    IQR = Q3 - Q1
    print(f"IQR: {IQR}")

    # Remove outliers
    if right_tail_only:
        mask = df["Amount"] > (Q3 + iqr_multiplier * IQR)
    else:
        mask = (df["Amount"] < (Q1 - iqr_multiplier * IQR)) | (
            df["Amount"] > (Q3 + iqr_multiplier * IQR)
        )
    df_filtered = df[~(mask)]

    print(
        f"Removed {len(df) - len(df_filtered)} outliers out of {len(df)} transactions"
    )
    df.loc[mask].to_csv(os.path.join(DATA_DIR, "outliers.csv"), index=False)
    df_filtered.to_csv(os.path.join(DATA_DIR, "filtered.csv"), index=False)

    return (
        df_filtered,
        df[(mask)],
    )


def filter_category(df, categories=[]):
    return df[~df["Category"].isin(categories)]


def kde_bucket_thresholds(amount, bucket_thresholds):
    for idx, thresh in enumerate(bucket_thresholds):
        if amount < thresh:
            return idx

    return len(bucket_thresholds)


def main():
    df_2024 = load_data(filter_by=">=2024")
    df_2023 = load_data(filter_by="<=2023")

    # Combine filtered 2023 data with 2024 data
    df = filter_category(
        pd.concat([df_2023, df_2024]),
        categories=[
            "Credit Card Payment",
            "Transfer",
            "Direct Payment",
            "Investments",
            "Buy",
        ],
    )

    mean_txn_spend = df["Amount"].mean()

    # Plot daily spend distribution
    daily_spend = plot_spend(df, monthly=False, weekly=False, daily=True, plot=False)
    bucket_thresholds = analyze_daily_patterns(df)
    bucket_thresholds = bucket_thresholds[:6]

    df["bucket"] = df["Amount"].apply(
        lambda x: kde_bucket_thresholds(x, bucket_thresholds)
    )

    bucket_models = {}
    avg_bucket_spend = {}

    # Loop through each bucket and find the best fit model
    for idx, threshold in enumerate(bucket_thresholds):
        # Select the data for the current bucket
        bucket_data = df[df["bucket"] == idx]

        if len(bucket_data) == 0:
            continue  # Skip empty buckets

        # Compute df_txns_per_day for this bucket (daily transaction counts)
        bucket_txns_per_day = (
            get_txns_per_day(bucket_data)
            .groupby("Date")["Transactions"]
            .sum()
            .reset_index()
        )
        bucket_data = bucket_data.groupby("Date")["Amount"].sum().reset_index()
        bucket_data.index = bucket_data["Date"]
        new_index = pd.date_range(
            start=bucket_data.index.min(), end=bucket_data.index.max(), freq="D"
        )
        bucket_data = bucket_data.reindex(new_index, fill_value=0.01)
        bucket_data["Date"] = bucket_data.index

        # Call find_best_fit_model for each bucket
        best_model = find_best_fit_model(bucket_data, bucket_txns_per_day)

        # Store the best model for this bucket
        bucket_models[threshold] = best_model

        # Calculate average spend for this bucket
        avg_bucket_spend[threshold] = bucket_data[bucket_data["Amount"] != 0.01][
            "Amount"
        ].mean()

    # Group by date and bucket to get daily counts
    daily_counts = (
        df.groupby(["Date", "bucket"])["Amount"].size().reset_index(name="count")
    )

    # Pivot so each row is a day, and each column is the count for that bucket
    daily_bucket_counts = daily_counts.pivot_table(
        index="Date", columns="bucket", values="count", fill_value=0
    )

    spend_amounts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    likelihoods = sophis_spend_likelihood_monte_carlo(
        spend_amounts, bucket_models, avg_bucket_spend
    )
    return

    print(
        f"Mean daily spend w/ std: {daily_spend['Amount'].mean():.2f} {daily_spend['Amount'].std():.2f}"
    )

    # Plot weekly spend distribution
    # weekly_spend = plot_spend(df, monthly=False, weekly=True, daily=False, plot=True)
    # print(
    #     f"Mean weekly spend w/ std: {weekly_spend['Amount'].mean():.2f} {weekly_spend['Amount'].std():.2f}"
    # )

    # Plot monthly spend distribution
    monthly_spend = plot_spend(df, monthly=True, weekly=False, daily=False, plot=True)
    print(
        f"Mean monthly spend w/ std: {monthly_spend['Amount'].mean():.2f} {monthly_spend['Amount'].std():.2f}"
    )

    # answer likelihood to spend amount in week
    print(
        f"Likelihood to spend ${100} this day: {spend_likelihood(100, daily_spend):.4f}"
    )
    # answer likelihood to spend amount in week
    print(
        f"Likelihood to spend ${200} this day: {spend_likelihood(200, daily_spend):.4f}"
    )
    print(
        f"Likelihood to spend <$10 this day: {1-spend_likelihood(10, daily_spend):.4f}"
    )
    print(
        f"Likelihood to spend <$30 this day: {1-spend_likelihood(30, daily_spend):.4f}"
    )
    print(
        f"Likelihood to spend <$50 this day: {1-spend_likelihood(50, daily_spend):.4f}"
    )
    print(f"Likelihood to spend $0 this day: {1-spend_likelihood(0, daily_spend):.4f}")
    # answer likelihood to spend amount in week
    print(
        f"Likelihood to spend ${1000} this week: {spend_likelihood(1000, weekly_spend):.4f}"
    )
    # answer likelihood to spend amount in week
    print(
        f"Likelihood to spend ${6000} this week: {spend_likelihood(6000, weekly_spend):.4f}"
    )

    # TODO: do the same with txns per week and per month
    df_txns_per_day = get_txns_per_day(df)
    model_results = find_best_fit_model(df, df_txns_per_day)

    # answer likelihood to spend amount in week
    print(
        f"Likelihood to spend ${1000} this week: {sophis_spend_likelihood(1000/7, model_results, mean_txn_value=mean_txn_spend):.4f}"
    )
    # answer likelihood to spend amount in week
    print(
        f"Likelihood to spend ${2000} this week: {sophis_spend_likelihood(2000/7, model_results, mean_txn_value=mean_txn_spend):.4f}"
    )
    print(
        f"Likelihood to spend ${3000} this week: {sophis_spend_likelihood(3000/7, model_results, mean_txn_value=mean_txn_spend):.4f}"
    )

    plt.show()

    # # Fit Gamma-Poisson model
    # gamma_poisson_model = gamma.fit(
    #     df["Amount"][df_txns_per_day["Transactions"] > 0], floc=0
    # )

    # # Fit Tweedie model (approximated as Log-Normal)
    # tweedie_model = norm.fit(np.log(df["Amount"][df_txns_per_day["Transactions"] > 0]))


if __name__ == "__main__":
    main()
