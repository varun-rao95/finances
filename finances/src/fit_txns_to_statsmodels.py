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
    Loads the transactions data from a CSV file and optionally filters by date.

    Args:
        filter_by (str, optional): String specifying filter conditions on the "Date"
            column.
            Supported formats:
                - ">=YYYY-MM-DD" to keep rows on or after the given date
                - "<=YYYY-MM-DD" to keep rows on or before the given date
                - "=YYYY-MM-DD"  to keep rows on the given date

    Returns:
        pd.DataFrame: The loaded (and possibly filtered) transactions DataFrame.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, TXNS_FILE))

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Apply any requested date filtering
    if filter_by is not None:
        if filter_by.startswith(">="):
            date_string = filter_by.lstrip(">=")
            df = df[df["Date"] >= date_string]
        elif filter_by.startswith("<="):
            date_string = filter_by.lstrip("<=")
            df = df[df["Date"] <= date_string]
        else:
            # Assume exact match if not >= or <=
            date_string = filter_by.lstrip("=")
            df = df[df["Date"] == date_string]

    return df


def plot_spend(df, monthly=False, weekly=False, daily=False, plot=True):
    """
    Aggregates and optionally plots spend data on a monthly, weekly, or daily basis.
    Exactly one of monthly, weekly, or daily must be True.

    Args:
        df (pd.DataFrame): Transactions DataFrame. Must contain "Date" and "Amount" columns.
        monthly (bool): If True, aggregate by month.
        weekly (bool): If True, aggregate by week.
        daily (bool): If True, aggregate by day.
        plot (bool): If True, produce a histogram plot with labels for each bin.

    Returns:
        pd.DataFrame: A DataFrame of the aggregated spend for the chosen period
            with columns ["<Period>", "Amount"].
    """
    # Check that exactly one of (monthly, weekly, daily) is True
    if sum([monthly, weekly, daily]) != 1:
        raise ValueError("Exactly one of monthly, weekly, or daily must be True.")

    # Calculate aggregated spend
    if monthly:
        df["Month"] = df["Date"].dt.to_period("M")
        spend = df.groupby("Month")["Amount"].sum().reset_index()
        x_label = "Monthly Spend"
    elif weekly:
        df["Week"] = df["Date"].dt.isocalendar().week
        spend = df.groupby("Week")["Amount"].sum().reset_index()
        x_label = "Weekly Spend"
    else:
        spend = df.groupby("Date")["Amount"].sum().reset_index()
        x_label = "Daily Spend"

    if not plot:
        return spend

    # Plot the spend distribution
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=spend, x="Amount", kde=True, stat="density")
    n = len(spend)
    for patch in ax.patches:
        bin_width = patch.get_width()
        density = patch.get_height()
        count = density * n * bin_width
        bin_center = patch.get_x() + bin_width / 2
        ax.text(
            bin_center,
            density,
            f"{int(np.round(count))}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.title(f"{x_label} Distribution")
    plt.show()

    return spend


def analyze_daily_patterns(spend_df, plot_cv_search=False, plot=False):
    """
    Analyzes daily spend patterns using KDE with bandwidth selection via GridSearchCV.
    Also plots the histogram of daily spend with various bandwidth adjustments.

    Args:
        spend_df (pd.DataFrame): A DataFrame containing 'Amount' for daily spending.
        plot_cv_search (bool): If True, plots the cross-validated log-likelihood
            versus bandwidth.
        plot (bool): If True, plots the distribution w/ kde lines for given bandwidth.

    Returns:
        np.ndarray: The array of daily spend 'bucket thresholds' found by local minima
            of the KDE.
    """
    # Reshape data for KDE
    data = spend_df["Amount"].values[:, None]

    # GridSearchCV for bandwidth selection
    bandwidths = np.linspace(0.1, 30, 120)
    params = {"bandwidth": bandwidths}
    grid = GridSearchCV(KernelDensity(kernel="gaussian"), params, cv=5)
    grid.fit(data)

    # Extract info about bandwidth search
    mean_log_likelihood = grid.cv_results_["mean_test_score"]
    bandwidths_tested = grid.cv_results_["param_bandwidth"].data

    if plot_cv_search:
        plt.figure(figsize=(8, 5))
        plt.plot(bandwidths_tested, mean_log_likelihood, marker="o")
        plt.xlabel("Bandwidth")
        plt.ylabel("Mean Log-Likelihood")
        plt.title("Cross-Validated Log-Likelihood vs. Bandwidth")
        plt.grid(True)
        plt.show()

    # Fit a KDE with a (custom) bandwidth
    kde_custom = gaussian_kde(
        data.reshape(1, -1),
        bw_method=10 / np.std(data),  # Example: artificially wide or spiky bandwidth
    )

    # Evaluate the KDE at a range of points
    x = np.linspace(min(data), max(data), 1000)
    x = x.T
    kde_values = kde_custom.evaluate(x)

    # Identify local minima indices
    minima = argrelextrema(kde_values, np.less)[0]

    # Evaluate a default KDE to compare bandwidth
    kde_default = gaussian_kde(data.reshape(1, -1))
    bw_default = kde_default.factor * np.std(data)
    print("Default bandwidth:", bw_default)

    # Example set of bandwidths to overlay
    best_bandwidths = [grid.best_params_["bandwidth"], 20, 10, 100, 90]
    print(f"Optimal bandwidth: {best_bandwidths}")

    # Calculate how each bandwidth compares to the default
    bw_adjust = [best_bw / bw_default for best_bw in best_bandwidths]
    print(f"Optimal bw_adjust: {bw_adjust}")

    if not plot:
        return x[0, minima]

    # Final plot with multiple KDE lines
    plt.figure(figsize=(15, 10))
    sns.histplot(
        data=spend_df, x="Amount", stat="density", color="lightgrey", alpha=0.5
    )

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
    plt.show()

    return x[0, minima]  # The local minima "bucket thresholds"


def spend_likelihood(amount, spend):
    """
    Calculates the survival function (1 - CDF) of a normal distribution of spend.

    Args:
        amount (float): The spend value of interest.
        spend (pd.DataFrame): DataFrame containing 'Amount' column.

    Returns:
        float: Probability that a normal distribution (fit by mean and std of 'Amount')
            exceeds the given amount.
    """
    mean_spend = spend["Amount"].mean()
    std_spend = spend["Amount"].std()
    return norm.sf(amount, loc=mean_spend, scale=std_spend)


def sophis_spend_likelihood(amount, model, mean_txn_value=100):
    """
    Computes the probability of exceeding a certain spend level,
    given a model (negative binomial, poisson, or continuous).

    Args:
        amount (float): Amount threshold to calculate the probability of exceeding.
        model (statsmodels results): The fitted model used to predict likelihood.
        mean_txn_value (float, optional): Average transaction value used to estimate
            number of transactions.

    Returns:
        float: Probability of exceeding the `amount`.
    """
    print(f"Mean txn value: {mean_txn_value:.2f}")

    # Estimate the number of transactions needed to reach the amount
    k_threshold = int(amount / mean_txn_value)
    print(f"Num transactions to achieve {amount:.2f}: {k_threshold}")

    # Negative Binomial
    if isinstance(
        model, statsmodels.discrete.discrete_model.NegativeBinomialResultsWrapper
    ):
        predicted_mean_nb = model.predict()[0]
        alpha = model.scale
        n_param = 1.0 / alpha
        p_param = n_param / (n_param + predicted_mean_nb)

        p_spend_over_nb = nbinom.sf(k_threshold, n_param, p_param)
        print(
            f"Probability (Negative Binomial) > {k_threshold} transactions: {p_spend_over_nb:.4f}"
        )
        return p_spend_over_nb

    # Poisson (Note: this was likely intended for a PoissonResultsWrapper check)
    elif isinstance(model, statsmodels.discrete.discrete_model.PoissonResultsWrapper):
        predicted_mean_poisson = model.predict()[0]
        p_spend_over_poisson = poisson.sf(k_threshold, predicted_mean_poisson)
        print(
            f"Probability (Poisson) > {k_threshold} transactions: {p_spend_over_poisson:.4f}"
        )
        return p_spend_over_poisson

    # Otherwise, default to a continuous distribution assumption
    else:
        # This code assumes the model has a .cdf method (e.g., from a frozen scipy distribution)
        return 1 - model.cdf(amount)


def sophis_spend_likelihood_monte_carlo(
    amount, bucket_models, avg_bucket_spend, num_simulations=10_000
):
    """
    Monte Carlo simulation of spend likelihood using fitted bucket models.

    Args:
        amount (float or list): Threshold amount(s) to assess. If a list is given, it
            calculates probability for each amount in that list.
        bucket_models (dict): Dictionary of bucket thresholds -> fitted model.
        avg_bucket_spend (dict): Dictionary of bucket thresholds -> mean spend for that bucket.
        num_simulations (int, optional): Number of Monte Carlo draws.

    Returns:
        tuple:
            (probability, simulated_spend_array)
            - probability: float or list of floats (if multiple `amount` values).
            - simulated_spend_array: A 1D NumPy array of simulated total daily spends.
    """
    simulated_spend = simulate_daily_spend(
        bucket_models, avg_bucket_spend, num_simulations
    )

    if isinstance(amount, list):
        probability = [np.mean(simulated_spend > amt) for amt in amount]
        for amt, prob in zip(amount, probability):
            print(f"Probability of spending more than {amt:.2f}: {prob:.4f}")
    else:
        probability = np.mean(simulated_spend > amount)
        print(f"Probability of spending more than {amount:.2f}: {probability:.4f}")

    return probability, simulated_spend


def get_txns_per_day(df, colname="Transactions"):
    """
    Aggregates the given DataFrame by date, counting the number of rows per day.

    Args:
        df (pd.DataFrame): Transaction data with a 'Date' column.
        colname (str, optional): Name of the column that will hold the daily transaction count.

    Returns:
        pd.DataFrame: A DataFrame indexed by daily date range (including missing days),
            with one column for transaction counts.
    """
    df_transactions_per_day = df.groupby("Date").size().reset_index(name=colname)
    df_transactions_per_day.index = df_transactions_per_day["Date"]

    # Reindex to fill missing days
    new_index = pd.date_range(
        start=df_transactions_per_day.index.min(),
        end=df_transactions_per_day.index.max(),
        freq="D",
    )
    df_transactions_per_day = df_transactions_per_day.reindex(new_index, fill_value=0)
    df_transactions_per_day["Date"] = df_transactions_per_day.index
    return df_transactions_per_day


def find_best_fit_model(df_amounts, df_txns_per_day, colname="Transactions"):
    """
    Fits Poisson, Negative Binomial, Gamma, and Log-Normal models to the given data
    and selects the best model by comparing AIC.

    Args:
        df_amounts (pd.DataFrame): Data containing 'Amount' (used for continuous distributions).
        df_txns_per_day (pd.DataFrame): Data containing daily counts (for discrete distributions).
        colname (str, optional): Name of the column in df_txns_per_day that holds
            transaction counts.

    Returns:
        statsmodels.base.model.Results: The best-fitting model results object.
    """
    # Fit Poisson
    poisson_model_results = sm.Poisson(
        df_txns_per_day[colname], np.ones(len(df_txns_per_day))
    ).fit(disp=0)

    # Fit Negative Binomial
    nbinom_model_results = sm.NegativeBinomial(
        df_txns_per_day[colname], np.ones(len(df_txns_per_day))
    ).fit(disp=0)

    # Fit Gamma
    gamma_model_results = sm.GLM(
        df_amounts["Amount"],
        sm.add_constant(np.ones(len(df_amounts))),
        family=sm.families.Gamma(),
    ).fit(disp=0)

    # Fit Log-Normal
    lognorm_model_results = sm.OLS(
        np.log(df_amounts["Amount"]), sm.add_constant(np.ones(len(df_amounts)))
    ).fit(disp=0)

    models = {
        "Poisson": poisson_model_results,
        "Negative Binomial": nbinom_model_results,
        "Gamma": gamma_model_results,
        "Log-Normal": lognorm_model_results,
    }

    # Choose the best model by min AIC
    best_model_name, best_model_results = min(
        models.items(),
        key=lambda item: item[1].aic,
    )
    print(f"\nBest model: {best_model_name} with AIC: {best_model_results.aic}")

    return best_model_results


def simulate_spend(model, n=1000, plot=False):
    """
    Simulates spend data from a provided model's .rvs() method and optionally plots
    the distribution.

    Args:
        model (scipy.stats distribution or similar): A model with an .rvs(size=n) method.
        n (int, optional): Number of samples to draw.
        plot (bool, optional): If True, plots the resulting histogram.

    Returns:
        np.ndarray: Samples drawn from the model.
    """
    samples = model.rvs(size=n)

    if plot:
        plt.hist(samples, bins=20, alpha=0.7)
        plt.xlabel("Spend")
        plt.ylabel("Frequency")
        plt.title("Simulated Spend Distribution")
        plt.show()

    return samples


def simulate_daily_spend(bucket_models, avg_bucket_spend, num_simulations=10_000):
    """
    Monte Carlo simulation of daily spend using a collection of fitted models
    (one per 'bucket'). Each model handles a portion of spend or transaction distribution.

    Args:
        bucket_models (dict): Mapping of bucket threshold -> model result object.
        avg_bucket_spend (dict): Mapping of bucket threshold -> average spend for that bucket.
        num_simulations (int, optional): Number of daily simulations.

    Returns:
        np.ndarray: Array of size `num_simulations` with total spend for each simulated day.
    """
    daily_spends = np.zeros(num_simulations)

    for bucket, model_results in bucket_models.items():
        # Check model type and simulate accordingly
        if isinstance(
            model_results,
            statsmodels.discrete.discrete_model.NegativeBinomialResultsWrapper,
        ):
            predicted_mean_nb = model_results.predict()[0]
            alpha = model_results.scale
            n_param = 1.0 / alpha
            p_param = n_param / (n_param + predicted_mean_nb)
            counts = nbinom.rvs(n_param, p_param, size=num_simulations)
            spend_for_bucket = counts * avg_bucket_spend[bucket]

        elif isinstance(
            model_results, statsmodels.discrete.discrete_model.PoissonResultsWrapper
        ):
            predicted_mean_poisson = model_results.predict()[0]
            counts = poisson.rvs(predicted_mean_poisson, size=num_simulations)
            spend_for_bucket = counts * avg_bucket_spend[bucket]

        elif isinstance(
            model_results, statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        ):
            # Interpreted as a Gamma model
            shape = model_results.params[0]  # Intercept (rough approximation)
            scale = model_results.scale
            spend_for_bucket = gamma.rvs(shape, scale=scale, size=num_simulations)

        elif isinstance(
            model_results, statsmodels.regression.linear_model.RegressionResultsWrapper
        ):
            # Interpreted as a LogNormal model
            mean_log = model_results.params[0]
            std_log = model_results.bse[0]
            spend_for_bucket = lognorm.rvs(
                std_log, scale=np.exp(mean_log), size=num_simulations
            )

        else:
            # If other model types appear in future, handle here
            spend_for_bucket = np.zeros(num_simulations)

        daily_spends += spend_for_bucket

    return daily_spends


def compare_distribution_for_two_time_periods(df1, df2):
    """
    Placeholder function to compare distributions of two time periods.

    Args:
        df1 (pd.DataFrame): DataFrame for the first time period.
        df2 (pd.DataFrame): DataFrame for the second time period.

    Returns:
        None
    """
    pass


def estimate_budget_for_confidence_level(model, confidence_level=0.95):
    """
    Estimates the budget for a given confidence level from a model's .ppf().

    Args:
        model (scipy.stats distribution): A distribution with .ppf() method.
        confidence_level (float, optional): Confidence level for the budget estimate.

    Returns:
        float: The estimated budget threshold.
    """
    budget = model.ppf(confidence_level)
    print(f"Estimated budget at {confidence_level*100:.0f}% confidence level: {budget}")
    return budget


def remove_outliers(df, iqr_multiplier=1.5, right_tail_only=False):
    """
    Removes outliers from 'Amount' column based on IQR.

    Args:
        df (pd.DataFrame): The DataFrame containing an 'Amount' column.
        iqr_multiplier (float, optional): The factor multiplied by IQR to determine cutoff.
        right_tail_only (bool, optional): If True, only remove outliers on the right tail.

    Returns:
        tuple: (df_filtered, df_outliers)
            - df_filtered: DataFrame without outliers.
            - df_outliers: DataFrame of outliers only.
    """
    Q1 = df["Amount"].quantile(0.25)
    Q3 = df["Amount"].quantile(0.75)
    IQR = Q3 - Q1
    print(f"IQR: {IQR}")

    if right_tail_only:
        mask = df["Amount"] > (Q3 + iqr_multiplier * IQR)
    else:
        mask = (df["Amount"] < (Q1 - iqr_multiplier * IQR)) | (
            df["Amount"] > (Q3 + iqr_multiplier * IQR)
        )
    df_outliers = df[mask]
    df_filtered = df[~mask]

    print(f"Removed {len(df_outliers)} outliers out of {len(df)} transactions")

    # Save outliers and filtered data for inspection
    df_outliers.to_csv(os.path.join(DATA_DIR, "outliers.csv"), index=False)
    df_filtered.to_csv(os.path.join(DATA_DIR, "filtered.csv"), index=False)

    return df_filtered, df_outliers


def filter_category(df, categories=None):
    """
    Filters out transactions whose 'Category' is in a given list.

    Args:
        df (pd.DataFrame): Transaction DataFrame with 'Category' column.
        categories (list, optional): Categories to exclude.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if categories is None:
        categories = []
    return df[~df["Category"].isin(categories)]


def kde_bucket_thresholds(amount, bucket_thresholds):
    """
    Identifies which bucket an amount belongs to based on threshold cutoffs.

    Args:
        amount (float): The transaction amount.
        bucket_thresholds (list): Sorted list of threshold boundaries.

    Returns:
        int: The bucket index for the given amount.
    """
    for idx, thresh in enumerate(bucket_thresholds):
        if amount < thresh:
            return idx
    return len(bucket_thresholds)


def fit_models_for_each_bucket(df, bucket_thresholds):
    """
    Splits the data into buckets, finds best fit model for each bucket, and calculates
    average spend within each bucket.

    Args:
        df (pd.DataFrame): DataFrame with "bucket" column assigned (via `kde_bucket_thresholds`)
            and "Amount".
        bucket_thresholds (list): List of threshold values that define buckets.

    Returns:
        tuple: (bucket_models, avg_bucket_spend)
            - bucket_models: dict of bucket threshold -> best fit model
            - avg_bucket_spend: dict of bucket threshold -> average spend for that bucket
    """
    bucket_models = {}
    avg_bucket_spend = {}

    for idx, threshold in enumerate(bucket_thresholds):
        # Data for this bucket index
        bucket_data = df[df["bucket"] == idx]
        if bucket_data.empty:
            continue

        # Prepare daily transaction and daily amounts
        bucket_txns_per_day = (
            get_txns_per_day(bucket_data)
            .groupby("Date")["Transactions"]
            .sum()
            .reset_index()
        )
        bucket_data_amounts = bucket_data.groupby("Date")["Amount"].sum().reset_index()

        # Reindex to fill missing days
        bucket_data_amounts.index = bucket_data_amounts["Date"]
        new_index = pd.date_range(
            start=bucket_data_amounts.index.min(),
            end=bucket_data_amounts.index.max(),
            freq="D",
        )
        bucket_data_amounts = bucket_data_amounts.reindex(new_index, fill_value=0.01)
        bucket_data_amounts["Date"] = bucket_data_amounts.index

        # Find best fit for this bucket
        best_model = find_best_fit_model(bucket_data_amounts, bucket_txns_per_day)

        # Store the best model
        bucket_models[threshold] = best_model

        # Calculate average spend for this bucket
        # (Exclude the artificially filled 0.01 rows)
        avg_bucket_spend[threshold] = bucket_data_amounts[
            bucket_data_amounts["Amount"] != 0.01
        ]["Amount"].mean()

    return bucket_models, avg_bucket_spend


def empirical_cdf(data):
    """Returns x, cdf(x) for the given data."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
    return sorted_data, cdf


def sim_real_cdf_overlay(real_spend, simulated_spend):
    cmp_simulated_daily_spends = np.random.choice(
        simulated_spend, size=len(real_spend), replace=False
    )
    x_real, cdf_real = empirical_cdf(real_spend)
    x_sim, cdf_sim = empirical_cdf(cmp_simulated_daily_spends)

    plt.figure()
    plt.plot(x_real, cdf_real, label="Empirical CDF")
    plt.plot(x_sim, cdf_sim, label="Simulated CDF")
    plt.title("CDF Comparison: Daily Totals")
    plt.xlabel("Daily Spend")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.show()


def main():
    """
    Main function to demonstrate a workflow of:
    - Loading data for different time windows
    - Filtering unwanted categories
    - Plotting daily spend distributions
    - Analyzing daily patterns with KDE
    - Bucket assignment
    - Fitting models for each bucket
    - Monte Carlo simulation for probability of exceeding various spend amounts
    """
    df_2024 = load_data(filter_by=">=2024")
    df_2023 = load_data(filter_by="<=2023")

    df_combined = pd.concat([df_2023, df_2024])
    df_filtered = filter_category(
        df_combined,
        categories=[
            "Credit Card Payment",
            "Transfer",
            "Direct Payment",
            "Investments",
            "Buy",
        ],
    )

    # Get daily spend but don't plot (just returning aggregated data)
    daily_spend = plot_spend(
        df_filtered, monthly=False, weekly=False, daily=True, plot=False
    )

    # Identify bucket thresholds with local minima from daily spend distribution
    bucket_thresholds = analyze_daily_patterns(df_filtered)
    bucket_thresholds = bucket_thresholds[:6]  # keep the first 6 minima, for example

    # Create a 'bucket' column
    df_filtered["bucket"] = df_filtered["Amount"].apply(
        lambda x: kde_bucket_thresholds(x, bucket_thresholds)
    )

    # Fit models for each bucket
    bucket_models, avg_bucket_spend = fit_models_for_each_bucket(
        df_filtered, bucket_thresholds
    )

    # Example: run a Monte Carlo to see probabilities for multiple spend levels
    spend_amounts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    likelihoods, simulated_daily_spends = sophis_spend_likelihood_monte_carlo(
        spend_amounts, bucket_models, avg_bucket_spend
    )

    sim_real_cdf_overlay(daily_spend["Amount"], simulated_daily_spends)


if __name__ == "__main__":
    main()
