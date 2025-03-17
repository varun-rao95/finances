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
    # TOOO: do correlations between buckets for prob of bucket i firing here too
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
    Sigma, bucket_models, num_simulations=10_000, by_bucket=False
):
    """
    Monte Carlo simulation of spend likelihood using fitted bucket models.

    Args:
        Sigma (np.array) BxB: Correlation matrix for buckets.
        bucket_models (dict): Dictionary of bucket thresholds -> fitted model.
        num_simulations (int, optional): Number of Monte Carlo draws.

    Returns:
        tuple:
            (probability, simulated_spend_array)
            - probability: float or list of floats (if multiple `amount` values).
            - simulated_spend_array: A 1D NumPy array of simulated total daily spends.
    """
    B = Sigma.shape[0]
    Z = np.random.multivariate_normal(mean=np.zeros(B), cov=Sigma, size=num_simulations)

    U = norm.cdf(Z)  # shape (N, B)
    p_b = [bucket_models[idx]["logit"].predict([1])[0] for idx in range(B)]
    Y_sim = (U <= p_b).astype(int)
    X_sim = np.zeros_like(Y_sim, dtype=float)  # same shape, to store final spend

    for b in range(B):
        active_rows = np.where(Y_sim[:, b] == 1)[0]
        if "gamma" in bucket_models[b]:
            X_sim[active_rows, b] = get_rvs_gamma(
                len(active_rows), bucket_models[b]["gamma"]
            )
        elif "lognorm" in bucket_models[b]:
            X_sim[active_rows, b] = get_rvs_lognorm(
                len(active_rows), bucket_models[b]["lognorm"]
            )
    if by_bucket:
        return X_sim
    return X_sim.sum(axis=1)


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


def find_best_fit_model(X):
    """
    Compare two-part Gamma vs two-part Lognormal by AIC.
    Return whichever has lower AIC.
    """
    # Fit each two-part model
    res_gamma = fit_two_part_gamma(X)
    res_lognorm = fit_two_part_lognorm(X)

    gamma_mod = res_gamma["gamma"]
    print("Gamma intercept:", gamma_mod.params)
    print("Gamma scale:", gamma_mod.scale)  # the dispersion

    lognorm_mod = res_lognorm["lognorm"]
    b0 = lognorm_mod.params[0]  # the intercept on the log scale
    sigma2 = lognorm_mod.scale  # residual variance on the log scale
    sigma = np.sqrt(sigma2)  # std dev on the log scale
    print(f"Lognormal intercept (b0): {b0}")
    print(f"Lognormal std dev on log scale (sigma): {sigma}")
    mean_lognormal = np.exp(b0 + 0.5 * sigma2)
    print(f"Implied mean in original units: {mean_lognormal}")
    median_lognormal = np.exp(b0)
    print(f"Implied median in original units: {median_lognormal}")

    aic_gamma = two_part_aic(res_gamma)
    aic_ln = two_part_aic(res_lognorm)
    print(f"Two-Part Gamma AIC:    {aic_gamma:.2f}")
    print(f"Two-Part Lognormal AIC:{aic_ln:.2f}")

    if aic_gamma < aic_ln:
        print("Best model: Two-Part Gamma")
        res_gamma["txns"] = X
        return res_gamma
    else:
        print("Best model: Two-Part Lognormal")
        res_lognorm["txns"] = X
        return res_lognorm


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


def get_rvs_lognorm(num_draws, lognorm_res):
    b0 = lognorm_res.params[0]
    sigma2 = lognorm_res.scale
    sigma = np.sqrt(sigma2)
    return lognorm.rvs(s=sigma, scale=np.exp(b0), size=num_draws)


def get_rvs_gamma(num_draws, gamma_res):
    beta0 = gamma_res.params[0]
    phi = gamma_res.scale
    shape = 1.0 / phi
    mean_ = np.exp(beta0)
    rate = shape / mean_
    return gamma.rvs(a=shape, scale=1.0 / rate, size=num_draws)


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


def fit_two_part_gamma(X):
    """
    Fits a two-part model to daily spend amounts X:
      1) Logistic regression for zero vs. > 0
      2) Gamma GLM for strictly positive data

    Returns:
        dict with keys:
            'logit': fitted logistic model (statsmodels LogitResults)
            'gamma': fitted gamma GLM (statsmodels GLMResults)
    """

    # -------------------------------------------------------
    # 1) Logistic Model: P(spend > 0)
    # -------------------------------------------------------
    # Binary indicator
    y_bin = (X > 0).astype(int)

    # Minimal "design matrix" = just intercept
    # (Or add other features if you prefer)
    X_bin = np.ones(len(X))  # 1-column of intercepts

    # Fit logistic
    logit_mod = sm.Logit(y_bin, X_bin, missing="drop").fit(disp=0)
    ll_logit = logit_mod.llf
    k_logit = logit_mod.df_model + 1

    # -------------------------------------------------------
    # 2) Gamma Model for positive spend
    # -------------------------------------------------------
    # Subset to positive amounts
    Xpos = X[X > 0]
    # Design matrix for gamma (again, just intercept for simplicity)
    Gpos = np.ones((len(Xpos), 1))  # shape (n,1) => a single column of 1s

    gamma_mod = sm.GLM(
        Xpos, Gpos, family=sm.families.Gamma(link=sm.genmod.families.links.log())
    ).fit(disp=0)
    ll_gamma = gamma_mod.llf
    k_gamma = gamma_mod.df_model + 1

    combined_loglike = ll_logit + ll_gamma
    total_params = k_logit + k_gamma

    return {
        "logit": logit_mod,
        "gamma": gamma_mod,
        "loglike": combined_loglike,
        "k": total_params,
    }


def fit_two_part_lognorm(X):
    """
    Two-part model for daily spend X:
      1) Logistic regression for P(X>0)
      2) OLS on log(X) for X>0 => Lognormal

    Returns:
        {
            'logit': <LogitResults>,
            'lognorm': <RegressionResults>,
            'loglike': combined_loglike,
            'k': total_num_params
        }
    """
    import statsmodels.api as sm
    import numpy as np

    # -------------------------------------------------------
    # 1) Logistic Model: P(spend > 0)
    # -------------------------------------------------------
    # Binary indicator
    y_bin = (X > 0).astype(int)
    X_bin = np.ones(len(X))
    logit_res = sm.Logit(y_bin, X_bin, missing="drop").fit(disp=0)
    ll_logit = logit_res.llf
    k_logit = logit_res.df_model + 1

    # -------------------------------------------------------
    # (2) OLS on log(X>0)
    # -------------------------------------------------------
    # Subset to positive amounts
    X_pos = X[X > 0]
    X_ln = np.log(X_pos)
    # Intercept only
    G_ln = np.ones((len(X_pos), 1))
    ln_res = sm.OLS(X_ln, G_ln).fit(disp=0)

    # To get the OLS log-likelihood:
    # statsmodels sets ln_res.llf. We'll use that directly.
    ll_ln = ln_res.llf
    k_ln = ln_res.df_model + 1

    combined_loglike = ll_logit + ll_ln
    total_params = k_logit + k_ln

    return {
        "logit": logit_res,
        "lognorm": ln_res,
        "loglike": combined_loglike,
        "k": total_params,
    }


def two_part_aic(two_part_res):
    """
    Compute the AIC for a two-part model result dictionary
    which has:
      'loglike': sum of log-likelihoods
      'k': total number of parameters
    """
    ll = two_part_res["loglike"]
    k = two_part_res["k"]
    return -2 * ll + 2 * k


def fit_models_for_each_bucket(df, bucket_thresholds):
    bucket_models = {}

    global_min = df.Date.min()
    global_max = df.Date.max()
    new_index = pd.date_range(start=global_min, end=global_max, freq="D")

    for idx, threshold in enumerate(bucket_thresholds):
        bucket_data = df[df["bucket"] == idx]
        if bucket_data.empty:
            continue

        # Summarize daily
        bucket_data_amounts = bucket_data.groupby("Date")["Amount"].sum().reset_index()

        # Reindex to fill missing days => 0 spend
        bucket_data_amounts.index = bucket_data_amounts["Date"]
        bucket_data_amounts = bucket_data_amounts.reindex(new_index)
        bucket_data_amounts["Date"] = bucket_data_amounts.index

        X = bucket_data_amounts["Amount"].fillna(0)

        print(f"\n----- Bucket {idx}, threshold {threshold} -----")
        best_two_part = find_best_fit_model(X)

        # best_two_part is a dictionary { 'logit':..., 'gamma' or 'lognorm':..., 'loglike':..., 'k':...}
        bucket_models[idx] = best_two_part

    return bucket_models


def get_bucket_correlations(df):
    day_bucket_counts = (
        df.groupby(["Date", "bucket"])["Amount"]
        .sum()  # or count() if you only need to know "did it happen?"
        .gt(0)  # True/False if sum>0
        .unstack(fill_value=0)
        .astype(int)
    )
    full_dates = pd.date_range(
        start=day_bucket_counts.index.min(), end=day_bucket_counts.index.max(), freq="D"
    )
    day_bucket_counts = day_bucket_counts.reindex(full_dates, fill_value=0)
    return day_bucket_counts.corr(method="spearman").values


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
    df_2024 = load_data(filter_by=">=2024-01-01")
    df_2023 = load_data(filter_by="<=2023-12-31")

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
    # bucket_thresholds = analyze_daily_patterns(df_filtered)
    # bucket_thresholds = bucket_thresholds[:6]  # keep the first 6 minima, for example
    bucket_thresholds = [
        166.49687688,
        218.5209009,
        286.15213213,
        312.16414414,
        348.58096096,
        421.41459459,
    ]  # just hardcoding for now

    # Create a 'bucket' column
    df_filtered["bucket"] = df_filtered["Amount"].apply(
        lambda x: kde_bucket_thresholds(x, bucket_thresholds)
    )

    # Fit models for each bucket
    bucket_thresholds.append(np.nan)  # we want to fit a model for the last bucket
    bucket_models = fit_models_for_each_bucket(df_filtered, bucket_thresholds)
    Sigma = get_bucket_correlations(df_filtered)

    # Example: run a Monte Carlo to see probabilities for multiple spend levels
    simulated_daily_spends = sophis_spend_likelihood_monte_carlo(Sigma, bucket_models)

    spend_amounts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    probability = [np.mean(simulated_daily_spends > amt) for amt in spend_amounts]
    for amt, prob in zip(spend_amounts, probability):
        print(f"Probability of spending more than {amt:.2f}: {prob:.4f}")

    sim_real_cdf_overlay(daily_spend["Amount"], simulated_daily_spends)


if __name__ == "__main__":
    main()
