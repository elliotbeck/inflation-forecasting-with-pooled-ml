from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from src.config import MIN_TRAIN_OBSERVATIONS, NUM_LAGS, ROLLING_WINDOW_SIZE
from src.utils.hrf.ewma_moments import get_ewma_moments
from src.utils.hrf.quadratic_inverse_shrinkage import QIS


def _forecast_step_multitarget(
    t: int,
    panel_df: pd.DataFrame,
    all_dates: pd.Index,
    feature_cols: list[str],
    rolling_window: int,
    min_train_observations: int,
    max_features: int,
) -> tuple[pd.Timestamp, dict[str, float]]:
    """
    Internal helper function for forecasting all countries at a given time step using a pooled Random Forest.

    This function builds one model using the full training window and predicts
    one-step-ahead values for all countries that have enough training history.

    Parameters:
        t (int): Current time step index.
        panel_df (pd.DataFrame): Long-format panel data with columns: ['Date', 'Country', 'YoY', lags, dummies].
        all_dates (pd.Index): Ordered list of all time indices.
        feature_cols (list[str]): List of feature column names (lags + country dummies).
        country_list (list[str]): List of all country column names in the original wide DataFrame.
        rolling_window (int): Number of months to include in the training window.
        min_train_observations (int): Minimum samples required for a country to be predicted.
        max_features (int): Number of features to consider at each split in the Random Forest.

    Returns:
        tuple: (current_date, {country: prediction, ...}) with predicted values for all eligible countries.
    """
    current_date = all_dates[t]
    train_window = all_dates[t - rolling_window : t]

    train_df = panel_df[panel_df["Date"].isin(train_window)].dropna(
        subset=feature_cols + ["YoY"]
    )
    test_df = panel_df[panel_df["Date"] == current_date].dropna(subset=feature_cols)

    if train_df.empty or test_df.empty:
        return current_date, {}

    # Only include countries with enough training data
    country_counts = train_df["Country"].value_counts()
    eligible_countries = country_counts[country_counts >= min_train_observations].index
    test_df = test_df[test_df["Country"].isin(eligible_countries)]

    if test_df.empty:
        return current_date, {}

    X_train = train_df[feature_cols]
    y_train = train_df["YoY"]
    X_test = test_df[feature_cols]

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        max_features=max_features,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return current_date, dict(zip(test_df["Country"], y_pred))


def _forecast_step_multitarget_hrf(
    t: int,
    panel_df: pd.DataFrame,
    all_dates: pd.Index,
    feature_cols: list[str],
    rolling_window: int,
    min_train_observations: int,
    max_features: int,
) -> tuple[pd.Timestamp, dict[str, float]]:
    """
    Forecast all countries at time t using a pooled Random Forest,
    then compute optimal per-country weights using shrinkage-based
    bias² + variance decomposition on residuals from that country only.
    """

    current_date = all_dates[t]
    train_window = all_dates[t - rolling_window : t]

    train_df = panel_df[panel_df["Date"].isin(train_window)].dropna(
        subset=feature_cols + ["YoY"]
    )
    test_df = panel_df[panel_df["Date"] == current_date].dropna(subset=feature_cols)

    if train_df.empty or test_df.empty:
        return current_date, {}

    # Only include countries with enough training data
    country_counts = train_df["Country"].value_counts()
    eligible_countries = country_counts[country_counts >= min_train_observations].index
    test_df = test_df[test_df["Country"].isin(eligible_countries)]

    if test_df.empty:
        return current_date, {}

    X_train = train_df[feature_cols]
    y_train = train_df["YoY"]
    X_test = test_df[feature_cols]

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        random_state=42,
        max_features=max_features,
    )
    model.fit(X_train, y_train)

    # In-sample predictions for training data from each tree
    n_estimators = len(model.estimators_)
    train_preds = np.column_stack([tree.predict(X_train) for tree in model.estimators_])

    predictions = {}

    for country in test_df["Country"].unique():
        X_test_c = X_test[test_df["Country"] == country]
        if X_test_c.empty:
            continue

        # Use only training samples from the same country
        country_train_mask = train_df["Country"] == country
        if country_train_mask.sum() < min_train_observations:
            continue

        y_train_c = y_train[country_train_mask]
        train_preds_c = train_preds[country_train_mask.to_numpy(), :]
        residuals = y_train_c.to_numpy()[:, np.newaxis] - train_preds_c

        # ewma_moments = get_ewma_moments(residuals, alpha=0.15, eps=0.01, bw=3)
        # mu = ewma_moments["mu_hat"]
        # Sigma = ewma_moments["sigma_hat2"]
        mu = np.mean(residuals, axis=0)
        residuals = pd.DataFrame(residuals)
        Sigma = QIS(residuals).values
        # Sigma, _ = ledoit_wolf(residuals)

        w = cp.Variable(n_estimators)
        objective = cp.Minimize(cp.square(w @ mu) + cp.quad_form(w, cp.psd_wrap(Sigma)))
        constraints = [cp.sum(w) == 1, cp.norm1(w) <= 2]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCS")

        if prob.status != cp.OPTIMAL or w.value is None:
            raise ValueError("Solver failed")

        test_preds = np.column_stack(
            [tree.predict(X_test_c) for tree in model.estimators_]
        )
        predictions[country] = (test_preds @ w.value).item()

    return current_date, predictions


def rolling_rf_pooled_forecast(
    yoy_data: pd.DataFrame,
    lags: int = NUM_LAGS,
    min_train_observations: int = MIN_TRAIN_OBSERVATIONS,
    max_features: Optional[int] = None,
    n_jobs: int = 20,
) -> pd.DataFrame:
    """
    Forecasts one-step-ahead YoY inflation for all countries using a pooled Random Forest,
    training a single model at each time step and predicting simultaneously for all eligible countries.

    Parameters:
        yoy_data (pd.DataFrame): Wide-format DataFrame with time as index and columns as countries (YoY inflation).
        lags (int): Number of lagged features to include for each country.
        min_train_observations (int): Minimum number of training samples required per country to be included in prediction.
        max_features (Optional[int]): Number of features to consider at each split in the RF.
                                      If None, defaults to one-third of the total features.
        n_jobs (int): Number of parallel jobs to run (default: -1 = use all available cores).

    Returns:
        pd.DataFrame: Forecasted values with the same shape and index as input `yoy_data`.
                      Missing values will remain NaN if the country couldn't be predicted.
    """
    forecast = pd.DataFrame(index=yoy_data.index, columns=yoy_data.columns, dtype=float)

    # Stack panel
    panel_df = yoy_data.stack().reset_index()
    panel_df.columns = ["Date", "Country", "YoY"]
    panel_df = panel_df.sort_values(["Country", "Date"])

    # Create lags
    for lag in range(1, lags + 1):
        panel_df[f"lag_{lag}"] = panel_df.groupby("Country")["YoY"].shift(lag)

    # One-hot encode countries
    country_dummies = pd.get_dummies(panel_df["Country"], prefix="country")
    panel_df = pd.concat([panel_df, country_dummies], axis=1)

    feature_cols = [f"lag_{i}" for i in range(1, lags + 1)] + list(
        country_dummies.columns
    )
    all_dates = panel_df["Date"].sort_values().unique()

    if max_features is None:
        max_features = int(np.floor(len(feature_cols) / 3))

    # Parallel time-step forecasting
    results = Parallel(n_jobs=n_jobs)(
        delayed(_forecast_step_multitarget)(
            t,
            panel_df,
            all_dates,
            feature_cols,
            ROLLING_WINDOW_SIZE,
            min_train_observations,
            max_features,
        )
        for t in range(ROLLING_WINDOW_SIZE, len(all_dates))
    )

    # Fill in forecast matrix
    for date, preds in results:
        for country, value in preds.items():
            forecast.loc[date, country] = value

    return forecast
